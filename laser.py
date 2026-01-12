import math
from typing import Callable, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance
from atari_vlm import AtariVLM

# ---------------------------------------------------------------------
# Utility: convert Atari frame (numpy) to PIL for the VLM
# ---------------------------------------------------------------------
from PIL import Image

class LaserPPO(PPO):
    """
    PPO with an additional behavioural semantic KL loss term:

        L_beh = lambda_kl * E_t[ D_KL( pi_theta(.|s_t) || q(.|s_t) ) ]

    where q(.|s_t) is a VLM-derived action prior.
    """

    def __init__(
        self,
        *args,
        lambda_kl: float,
        vlm_prior_fn: Callable[[torch.Tensor], torch.Tensor],
        **kwargs,
    ):
        """
        Args:
            lambda_kl: weight of the behavioural semantic KL term.
            vlm_prior_fn: function mapping obs_tensor -> q_probs
                          obs_tensor: (batch, ...) on self.device
                          q_probs:    (batch, n_actions), valid prob. dist.
        """
        super().__init__(*args, **kwargs)
        self.lambda_kl = lambda_kl
        self.vlm_prior_fn = vlm_prior_fn

    def _compute_reverse_kl(
        self,
        obs: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reverse KL: D_KL( pi(.|s) || q(.|s) )
        for a batch of observations.

        action_probs: (batch, n_actions) from pi_theta.
        Returns: (batch,) tensor of KL values.
        """
        # q_probs from VLM prior; no gradients through VLM.
        with torch.no_grad():
            q_probs = self.vlm_prior_fn(obs)  # (batch, n_actions)

        eps = 1e-8
        p = torch.clamp(action_probs, eps, 1.0)
        q = torch.clamp(q_probs, eps, 1.0)

        log_p = torch.log(p)
        log_q = torch.log(q)

        kl = torch.sum(p * (log_p - log_q), dim=-1)  # (batch,)
        return kl

    def train(self) -> None:  # adapted from SB3 PPO, with extra KL term
        """
        Override PPO.train() to inject the behavioural semantic KL loss
        into the policy objective.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # get data from buffer
        rollout_buffer = self.rollout_buffer
        assert rollout_buffer is not None

        # Normalize advantages if needed
        advantages = rollout_buffer.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            for rollout_data in rollout_buffer.get(self.batch_size):
                # Convert samples to tensors
                obs = rollout_data.observations
                actions = rollout_data.actions
                old_values = rollout_data.values
                old_log_prob = rollout_data.log_probs
                adv = advantages[rollout_data.indices]

                # Evaluate current policy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    obs, actions
                )
                # PPO ratio
                log_ratio = log_prob - old_log_prob
                ratio = torch.exp(log_ratio)

                # Policy loss
                adv = adv.unsqueeze(-1)
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + torch.clamp(
                        values - old_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Behavioural semantic KL loss
                # Get discrete action distribution
                dist = self.policy.get_distribution(obs)
                # assumes Categorical
                action_probs = dist.distribution.probs  # (batch, n_actions)

                kl_vals = self._compute_reverse_kl(obs, action_probs)
                beh_loss = self.lambda_kl * kl_vals.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                    + beh_loss
                )

                # Optimize
                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

class LASER:
    """
    LASER controller.
    """

    def __init__(
        self,
        env: VecEnv,
        vlm: AtariVLM,
        task_prompt: str,
        lambda_kl: float = 1e-2,
        beta: float = 1.0,
        device: str = "cuda",
        ppo_policy: str = "CnnPolicy",
        ppo_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            env        : VecEnv over HackAtari pixels (VecTransposeImage already applied).
            vlm        : AtariVLM instance (frozen).
            task_prompt: textual description of the current task.
            lambda_kl  : weight of the KL penalty.
            beta       : sharpness parameter for q(a|s) ∝ exp(beta * g_act(z_t,a)).
            device     : "cuda" or "cpu".
            ppo_policy : SB3 policy type, e.g. "CnnPolicy" for pixel inputs.
            ppo_kwargs : Extra kwargs passed to LaserPPO (e.g., learning_rate, n_steps).
        """
        if ppo_kwargs is None:
            ppo_kwargs = {}

        self.env = env
        self.vlm = vlm
        self.task_prompt = task_prompt
        self.lambda_kl = lambda_kl
        self.beta = beta
        self.device = torch.device(device)

        # Number of discrete actions
        assert env.action_space.is_discrete, "LASER currently assumes discrete actions."
        self.n_actions = env.action_space.n

        # Build g_act: action scoring head on top of VLM embedding
        self._build_action_head()

        # Prior function used inside LaserPPO
        def vlm_prior_fn(obs_tensor: torch.Tensor) -> torch.Tensor:
            """
            obs_tensor: (batch, C, H, W) on model.device
            Returns: (batch, n_actions) VLM-derived action prior q(a|s).
            """
            return self._vlm_prior_from_obs(obs_tensor)

        # Instantiate LaserPPO
        self.model = LaserPPO(
            policy=ppo_policy,
            env=self.env,
            lambda_kl=lambda_kl,
            vlm_prior_fn=vlm_prior_fn,
            device=device,
            **ppo_kwargs,
        )

    # -------------------------------
    # Internal: build g_act head
    # -------------------------------
    def _build_action_head(self) -> None:
        """
        Initialise g_act by running a single frame through AtariVLM to
        determine the embedding dimension, then define a small MLP head.
        """
        # Sample one observation from the VecEnv
        obs = self.env.reset()  # typically (n_envs, C, H, W)
        if isinstance(obs, tuple):  # SB3 vec envs sometimes return (obs, info)
            obs = obs[0]
        # take first env
        if obs.ndim == 4:
            frame = obs[0]
        else:
            frame = obs

        # Convert to HWC for VLM
        frame_np = np.array(frame)
        if frame_np.ndim == 3 and frame_np.shape[0] in (1, 3, 4):
            frame_np = np.transpose(frame_np, (1, 2, 0))

        with torch.inference_mode():
            last_hidden, _ = self.vlm.snapshot_hidden(self.task_prompt, frame_np)
        # last_hidden: (1, seq_len, dim)
        # simple token-mean pooling
        z = last_hidden.mean(dim=1)  # (1, dim)
        embed_dim = z.shape[-1]

        # Small MLP head: g_act(z) -> logits over actions
        self.g_act = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.n_actions),
        ).to(self.device)

        # NOTE: By default this head is NOT wired to PPO's optimizer.
        # If you want to jointly train g_act with the KL loss, you can:
        #   - add its parameters into self.model.policy.optimizer, or
        #   - move g_act into self.model.policy as a submodule.
        # For now, we treat g_act as a fixed prior head (e.g., pre-trained elsewhere).

        for p in self.g_act.parameters():
            p.requires_grad = False  # keep prior fixed by default

    # compute q(a|s) batch
    @torch.inference_mode()
    def _vlm_prior_from_obs(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of observations from the VecEnv, compute VLM-derived
        action priors q(a|s) for each frame.

        obs_tensor: (batch, C, H, W) on the same device as PPO.
        Returns: (batch, n_actions) probs.
        """
        obs_np = obs_tensor.detach().cpu().numpy()

        # Convert CHW -> HWC if needed
        if obs_np.ndim == 4 and obs_np.shape[1] in (1, 3, 4):
            frames = np.transpose(obs_np, (0, 2, 3, 1))
        else:
            frames = obs_np

        batch_q = []
        for frame in frames:
            # frame: (H, W, C)
            last_hidden, _ = self.vlm.snapshot_hidden(self.task_prompt, frame)
            # (1, seq_len, dim) -> (1, dim)
            z = last_hidden.mean(dim=1).to(self.device)  # pooled embedding
            logits = self.g_act(z)  # (1, n_actions)
            # Risk_VLM(s,a) = -g_act(z,a)
            # q (prop) exp(-beta * Risk) = exp(beta * g_act)
            scaled_logits = self.beta * logits
            q = torch.softmax(scaled_logits, dim=-1)  # (1, n_actions)
            batch_q.append(q.squeeze(0))

        q_batch = torch.stack(batch_q, dim=0)  # (batch, n_actions)
        return q_batch

    def train(self, total_timesteps: int, **learn_kwargs) -> None:
        """
        Train LASER (PPO + semantic regulariser) for a number of timesteps.
        """
        self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)

    def evaluate(
        self,
        eval_env,
        n_episodes: int = 10,
        render: bool = False,
    ) -> Tuple[float, float]:
        """
        Evaluate the current LASER policy on a given (non-vec) env.
        """
        import gymnasium as gym

        rewards = []

        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            ep_rew = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_rew += float(reward)
                done = terminated or truncated
                if render:
                    eval_env.render()
            rewards.append(ep_rew)

        mean_rew = float(np.mean(rewards))
        std_rew = float(np.std(rewards))
        return mean_rew, std_rew