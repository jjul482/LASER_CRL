# laser_trainer.py

import json
from typing import Dict, Any, List

import torch
import numpy as np
import random

from laser import LASER, AtariVLM
from trainer_old import make_pixel_env, make_env

import json
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
import numpy as np
import gymnasium as gym
from hackatari.core import HackAtari

from stable_baselines3.common.env_checker import check_env

class Float32Wrapper(gym.ObservationWrapper):
    """
    Casts HackAtari int observations to float32.
    """
    def __init__(self, env):
        super().__init__(env)
        low, high, shape = env.observation_space.low, env.observation_space.high, env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def observation(self, obs):
        return obs.astype(np.float32)

def make_env(game_name, modifications=[], rewardfunc_path=None):
    """
    Create a HackAtari environment and optionally apply a modification.
    """
    # rewardfunc_path = "freeway_custom_reward.py"
    env = HackAtari(game_name, modifs=modifications, rewardfunc_path=rewardfunc_path, obs_mode="ori")
    if len(modifications) > 0:
        print(f"Applied modifications: {modifications}")
    
    # Custom reward?
    print("Obs space:", env.observation_space)
    env = Float32Wrapper(env)
    return env

def make_pixel_env(game_name, mods, mode='ori'):  # mode in {"ori","dqn"}
    # Pixels instead of object features:
    env = HackAtari(game_name, modifs=mods, obs_mode=mode, render_mode=None)  # pixels come via obs_mode
    venv = DummyVecEnv([lambda: env])
    venv = VecMonitor(venv)

    print("Obs space:", env.observation_space)

    # SB3’s CNN expects channel-first (C,H,W).
    # If the env is HWC (typical for "ori"), transpose it.
    if len(venv.observation_space.shape) == 3 and venv.observation_space.shape[-1] in (1, 3, 4):
        venv = VecTransposeImage(venv)  # HWC -> CHW

    return venv

def build_ppo_kwargs(args: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        learning_rate=args.get("learning_rate", 3e-4),
        n_steps=args.get("n_steps", 128),
        batch_size=args.get("batch_size", 256),
        n_epochs=args.get("n_epochs", 4),
        gamma=args.get("gamma", 0.99),
        gae_lambda=args.get("gae_lambda", 0.95),
        clip_range=args.get("clip_range", 0.2),
        ent_coef=args.get("ent_coef", 0.0),
        vf_coef=args.get("vf_coef", 0.5),
        max_grad_norm=args.get("max_grad_norm", 0.5),
        verbose=args.get("verbose", 1),
        tensorboard_log=args.get("tensorboard_log", None),
    )


def train(args: Dict[str, Any]) -> None:
    """
    LASER training

    Expected keys:
        - game
        - task_mods           : list[list[str]]
        - task_prompt         : single string, used for all tasks
        - total_timesteps_per_task
        - eval_episodes
        - output_path
        - VLM + LASER params
        - PPO params
    """

    game: str = args["game"]
    task_mods_list: List[List[str]] = args["task_mods"]
    task_prompt: str = args["task_prompt"] 

    total_timesteps_per_task: int = int(args.get("total_timesteps_per_task", 200_000))
    eval_episodes: int = int(args.get("eval_episodes", 20))
    output_path: str = args.get("output_path", f"./results_laser_{game}.json")

    device: str = args.get("device", "cuda")
    lambda_kl: float = float(args.get("lambda_kl", 1e-2))
    beta: float = float(args.get("beta", 1.0))
    vlm_model_id: str = args.get("vlm_model_id", "llava-hf/llava-v1.6-mistral-7b-hf")
    vlm_cache_dir: str = args.get("vlm_cache_dir", None)

    ppo_policy: str = args.get("ppo_policy", "CnnPolicy")
    ppo_kwargs = build_ppo_kwargs(args)

    # Seed everything
    seed = args.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    vlm = AtariVLM(
        model_id=vlm_model_id,
        device=device,
        cache_dir=vlm_cache_dir,
        dtype=torch.float16,
    )

    laser_agent = None
    all_results = []

    for task_idx, mods in enumerate(task_mods_list):
        print(f"\n[LASER] ===== Task {task_idx} / {len(task_mods_list)-1} =====")
        print(f"[LASER] Game: {game}")
        print(f"[LASER] Mods: {mods}")
        print(f"[LASER] Using single prompt: {task_prompt}")

        venv = make_pixel_env(game, mods)

        eval_env = make_env(game, mods)

        if laser_agent is None:
            print("[LASER] Initialising LASER agent...")
            laser_agent = LASER(
                env=venv,
                vlm=vlm,
                task_prompt=task_prompt,      # same prompt for all tasks
                lambda_kl=lambda_kl,
                beta=beta,
                device=device,
                ppo_policy=ppo_policy,
                ppo_kwargs=ppo_kwargs,
            )
        else:
            print("[LASER] Switching env (continual). Prompt unchanged.")
            laser_agent.env = venv
            laser_agent.model.set_env(venv)
        
        print(f"[LASER] Training for {total_timesteps_per_task} timesteps...")
        laser_agent.train(total_timesteps=total_timesteps_per_task)

        mean_rew, std_rew = laser_agent.evaluate(
            eval_env=eval_env,
            n_episodes=eval_episodes,
            render=False,
        )
        print(f"[LASER] Task {task_idx} evaluation: "
              f"mean={mean_rew:.2f} ± {std_rew:.2f}")

        all_results.append(
            {
                "task_index": task_idx,
                "mods": mods,
                "prompt": task_prompt,
                "mean_reward": mean_rew,
                "std_reward": std_rew
            }
        )

    output = {
        "config": args,
        "results": all_results,
    }
    print(f"[LASER] Writing results to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)