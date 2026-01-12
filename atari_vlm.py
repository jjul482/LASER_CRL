from typing import Tuple

import numpy as np
import torch

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from PIL import Image

DEFAULT_VLM_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

def atari_frame_to_pil(frame: np.ndarray) -> Image.Image:
    """
    Convert an Atari frame (H, W, C) or (C, H, W) numpy array into a PIL Image.
    Assumes values are either uint8 in [0, 255] or floats in [0, 1].
    """
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):  # CHW -> HWC
        frame = np.transpose(frame, (1, 2, 0))

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)

    return Image.fromarray(frame)

class AtariVLM:
    """
    Thin wrapper around a Hugging Face VLM (LLaVA v1.6 Mistral) that:
      - downloads the model snapshot (on first use)
      - runs inference on (task_prompt, atari_frame)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VLM_ID,
        device: str = None,
        cache_dir: str = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        model_id : HF repo id of the VLM.
        device   : "cuda", "cpu", or None for auto.
        cache_dir: Optional local path to store the snapshot.
        dtype    : torch dtype for the model.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_id = model_id

        print(f"[AtariVLM] Loading VLM '{model_id}' on {self.device}...")

        self.processor = LlavaNextProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        ).to(self.device)

        self.model.eval()
        print("[AtariVLM] Loaded.")

    @torch.inference_mode()
    def describe_state(
        self,
        task_prompt: str,
        atari_frame: np.ndarray,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """
        Run the VLM on a single Atari frame and task prompt.
        """
        image = atari_frame_to_pil(atari_frame)

        # LLaVA v1.6 expects a specific prompt format; simplest is an [INST] template
        prompt = f"[INST] <image>\n{task_prompt} [/INST]"

        inputs = self.processor(
            image=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
        )

        # Decode full sequence and strip the prompt part if desired
        full_text = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
        return full_text

    @torch.inference_mode()
    def snapshot_hidden(
        self,
        task_prompt: str,
        atari_frame: np.ndarray,
    ) -> Tuple[torch.Tensor, str]:
        """
        Take a snapshot of internal hidden states.

        Returns:
           (last_hidden_state, decoded_text)
           last_hidden_state: (1, seq_len, dim)
        """
        image = atari_frame_to_pil(atari_frame)
        prompt = f"[INST] <image>\n{task_prompt} [/INST]"

        inputs = self.processor(
            image=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, dim)
        # short generation:
        gen_ids = self.model.generate(**inputs, max_new_tokens=32)
        text = self.processor.decode(gen_ids[0], skip_special_tokens=True)

        return last_hidden.detach().cpu(), text