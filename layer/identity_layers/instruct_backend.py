"""Frozen InstructPix2Pix internals used for layer discovery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass
class InstructScanSettings:
    model_id: str = "timbrooks/instruct-pix2pix"
    torch_dtype: str = "float16"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    seed: int = 1234


def dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return aliases[name.lower()]


def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


class InstructLayerBackend:
    def __init__(self, device: torch.device, settings: InstructScanSettings | None = None) -> None:
        self.device = device
        self.settings = settings or InstructScanSettings()
        self.pipe = self._load()

    def _load(self):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        if self.device.type != "cuda":
            raise RuntimeError("Identity-layer scans should run in the A6000 CUDA environment.")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=dtype_from_name(self.settings.torch_dtype),
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for module_name in ("vae", "text_encoder", "unet"):
            module = getattr(pipe, module_name, None)
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        return pipe

    def frozen_flags(self) -> dict[str, Any]:
        flags: dict[str, Any] = {}
        for module_name in ("vae", "text_encoder", "unet"):
            module = getattr(self.pipe, module_name, None)
            if module is None:
                continue
            params = list(module.parameters())
            flags[module_name] = {
                "num_parameters": int(sum(p.numel() for p in params)),
                "num_trainable_parameters": int(sum(p.numel() for p in params if p.requires_grad)),
                "all_frozen": all(not p.requires_grad for p in params),
            }
        return flags

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        if hasattr(self.pipe, "_encode_prompt"):
            return self.pipe._encode_prompt(prompt, self.device, 1, False)
        encoded = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        if isinstance(encoded, tuple):
            return encoded[0]
        return encoded

    def load_image_tensor(self, image_path: str | Path) -> torch.Tensor:
        return pil_to_tensor(Image.open(image_path).convert("RGB"), self.device)

    def encode_image_latent(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image = (image_tensor * 2.0 - 1.0).to(device=self.device, dtype=self.pipe.vae.dtype)
        latent = self.pipe.vae.encode(image).latent_dist.mode()
        return latent.to(dtype=self.pipe.unet.dtype)

    def resolved_timesteps(self) -> list[int]:
        self.pipe.scheduler.set_timesteps(self.settings.num_inference_steps, device=self.device)
        return [int(t.detach().cpu()) if torch.is_tensor(t) else int(t) for t in self.pipe.scheduler.timesteps]

    def timestep_by_index(self, timestep_index: int) -> torch.Tensor:
        self.pipe.scheduler.set_timesteps(self.settings.num_inference_steps, device=self.device)
        steps = self.pipe.scheduler.timesteps
        idx = min(max(0, int(timestep_index)), len(steps) - 1)
        return steps[idx]

    def fixed_noise(self, image_latent: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(self.settings.seed if seed is None else int(seed))
        return torch.randn(
            image_latent.shape,
            generator=generator,
            device=self.device,
            dtype=self.pipe.unet.dtype,
        ) * self.pipe.scheduler.init_noise_sigma

    def unet_forward(
        self,
        image_tensor: torch.Tensor,
        prompt: str,
        timestep_index: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        embedding = self.encode_prompt(prompt).detach()
        image_latent = self.encode_image_latent(image_tensor)
        timestep = self.timestep_by_index(timestep_index)
        fixed_noise = self.fixed_noise(image_latent, seed=seed)
        noisy = self.pipe.scheduler.scale_model_input(fixed_noise, timestep)
        sample = torch.cat([noisy.to(dtype=self.pipe.unet.dtype), image_latent.to(dtype=self.pipe.unet.dtype)], dim=1)
        prediction = self.pipe.unet(sample, timestep, encoder_hidden_states=embedding, return_dict=False)[0]
        return {
            "prediction": prediction,
            "image_latent": image_latent,
            "prompt_embedding": embedding,
            "timestep": timestep,
            "timestep_index": int(timestep_index),
            "fixed_noise": fixed_noise,
            "sample": sample,
        }
