"""Forward-hook helpers for activation inventory and pooled extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .pooling import activation_stats, first_tensor, pool_activation


def resolve_module(pipe: Any, module_path: str) -> torch.nn.Module:
    """Resolve module paths like ``unet.down_blocks.0`` from a diffusers pipe."""

    if module_path == "vae_image_latent":
        raise ValueError("vae_image_latent is a pseudo-layer and cannot be hooked")
    parts = module_path.split(".")
    obj: Any = pipe
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(f"{module_path!r} did not resolve to a torch.nn.Module")
    return obj


@dataclass
class HookRecord:
    layer_name: str
    module_path: str
    module_type: str
    output_shape: list[int] | None = None
    output_dtype: str | None = None
    output_device: str | None = None
    number_of_elements: int | None = None
    estimated_memory_mb: float | None = None
    error: str | None = None


class ShapeHookCapture:
    """Capture output shape metadata without retaining activation tensors."""

    def __init__(self, pipe: Any, module_paths: list[str]) -> None:
        self.pipe = pipe
        self.module_paths = module_paths
        self.records: dict[str, HookRecord] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "ShapeHookCapture":
        for module_path in self.module_paths:
            try:
                module = resolve_module(self.pipe, module_path)
                record = HookRecord(
                    layer_name=module_path,
                    module_path=module_path,
                    module_type=module.__class__.__name__,
                )
                self.records[module_path] = record

                def make_hook(path: str) -> Callable[..., None]:
                    def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
                        tensor = first_tensor(output)
                        if tensor is None:
                            self.records[path].error = "No tensor found in module output"
                            return
                        rec = self.records[path]
                        rec.output_shape = list(tensor.shape)
                        rec.output_dtype = str(tensor.dtype)
                        rec.output_device = str(tensor.device)
                        rec.number_of_elements = int(tensor.numel())
                        rec.estimated_memory_mb = float(tensor.numel() * tensor.element_size() / (1024**2))

                    return hook

                self._handles.append(module.register_forward_hook(make_hook(module_path)))
            except Exception as error:
                self.records[module_path] = HookRecord(
                    layer_name=module_path,
                    module_path=module_path,
                    module_type="unresolved",
                    error=repr(error),
                )
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@dataclass
class PooledActivation:
    layer_name: str
    module_path: str
    pooled: torch.Tensor
    stats: dict[str, float] = field(default_factory=dict)
    output_shape: list[int] | None = None
    output_dtype: str | None = None


class PooledHookCapture:
    """Capture pooled activations on CPU for a small set of layers."""

    def __init__(self, pipe: Any, module_paths: list[str]) -> None:
        self.pipe = pipe
        self.module_paths = module_paths
        self.activations: dict[str, PooledActivation] = {}
        self.failures: dict[str, str] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "PooledHookCapture":
        for module_path in self.module_paths:
            try:
                module = resolve_module(self.pipe, module_path)

                def make_hook(path: str) -> Callable[..., None]:
                    def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
                        tensor = first_tensor(output)
                        if tensor is None:
                            self.failures[path] = "No tensor found in module output"
                            return
                        with torch.no_grad():
                            pooled = pool_activation(tensor).detach().cpu()
                            stats = activation_stats(tensor)
                        self.activations[path] = PooledActivation(
                            layer_name=path,
                            module_path=path,
                            pooled=pooled,
                            stats=stats,
                            output_shape=list(tensor.shape),
                            output_dtype=str(tensor.dtype),
                        )

                    return hook

                self._handles.append(module.register_forward_hook(make_hook(module_path)))
            except Exception as error:
                self.failures[module_path] = repr(error)
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
