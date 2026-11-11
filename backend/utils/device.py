from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .cuda_profile import (
    CudaProfile,
    configure_nvidia_runtime,
    get_cuda_profile,
    prepare_nvidia_environment,
)

LOGGER = logging.getLogger(__name__)

MODEL_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "model_weights"


@dataclass
class DeviceInfo:
    torch_device: str = "cpu"
    description: str = ""


def _detect_nvidia() -> tuple[CudaProfile | None, str]:
    profile = get_cuda_profile()
    if profile.available and profile.supported:
        LOGGER.debug(
            "NVIDIA CUDA device: %s (%s, sm_%s)",
            profile.name,
            profile.family,
            profile.capability_text,
        )
        return profile, profile.name
    return None, ""


def cpu_fallback_enabled() -> bool:
    raw = os.environ.get("SPECTRA_ALLOW_CPU_FALLBACK", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "disabled")


def detect_device() -> DeviceInfo:
    if os.environ.get("SPECTRA_FORCE_CPU") == "1":
        LOGGER.debug("CPU forced by memory profile (SPECTRA_FORCE_CPU=1)")
        return DeviceInfo(
            torch_device="cpu",
            description="CPU (forced by memory profile)",
        )

    profile, gpu_name = _detect_nvidia()
    if profile is not None:
        return DeviceInfo(
            torch_device="cuda:0",
            description=(
                f"NVIDIA CUDA - {gpu_name} "
                f"({profile.family}, sm_{profile.capability_text})"
            ),
        )

    detected = get_cuda_profile()
    if detected.available and not detected.supported:
        LOGGER.warning(
            "NVIDIA GPU %r (sm_%s) is not supported by this PyTorch build: %s. "
            "Install the matching CUDA wheel or set SPECTRA_FORCE_CPU=1.",
            detected.name,
            detected.capability_text,
            detected.support_reason,
        )
        return DeviceInfo(
            torch_device="cpu",
            description=(
                f"CPU (GPU unsupported: {detected.name}, sm_{detected.capability_text})"
            ),
        )

    if detected.support_reason not in ("CUDA is not available", "CPU was explicitly requested"):
        LOGGER.warning("NVIDIA unavailable: %s; using CPU", detected.support_reason)

    LOGGER.debug("No GPU acceleration detected, falling back to CPU")
    return DeviceInfo(
        torch_device="cpu",
        description="CPU (no acceleration)",
    )


__all__ = [
    "DeviceInfo",
    "configure_nvidia_runtime",
    "cpu_fallback_enabled",
    "detect_device",
    "get_cuda_profile",
    "prepare_nvidia_environment",
    "release_gpu_memory",
]


def release_gpu_memory() -> None:
    import gc
    import torch
    # Collect first: an OOM traceback can keep the failed inference's GPU
    # tensors referenced. Emptying the cache before those references are gone
    # returns nothing to the driver.
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            LOGGER.debug("cuda.synchronize() failed during release: %s", exc)
        for fn in ("empty_cache", "ipc_collect"):
            try:
                getattr(torch.cuda, fn)()
            except Exception as exc:
                LOGGER.debug("cuda.%s() failed during release: %s", fn, exc)
