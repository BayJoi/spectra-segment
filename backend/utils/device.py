from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

MODEL_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "model_weights"


@dataclass
class DeviceInfo:
    torch_device: str = "cpu"
    description: str = ""


def _detect_nvidia() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
    except Exception:
        pass
    return False, ""


def detect_device() -> DeviceInfo:
    if os.environ.get("SPECTRA_FORCE_CPU") == "1":
        LOGGER.debug("CPU forced by memory profile (SPECTRA_FORCE_CPU=1)")
        return DeviceInfo(
            torch_device="cpu",
            description="CPU (forced by memory profile)",
        )

    is_nvidia, gpu_name = _detect_nvidia()
    if is_nvidia:
        LOGGER.debug("NVIDIA GPU detected: %s", gpu_name)
        return DeviceInfo(
            torch_device="cuda:0",
            description=f"NVIDIA CUDA - {gpu_name}",
        )

    LOGGER.debug("No GPU acceleration detected, falling back to CPU")
    return DeviceInfo(
        torch_device="cpu",
        description="CPU (no acceleration)",
    )


def release_gpu_memory() -> None:
    import gc
    import torch
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
    gc.collect()
