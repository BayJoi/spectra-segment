from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

BACKEND_MODULES: dict[str, Any] = {}


def _load_backend(backend: str) -> str:
    fork = "backend_amd_gpu" if backend == "amd" else "backend"
    BACKEND_MODULES["sam2"] = importlib.import_module(f"{fork}.models.ultralytics_backend")
    BACKEND_MODULES["matte"] = importlib.import_module(f"{fork}.models.matte_backend")
    BACKEND_MODULES["detector"] = importlib.import_module(f"{fork}.models.grounding_detector")
    BACKEND_MODULES["ram"] = importlib.import_module(f"{fork}.utils.ram_limit")
    return fork


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _image(height: int = 512, width: int = 512) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.stack(
        [
            xx / width * 255.0,
            yy / height * 255.0,
            (xx + yy) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    image[80:400, 120:380] = (220, 90, 40)
    return image


def _sam2(runs: int) -> tuple[bool, str]:
    backend = BACKEND_MODULES["sam2"].UltralyticsBackend(device="cuda")
    try:
        t0 = time.perf_counter()
        backend.load_model("sam2.1_b.pt")
        load_s = time.perf_counter() - t0
        image = _image()
        backend.set_image(image)
        for _ in range(runs):
            backend.predict(bboxes=[80, 80, 430, 430], points=None, labels=None, multimask_output=False)
        _sync()
        detail = f"load={load_s:.2f}s device={backend._device} encoder_cpu={backend._encoder_on_cpu} peak={_peak_gb():.2f}GB"
        return True, detail
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        backend.unload_model()


def _matte(runs: int) -> tuple[bool, str]:
    matte = BACKEND_MODULES["matte"]
    backend = matte.BiRefNetBackend(matte.MATTE_MODELS["birefnet-lite"], device="cuda")
    try:
        t0 = time.perf_counter()
        backend.load_model()
        load_s = time.perf_counter() - t0
        image = _image()
        for _ in range(runs):
            backend._forward([image])
        _sync()
        detail = f"load={load_s:.2f}s device={backend._device} peak={_peak_gb():.2f}GB"
        return True, detail
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        backend.unload_model()


def _detector(runs: int) -> tuple[bool, str]:
    backend = BACKEND_MODULES["detector"].GroundingDetector(device="cuda")
    try:
        t0 = time.perf_counter()
        backend.load_model("grounding-dino-tiny")
        load_s = time.perf_counter() - t0
        image = _image()
        for _ in range(runs):
            backend.detect(image, "person", confidence=0.1, max_detections=5)
        _sync()
        detail = f"load={load_s:.2f}s device={backend._actual_device} peak={_peak_gb():.2f}GB"
        return True, detail
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        backend.unload_model()


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU memory-limit stress check")
    parser.add_argument("--backend", choices=["amd", "nvidia"], default="amd")
    parser.add_argument("--cap", type=float, default=0.90)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--kind", choices=["all", "sam2", "matte", "detector"], default="all")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(f"FAIL: CUDA/ROCm is not available for the {args.backend} backend")
        return 1
    hip = bool(getattr(torch.version, "hip", None))
    cuda = bool(getattr(torch.version, "cuda", None))
    if args.backend == "amd" and not hip:
        print("FAIL: AMD backend selected, but this Python environment is not a ROCm build")
        return 1
    if args.backend == "nvidia" and (hip or not cuda):
        print("FAIL: NVIDIA backend selected, but this Python environment is not a CUDA build")
        return 1

    fork = _load_backend(args.backend)
    torch.cuda.set_per_process_memory_fraction(args.cap)
    if args.ram_mb > 0 and not BACKEND_MODULES["ram"].apply_process_memory_limit(args.ram_mb):
        print(f"FAIL: could not apply RAM cap {args.ram_mb} MB")
        return 1

    print(
        f"backend={args.backend} ({fork}) cap={args.cap:.2f} ram={args.ram_mb}MB "
        f"device={torch.cuda.get_device_name(0)} "
        f"total={torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
    )

    tests = {
        "sam2": ("SAM 2.1 Base", _sam2),
        "matte": ("BiRefNet Lite", _matte),
        "detector": ("Grounding DINO Tiny", _detector),
    }
    if args.kind == "all":
        selected = list(tests)
    else:
        selected = [args.kind]

    failures = 0
    for kind in selected:
        label, fn = tests[kind]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ok, detail = fn(args.runs)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        failures += 0 if ok else 1
        torch.cuda.empty_cache()

    print(f"RESULT: {len(selected) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
