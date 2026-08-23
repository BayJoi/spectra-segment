"""GPU / device health smoke test for Spectra Segment backends.

Verifies, per backend fork and per detected compute device:
  - app import
  - model availability (local only unless --download)
  - model load
  - encode runs under torch.inference_mode (regression guard)
  - post-encode memory footprint stays within budget
  - repeated encode does not grow the footprint (regression guard)
  - predict produces masks + timing

Results are printed to the console and written to a timestamped .log file
inside this checks/ folder. Exit code: 0 = all passed, 1 = failures.

Usage:
  python check_gpu_health.py [backend|backend_amd_gpu] [--model NAME]
                             [--runs N] [--download] [--force-cpu]

Run both forks:            check_gpu.bat
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(__file__).resolve().parent

FOOTPRINT_LIMIT_MB_GPU = 1024
FOOTPRINT_LIMIT_MB_CPU = 2500
GROWTH_LIMIT_MB = 64


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_logging(fork: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"check_gpu_health_{fork}_{stamp}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)

    class _Tee(_TeeStream):
        pass

    old = sys.stdout
    sys.stdout = _Tee(old, open(log_path, "a", encoding="utf-8"))
    print(f"log file: {log_path}")
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Spectra Segment GPU/device health check")
    parser.add_argument("fork", nargs="?", default="backend_amd_gpu",
                        choices=["backend", "backend_amd_gpu"],
                        help="backend fork to test")
    parser.add_argument("--model", default=None,
                        help="ultralytics model name (default: first available locally, else sam2.1_b.pt)")
    parser.add_argument("--runs", type=int, default=0,
                        help="predict repetitions for timing (default: 10 on GPU, 2 on CPU)")
    parser.add_argument("--download", action="store_true",
                        help="allow downloading the model if missing")
    parser.add_argument("--force-cpu", action="store_true",
                        help="run all checks on CPU regardless of GPU presence")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    log_path = setup_logging(args.fork)
    started = time.perf_counter()
    _log = logging.getLogger("health")

    print("=" * 62)
    print(f"Spectra Segment health check — fork: {args.fork}")
    print(f"python : {sys.version.split()[0]}  ({sys.executable})")
    print(f"mode   : NO downloads unless --download is passed")
    print("=" * 62)

    import platform
    import shutil

    print("-- diagnostics " + "-" * 48)
    print(f"platform      : {platform.platform()}")
    try:
        du = shutil.disk_usage(ROOT)
        print(f"disk free     : {du.free / 2**30:.1f} GB (repo drive)")
    except Exception:
        pass
    interesting = [
        k for k in os.environ
        if k.startswith("SPECTRA_")
        or k in (
            "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
            "TORCH_BLAS_PREFER_HIPBLASLT", "DISABLE_ADDMM_CUDA_LT",
            "MIOPEN_FIND_MODE", "MIOPEN_FIND_ENFORCE", "HSA_ENABLE_SDMA",
            "HF_HUB_DISABLE_PROGRESS_BARS", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
        )
    ]
    for k in sorted(interesting):
        print(f"env {k} = {os.environ[k]}")
    for k in ("SPECTRA_VRAM_FRACTION",):
        if not any(e == k for e in interesting):
            print(f"env {k} = <unset>")

    def log_exc(context: str) -> None:
        _log.error("---- traceback (%s) ----", context, exc_info=True)
        _log.error("---- end traceback ----")

    results: list[tuple[str, str, str]] = []

    def record(name: str, ok: bool | None, detail: str = "") -> bool | None:
        if ok is True:
            status = "PASS"
        elif ok is False:
            status = "FAIL"
        else:
            status = "SKIP"
        line = f"[{status}] {name}" + (f" — {detail}" if detail else "")
        print(line)
        results.append((name, status, detail))
        return ok

    # ---- imports -------------------------------------------------------
    try:
        mod = __import__(f"{args.fork}.main", fromlist=["app"])
        record("app imports", hasattr(mod, "app"))
    except Exception as e:
        log_exc("app imports")
        record("app imports", False, f"{type(e).__name__}: {e}")
        print(f"\nRESULT: FAIL (cannot import {args.fork})")
        return 1

    importlib_main = __import__(f"{args.fork}.models.ultralytics_backend", fromlist=["UltralyticsBackend"])
    UltralyticsBackend = importlib_main.UltralyticsBackend

    dev_mod = __import__(f"{args.fork}.utils.device", fromlist=["detect_device"])
    info = dev_mod.detect_device()
    device = "cpu" if args.force_cpu else info.torch_device
    is_gpu = device != "cpu"
    print(f"device : {info.description or device} ({device})")

    try:
        props = __import__("torch").cuda.get_device_properties(0)
        vram_total_mb = int(props.total_memory / 2**20)
        arch = getattr(props, "gcnArchName", "") or getattr(props, "name", "")
        print(f"gpu    : {props.name} · {vram_total_mb} MB · {arch}")
    except Exception:
        vram_total_mb = 0
        arch = ""

    footprint_limit = FOOTPRINT_LIMIT_MB_CPU if not is_gpu else max(
        512, min(FOOTPRINT_LIMIT_MB_GPU, int(vram_total_mb * 0.15)))

    # ---- model selection ----------------------------------------------
    mi = __import__(f"{args.fork}.models.ultralytics_backend", fromlist=["ULTRALYTICS_MODELS"])
    catalog = list(mi.ULTRALYTICS_MODELS.keys())
    model_name = args.model
    weights_dir = dev_mod.MODEL_WEIGHTS_DIR
    if not model_name:
        if "sam2.1_b.pt" in catalog and (weights_dir / "sam2.1_b.pt").exists():
            model_name = "sam2.1_b.pt"
        else:
            local_first = next((c for c in catalog if (weights_dir / c).exists()), None)
            model_name = local_first or ("sam2.1_b.pt" if "sam2.1_b.pt" in catalog else catalog[0])

    model_file = weights_dir / model_name
    if model_file.exists():
        record("model available locally", True, model_name)
    elif args.download:
        record("model available locally", None, f"{model_name} missing — --download set, load will fetch it")
    else:
        record("model available locally", False,
               f"{model_name} not in {weights_dir.name} (use --download to allow fetching)")
        print("\nRESULT: FAIL (no local model)")
        return 1

    # ---- load ----------------------------------------------------------
    t0 = time.perf_counter()
    try:
        be = UltralyticsBackend(device=device)
        be.load_model(model_name)
        load_s = time.perf_counter() - t0
        record("model loads", True, f"{load_s:.1f}s on {device}")
    except Exception as e:
        log_exc("model loads")
        record("model loads", False, f"{type(e).__name__}: {str(e)[:160]}")
        print("\nRESULT: FAIL")
        return 1

    # ---- encode + guards ----------------------------------------------
    import numpy as np
    import torch

    img = (np.random.rand(736, 736, 3) * 255).astype(np.uint8)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    try:
        be.set_image(img[:, :, ::-1])
        if is_gpu:
            torch.cuda.synchronize()
        enc_s = time.perf_counter() - t0
        print(f"[info] encode cold: {enc_s:.2f}s")
    except Exception as e:
        log_exc("encode")
        record("encode succeeds", False, f"{type(e).__name__}: {str(e)[:160]}")
        print("\nRESULT: FAIL")
        return 1
    record("encode succeeds", True, f"{enc_s:.2f}s")

    feats = getattr(be._predictor, "features", None)
    tensors = [t for t in (feats or {}).values() if isinstance(t, torch.Tensor)]
    all_inference = bool(tensors) and all(t.is_inference() for t in tensors)
    record("encode runs under inference_mode", all_inference,
           "" if all_inference else "features are NOT inference tensors — inference_mode guard was removed?")

    alloc_mb = int(torch.cuda.memory_allocated() / 2**20) if is_gpu and torch.cuda.is_available() \
        else int(torch.cuda.memory_allocated() / 2**20)
    limit = footprint_limit if is_gpu else FOOTPRINT_LIMIT_MB_CPU
    record(f"post-encode footprint < {limit} MB", alloc_mb < limit, f"{alloc_mb} MB")

    t0 = time.perf_counter()
    be.set_image(img[:, :, ::-1])
    if is_gpu:
        torch.cuda.synchronize()
    enc_warm = time.perf_counter() - t0
    alloc2_mb = int(torch.cuda.memory_allocated() / 2**20)
    grew = alloc2_mb - alloc_mb
    record("second encode does not grow footprint", grew <= GROWTH_LIMIT_MB,
           f"{alloc_mb} -> {alloc2_mb} MB (+{grew})")
    print(f"[info] encode warm: {enc_warm:.2f}s")

    # ---- predict -------------------------------------------------------
    runs = args.runs or (10 if is_gpu else 2)
    bbox = [[100, 100, 500, 500]]
    times = []
    last_err = ""
    masks_ok = False
    for _ in range(runs):
        try:
            t0 = time.perf_counter()
            r = be.predict(points=[], labels=[], bboxes=bbox, multimask_output=True)
            if is_gpu:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            m = r.get("masks") if isinstance(r, dict) else None
            masks_ok = masks_ok or (m is not None and len(m) > 0)
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            log_exc("predict")
    record("predict returns masks", masks_ok, last_err)
    if times:
        avg = sum(times) / len(times)
        print(f"[info] predict over {len(times)} runs: avg {avg:.1f} ms · min {min(times):.1f} ms")

    be.unload_model()

    # ---- summary -------------------------------------------------------
    elapsed = time.perf_counter() - started
    fails = [r for r in results if r[1] == "FAIL"]
    skips = [r for r in results if r[1] == "SKIP"]
    print("=" * 62)
    print(f"RESULT: {'ALL CHECKS PASSED' if not fails else 'FAILURES PRESENT'} "
          f"({len(results) - len(fails) - len(skips)} passed, {len(fails)} failed, {len(skips)} skipped) "
          f"in {elapsed:.1f}s")
    print(f"log: {log_path}")
    print("=" * 62)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
