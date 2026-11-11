from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CHECK_DIR = Path(__file__).resolve().parent

FORK_FOR_BACKEND = {
    "amd": "backend_amd_gpu",
    "nvidia": "backend",
    "cpu": "backend",
}

EXPECTED_CATALOG = {
    "sam2": 8,
    "sam3": 1,
    "grounding": 7,
    "yoloe": 7,
    "matte": 10,
}
EXPECTED_TOTAL = sum(EXPECTED_CATALOG.values())


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def _log_path(tag: str) -> Path:
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CHECK_DIR / f"check_gpu_health_{tag}_{stamp}.log"


def setup_logging(tag: str) -> Path:
    path = _log_path(tag)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    sys.stdout = _Tee(sys.__stdout__, open(path, "a", encoding="utf-8"))
    return path


def _running_as_child() -> bool:
    return "--single" in sys.argv or "--no-log" in sys.argv


def _clean_detail(value: Any, limit: int = 280) -> str:
    text = " ".join(str(value).replace("\r", " ").replace("\n", " ").split())
    return text[:limit]


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _peak_mb() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.max_memory_allocated() / 2**20)
    except Exception:
        return 0


def _allocated_mb() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.memory_allocated() / 2**20)
    except Exception:
        return 0


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _image(height: int = 810, width: int = 1080, *, four_k: bool = False) -> Any:
    import cv2
    import numpy as np

    if four_k:
        height, width = 2160, 3840
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.stack(
        [
            (xx / max(1, width - 1) * 255.0),
            (yy / max(1, height - 1) * 255.0),
            ((xx + yy) % 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    # A few synthetic shapes give the detectors something with edges to inspect.
    cv2.rectangle(image, (width // 8, height // 5), (width // 3, height * 4 // 5), (40, 110, 220), -1)
    cv2.circle(image, (width * 3 // 4, height // 2), min(width, height) // 8, (220, 80, 40), -1)
    return image


def _read_image(path: str | None) -> Any:
    if not path:
        return _image()
    import cv2

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {candidate}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _set_cache_environment(weights_dir: Path) -> None:
    defaults = {
        "HF_HOME": weights_dir / "hf_cache",
        "HF_HUB_CACHE": weights_dir / "hf_cache" / "hub",
        "TRANSFORMERS_CACHE": weights_dir / "hf_cache" / "hub",
        "HUGGINGFACE_HUB_CACHE": weights_dir / "hf_cache" / "hub",
        "HF_MODULES_CACHE": weights_dir / "hf_cache" / "modules",
        "HF_DATASETS_CACHE": weights_dir / "hf_datasets",
        "TORCH_HOME": weights_dir / "torch_cache",
        "ULTRALYTICS_HOME": weights_dir,
        "YOLO_CONFIG_DIR": weights_dir,
        "MPLCONFIGDIR": weights_dir / "matplotlib",
        "XDG_CACHE_HOME": weights_dir / "cache",
        "XDG_CONFIG_HOME": weights_dir / "config",
        "XDG_DATA_HOME": weights_dir / "data",
        "MIOPEN_USER_DB_PATH": weights_dir / "miopen",
        "MIOPEN_CUSTOM_CACHE_DIR": weights_dir / "miopen" / "cache",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    for path in (
        weights_dir / "hf_cache",
        weights_dir / "hf_cache" / "hub",
        weights_dir / "torch_cache",
        weights_dir / "matplotlib",
        weights_dir / "cache",
        weights_dir / "config",
        weights_dir / "data",
        weights_dir / "miopen" / "cache",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _choose_backend(requested: str) -> str:
    import torch

    hip = bool(getattr(torch.version, "hip", None))
    cuda = bool(getattr(torch.version, "cuda", None))
    if requested == "auto":
        if hip:
            return "amd"
        if cuda and torch.cuda.is_available():
            return "nvidia"
        return "cpu"
    if requested == "amd" and not hip:
        raise RuntimeError(
            "AMD backend requested, but this Python environment is not a ROCm build. "
            "Run checks\\check_gpu_amd.bat with backend_amd_gpu\\.venv."
        )
    if requested == "nvidia" and (hip or not cuda):
        raise RuntimeError(
            "NVIDIA backend requested, but this Python environment is not a CUDA build. "
            "Run checks\\check_gpu_nvidia.bat with backend\\.venv."
        )
    return requested


def _import_fork(fork: str) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    modules = {
        "app": importlib.import_module(f"{fork}.main"),
        "device": importlib.import_module(f"{fork}.utils.device"),
        "sam2": importlib.import_module(f"{fork}.models.ultralytics_backend"),
        "sam3": importlib.import_module(f"{fork}.models.sam3_backend"),
        "grounding": importlib.import_module(f"{fork}.models.grounding_detector"),
        "yoloe": importlib.import_module(f"{fork}.models.yoloe_detector"),
        "matte": importlib.import_module(f"{fork}.models.matte_backend"),
    }
    return modules


def _catalog(modules: dict[str, Any]) -> dict[str, list[str]]:
    sam2 = list(modules["sam2"].ULTRALYTICS_MODELS)
    sam3 = [modules["sam3"].SAM3_MODEL]
    grounding = list(modules["grounding"].GROUNDING_DETECTORS)
    yoloe = list(modules["yoloe"].YOLOE_MODELS)
    matte = list(modules["matte"].MATTE_MODELS)
    return {"sam2": sam2, "sam3": sam3, "grounding": grounding, "yoloe": yoloe, "matte": matte}


def _missing_weights(modules: dict[str, Any], catalog: dict[str, list[str]]) -> list[str]:
    weights_dir = modules["device"].MODEL_WEIGHTS_DIR
    missing: list[str] = []
    for name in catalog["sam2"]:
        if not (weights_dir / name).is_file():
            missing.append(f"sam2/{name}")
    for name in catalog["sam3"]:
        if not (weights_dir / name).is_file():
            missing.append(f"sam3/{name}")
    for name in catalog["yoloe"]:
        encoder = "mobileclip2_b.ts" if name.startswith("yoloe-26") else "mobileclip_blt.ts"
        if not (weights_dir / f"{name}.pt").is_file():
            missing.append(f"yoloe/{name}.pt")
        if not (weights_dir / encoder).is_file():
            missing.append(f"yoloe/{encoder}")
    grounding = modules["grounding"]
    for name in catalog["grounding"]:
        hf_id = grounding.HF_MODEL_IDS[name]
        revision = grounding.HF_MODEL_REVISIONS[name]
        if not grounding.hf_weights_downloaded(
            hf_id, cache_root=weights_dir / "hf_cache", revision=revision
        ):
            missing.append(f"grounding/{name}")
    matte = modules["matte"]
    for name in catalog["matte"]:
        spec = matte.MATTE_MODELS[name]
        if not matte._is_downloaded(spec):
            missing.append(f"matte/{name}")
    return missing


def _gpu_description(backend: str) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "backend": backend,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "name": str(props.name),
                "total_mb": int(props.total_memory / 2**20),
                "major": int(getattr(props, "major", 0)),
                "minor": int(getattr(props, "minor", 0)),
                "arch": getattr(props, "gcnArchName", "") or "",
            }
        )
    return info


def _run_model(
    modules: dict[str, Any],
    kind: str,
    name: str,
    device: str,
    image: Any,
    runs: int,
    *,
    allow_download: bool,
) -> dict[str, Any]:
    """Load and exercise exactly one catalogue entry.

    Returns a serialisable result. The caller decides whether a zero-detection
    result is a failure; this function only fails when the model raises or when a
    segmentation/matte result has the wrong shape.
    """
    import numpy as np
    import torch

    started = time.perf_counter()
    load_s = 0.0
    encode_s = 0.0
    infer_s = 0.0
    peak_mb = 0
    allocated_mb = 0
    backend = None
    try:
        if kind == "sam2":
            if not (modules["device"].MODEL_WEIGHTS_DIR / name).is_file() and not allow_download:
                raise FileNotFoundError(f"missing {name}")
            backend = modules["sam2"].UltralyticsBackend(device=device)
            t0 = time.perf_counter()
            backend.load_model(name)
            load_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            backend.set_image(image)
            _sync()
            encode_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            masks = 0
            for _ in range(max(1, runs)):
                result = backend.predict(
                    points=None,
                    labels=None,
                    bboxes=[10, 180, 800, 830],
                    multimask_output=True,
                )
                masks = int(len(result.get("masks", [])))
            _sync()
            infer_s = time.perf_counter() - t0
            allocated_mb = _allocated_mb()
            peak_mb = _peak_mb()
            if masks < 1:
                raise RuntimeError("SAM 2 returned no masks")
            detail = f"{masks} mask(s), allocated {allocated_mb} MB"

        elif kind == "sam3":
            if not (modules["device"].MODEL_WEIGHTS_DIR / name).is_file() and not allow_download:
                raise FileNotFoundError(f"missing {name}")
            backend = modules["sam3"].SAM3Backend(device=device)
            t0 = time.perf_counter()
            backend.load_model(name)
            load_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            backend.set_image(image)
            _sync()
            encode_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            result = backend.predict_text("bus", confidence=0.25)
            _sync()
            infer_s = time.perf_counter() - t0
            masks = int(len(result.get("masks", [])))
            allocated_mb = _allocated_mb()
            peak_mb = _peak_mb()
            detail = f"{masks} mask(s), allocated {allocated_mb} MB"

        elif kind in ("grounding", "yoloe"):
            mod = modules[kind]
            cls = mod.GroundingDetector if kind == "grounding" else mod.YOLEDetector
            backend = cls(device=device)
            t0 = time.perf_counter()
            backend.load_model(name)
            load_s = time.perf_counter() - t0
            peak_mb = 0
            _reset_peak()
            t0 = time.perf_counter()
            detections = backend.detect(
                image, "bus, person", confidence=0.05, max_detections=20
            )
            _sync()
            infer_s = time.perf_counter() - t0
            allocated_mb = _allocated_mb()
            peak_mb = _peak_mb()
            detail = f"{len(detections)} detection(s), allocated {allocated_mb} MB"

        elif kind == "matte":
            spec = modules["matte"].MATTE_MODELS[name]
            backend = modules["matte"].BiRefNetBackend(spec, device=device)
            t0 = time.perf_counter()
            backend.load_model()
            load_s = time.perf_counter() - t0
            peak_mb = 0
            _reset_peak()
            t0 = time.perf_counter()
            alpha = backend.matte(image)
            _sync()
            infer_s = time.perf_counter() - t0
            allocated_mb = _allocated_mb()
            peak_mb = _peak_mb()
            if not isinstance(alpha, np.ndarray) or alpha.shape[:2] != image.shape[:2]:
                raise RuntimeError(f"matte returned wrong shape: {getattr(alpha, 'shape', None)}")
            if not np.isfinite(alpha).all():
                raise RuntimeError("matte returned non-finite alpha")
            detail = f"{alpha.shape[1]}x{alpha.shape[0]} alpha, allocated {allocated_mb} MB"
        else:
            raise ValueError(f"unknown model kind: {kind}")

        return {
            "ok": True,
            "kind": kind,
            "name": name,
            "load_s": round(load_s, 3),
            "encode_s": round(encode_s, 3),
            "infer_s": round(infer_s, 3),
            "allocated_mb": allocated_mb,
            "peak_mb": peak_mb,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "detail": detail,
        }
    except Exception as exc:
        return {
            "ok": False,
            "kind": kind,
            "name": name,
            "load_s": round(load_s, 3),
            "encode_s": round(encode_s, 3),
            "infer_s": round(infer_s, 3),
            "allocated_mb": allocated_mb,
            "peak_mb": peak_mb,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "detail": _clean_detail(f"{type(exc).__name__}: {exc}"),
            "traceback": traceback.format_exc(limit=8),
        }
    finally:
        if backend is not None:
            try:
                backend.unload_model()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _child_main(args: argparse.Namespace) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        modules = _import_fork(FORK_FOR_BACKEND[args.backend])
        _set_cache_environment(modules["device"].MODEL_WEIGHTS_DIR)
        image = _read_image(args.image)
        result = _run_model(
            modules,
            args.single,
            args.name,
            "cuda:0" if args.backend in ("amd", "nvidia") else "cpu",
            image,
            args.runs,
            allow_download=args.download,
        )
        print("SPECTRA_MODEL_RESULT=" + json.dumps(result))
        return 0 if result.get("ok") else 1
    except Exception as exc:
        result = {
            "ok": False,
            "kind": args.single,
            "name": args.name,
            "detail": _clean_detail(f"{type(exc).__name__}: {exc}"),
            "traceback": traceback.format_exc(limit=8),
        }
        print("SPECTRA_MODEL_RESULT=" + json.dumps(result))
        return 1


def _spawn_model(
    backend: str,
    kind: str,
    name: str,
    *,
    image_path: str | None,
    runs: int,
    allow_download: bool,
    timeout: int = 900,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(Path(__file__).resolve()),
        "--single",
        kind,
        "--name",
        name,
        "--backend",
        backend,
        "--runs",
        str(runs),
        "--no-log",
    ]
    if image_path:
        command.extend(["--image", str(Path(image_path).resolve())])
    if allow_download:
        command.append("--download")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "kind": kind,
            "name": name,
            "detail": f"timed out after {timeout}s",
        }
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("SPECTRA_MODEL_RESULT="):
            try:
                payload = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                payload = None
    if payload is None:
        payload = {
            "ok": False,
            "kind": kind,
            "name": name,
            "detail": _clean_detail(
                "child produced no result; stderr=" + (proc.stderr or "")[-400:]
            ),
        }
    if proc.returncode != 0 and payload.get("ok"):
        payload["ok"] = False
        payload["detail"] = _clean_detail(f"child exited {proc.returncode}: {payload.get('detail', '')}")
    return payload


def _record(checks: list[Check], name: str, ok: bool | None, detail: str = "") -> None:
    if ok is True:
        status = "PASS"
    elif ok is False:
        status = "FAIL"
    else:
        status = "SKIP"
    check = Check(name, status, detail)
    checks.append(check)
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def _print_banner(backend: str) -> None:
    print("=" * 74)
    print("Spectra Segment GPU health check")
    print(f"backend : {backend}")
    print(f"python  : {sys.version.split()[0]} ({sys.executable})")
    print(f"root    : {ROOT}")
    print("downloads: disabled unless --download is passed")
    print("=" * 74)
    if backend == "nvidia":
        print("[WARN] NVIDIA physical hardware is UNTESTED by this project.")
        print("[WARN] This run is the first runtime evidence for the exact NVIDIA stack.")
    elif backend == "amd":
        print("[INFO] Project runtime evidence exists only for AMD Radeon RX 6600 / gfx1032.")
        print("[INFO] Other AMD targets are exercised only if this run is executed on that card.")


def _nvidia_smi() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _run_smoke(
    modules: dict[str, Any],
    kind: str,
    name: str,
    device: str,
    image: Any,
    runs: int,
    allow_download: bool,
    checks: list[Check],
) -> None:
    import torch

    _reset_peak()
    result = _run_model(modules, kind, name, device, image, runs, allow_download=allow_download)
    if not result.get("ok"):
        _record(checks, f"smoke {kind}/{name}", False, result.get("detail", ""))
        return
    _record(
        checks,
        f"smoke {kind}/{name}",
        True,
        f"load {result['load_s']:.2f}s, encode {result.get('encode_s', 0):.2f}s, "
        f"infer {result['infer_s']:.2f}s, peak {result['peak_mb']} MB, {result['detail']}",
    )
    if kind == "sam2":
        _record(checks, "SAM 2 smoke returned masks", True, result["detail"])
    if kind == "sam3":
        _record(checks, "SAM 3 smoke completed", True, result["detail"])
    if kind in ("grounding", "yoloe"):
        _record(checks, f"{kind} smoke completed", True, result["detail"])
    if kind == "matte":
        _record(checks, "matte smoke alpha shape", True, result["detail"])


def _run_four_k(
    modules: dict[str, Any],
    kind: str,
    name: str,
    device: str,
    image_path: str | None,
    allow_download: bool,
    checks: list[Check],
) -> None:
    import torch

    try:
        image = _read_image(image_path) if image_path else None
        if image is not None and max(image.shape[:2]) < 3000:
            image = None
        if image is None:
            image = _image(four_k=True)
        result = _run_model(
            modules, kind, name, device, image, 1, allow_download=allow_download
        )
        if not result.get("ok"):
            _record(checks, "4K smoke", False, result.get("detail", ""))
            return
        _record(
            checks,
            "4K smoke",
            True,
            f"{image.shape[1]}x{image.shape[0]}, load {result['load_s']:.2f}s, "
            f"encode {result.get('encode_s', 0):.2f}s, infer {result['infer_s']:.2f}s, "
            f"peak {result['peak_mb']} MB",
        )
        total_mb = 0
        if torch.cuda.is_available():
            total_mb = int(torch.cuda.get_device_properties(0).total_memory / 2**20)
        if total_mb:
            ok = result["peak_mb"] < total_mb
            _record(
                checks,
                "4K peak within device memory",
                ok,
                f"{result['peak_mb']} / {total_mb} MB",
            )
    except Exception as exc:
        _record(checks, "4K smoke", False, f"{type(exc).__name__}: {exc}")


def _select_default_model(modules: dict[str, Any], requested: str | None) -> tuple[str, str]:
    catalog = _catalog(modules)
    if requested:
        for kind in ("sam2", "sam3", "grounding", "yoloe", "matte"):
            if requested in catalog[kind]:
                return kind, requested
        raise ValueError(
            f"unknown model {requested!r}; choose one of "
            f"{', '.join(sum(catalog.values(), []))}"
        )
    weights_dir = modules["device"].MODEL_WEIGHTS_DIR
    for name in catalog["sam2"]:
        if (weights_dir / name).is_file():
            return "sam2", name
    for kind in ("sam2", "sam3", "grounding", "yoloe", "matte"):
        if catalog[kind]:
            return kind, catalog[kind][0]
    raise RuntimeError("model catalogue is empty")


def _main_parent(args: argparse.Namespace) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    checks: list[Check] = []
    log_path = setup_logging("all")
    started = time.perf_counter()
    try:
        backend = _choose_backend(args.backend)
        fork = FORK_FOR_BACKEND[backend]
        _print_banner(backend)
        print(f"log     : {log_path}")

        import torch

        print("\n-- phase 1/6: preflight " + "-" * 41)
        print(f"platform: {platform.platform()}")
        try:
            usage = shutil.disk_usage(ROOT)
            print(f"disk    : {usage.free / 2**30:.1f} GB free")
        except Exception:
            pass
        print(f"torch   : {torch.__version__}")
        print(f"runtime : HIP={getattr(torch.version, 'hip', None)} "
              f"CUDA={getattr(torch.version, 'cuda', None)}")
        if backend == "nvidia":
            smi = _nvidia_smi()
            if smi:
                print(f"nvidia-smi: {smi}")
            else:
                print("nvidia-smi: unavailable or returned no GPU")
        if backend == "amd":
            try:
                gfx_mod = importlib.import_module(f"{fork}.amd_gfx")
                props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
                arch = getattr(props, "gcnArchName", "") if props else ""
                target = gfx_mod.target_info(arch)
                if target:
                    print(f"amd target : {target.gfx} / {target.family} / {target.support}")
                else:
                    print("amd target : unknown in the Windows target registry")
            except Exception as exc:
                print(f"amd target : could not inspect ({type(exc).__name__}: {exc})")

        print("\n-- phase 2/6: device " + "-" * 44)
        modules = _import_fork(fork)
        _set_cache_environment(modules["device"].MODEL_WEIGHTS_DIR)
        info = modules["device"].detect_device()
        gpu = _gpu_description(backend)
        if backend in ("amd", "nvidia") and 0 < gpu.get("total_mb", 0) < 12288:
            # Match the 8 GB launcher policy: the HR matte checkpoints are capped
            # to 1024 px so the health run tests the shipped configuration.
            os.environ.setdefault("SPECTRA_MATTE_INPUT_SIZE", "1024")
        print(f"selected device: {info.torch_device} - {info.description}")
        if gpu.get("cuda_available"):
            print(
                f"gpu: {gpu.get('name')} - {gpu.get('total_mb')} MB - "
                f"arch={gpu.get('arch') or f'sm_{gpu.get('major')}{gpu.get('minor')}'}"
            )
        if backend in ("amd", "nvidia"):
            _record(checks, f"{backend.upper()} runtime available", gpu.get("cuda_available", False),
                    f"{gpu.get('name', 'no GPU')} / {gpu.get('total_mb', 0)} MB")
            if backend == "nvidia":
                _record(checks, "NVIDIA test status", None, "untested by project; this run is the first evidence")
        else:
            _record(checks, "GPU runtime available", None, "CPU-only run")

        print("\n-- phase 3/6: catalogue " + "-" * 40)
        catalog = _catalog(modules)
        counts = {kind: len(names) for kind, names in catalog.items()}
        total = sum(counts.values())
        print(f"catalog: {counts} (total {total})")
        _record(checks, f"catalogue total is {EXPECTED_TOTAL}", total == EXPECTED_TOTAL, str(total))
        for kind, expected in EXPECTED_CATALOG.items():
            _record(checks, f"{kind} catalogue count is {expected}", counts.get(kind) == expected,
                    str(counts.get(kind)))
        missing = _missing_weights(modules, catalog)
        _record(checks, "all catalogue weights are local", not missing,
                f"{len(missing)} missing" + (f": {', '.join(missing[:6])}" if missing else ""))
        if missing and not args.download and args.all_models:
            _record(checks, "full sweep prerequisites", False,
                    "--all-models needs every weight; rerun with --download")
        elif missing:
            print("[WARN] missing weights will be downloaded only when --download is passed")

        print("\n-- phase 4/6: smoke " + "-" * 43)
        kind, name = _select_default_model(modules, args.model)
        print(f"selected model: {kind}/{name}")
        if args.skip_smoke:
            _record(checks, "smoke", None, "--skip-smoke")
        else:
            _run_smoke(modules, kind, name, info.torch_device, _read_image(args.image),
                       args.runs, args.download, checks)

        if args.all_models:
            print("\n-- phase 5/6: all-model sweep " + "-" * 34)
            if missing and not args.download:
                _record(checks, "all-model sweep", False, "missing weights and --download not set")
            else:
                order = [
                    ("sam2", catalog["sam2"]),
                    ("sam3", catalog["sam3"]),
                    ("grounding", catalog["grounding"]),
                    ("yoloe", catalog["yoloe"]),
                    ("matte", catalog["matte"]),
                ]
                for model_kind, names in order:
                    for model_name in names:
                        result = _spawn_model(
                            backend,
                            model_kind,
                            model_name,
                            image_path=args.image,
                            runs=1,
                            allow_download=args.download,
                        )
                        detail = (
                            f"load {result.get('load_s', 0):.2f}s, "
                            f"encode {result.get('encode_s', 0):.2f}s, "
                            f"infer {result.get('infer_s', 0):.2f}s, "
                            f"peak {result.get('peak_mb', 0)} MB; "
                            f"{result.get('detail', '')}"
                        )
                        _record(
                            checks,
                            f"sweep {model_kind}/{model_name}",
                            bool(result.get("ok")),
                            detail,
                        )
        else:
            print("\n-- phase 5/6: all-model sweep skipped (pass --all-models) " + "-" * 8)

        print("\n-- phase 6/6: 4K / memory " + "-" * 39)
        if args.skip_4k:
            _record(checks, "4K smoke", None, "--skip-4k")
        else:
            _run_four_k(modules, kind, name, info.torch_device, args.image, args.download, checks)

        elapsed = time.perf_counter() - started
        failures = [check for check in checks if check.status == "FAIL"]
        skips = [check for check in checks if check.status == "SKIP"]
        passes = [check for check in checks if check.status == "PASS"]
        print("\n" + "=" * 74)
        print(
            f"RESULT: {'PASS' if not failures else 'FAIL'} - "
            f"{len(passes)} passed, {len(failures)} failed, {len(skips)} skipped "
            f"in {elapsed:.1f}s"
        )
        print(f"log: {log_path}")
        print("=" * 74)
        if failures:
            print("failures:")
            for check in failures:
                print(f"  - {check.name}: {check.detail}")
        return 1 if failures else 0
    except Exception as exc:
        logging.getLogger("health").exception("fatal health-check error")
        print(f"\nRESULT: FAIL - {type(exc).__name__}: {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spectra Segment AMD / NVIDIA / CPU health check"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "amd", "nvidia", "cpu"],
        default="auto",
        help="auto chooses AMD ROCm, NVIDIA CUDA or CPU from the running torch build",
    )
    parser.add_argument("--model", default=None, help="one catalogue model to smoke-test")
    parser.add_argument("--runs", type=int, default=3, help="smoke inference repetitions")
    parser.add_argument("--download", action="store_true", help="allow missing weights to download")
    parser.add_argument("--image", default=None, help="optional image path instead of synthetic 810x1080")
    parser.add_argument("--all-models", action="store_true", help="sweep every catalogue entry in a child process")
    parser.add_argument("--skip-smoke", action="store_true", help="skip the single-model smoke phase")
    parser.add_argument("--skip-4k", action="store_true", help="skip the 3840x2160 memory pass")
    parser.add_argument("--single", choices=["sam2", "sam3", "grounding", "yoloe", "matte"], default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-log", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:
        if not args.name:
            parser.error("--single requires --name")
        if args.backend == "auto":
            args.backend = _choose_backend("auto")
        return _child_main(args)

    if _running_as_child():
        args.no_log = True
    return _main_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
