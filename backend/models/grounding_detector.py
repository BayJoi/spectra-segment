from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .detector_base import Detection, DetectorBackend
from ..utils.device import MODEL_WEIGHTS_DIR, cpu_fallback_enabled, release_gpu_memory
from ..utils.offline import offline_guard
from ..utils.stderr_progress import StderrInterceptor as _StderrInterceptor

LOGGER = logging.getLogger(__name__)

GROUNDING_DETECTORS = {
    "grounding-dino-tiny": "Grounding DINO (Tiny)",
    "grounding-dino-base": "Grounding DINO (Base)",
    "mm-gdino-base-all": "MM Grounding DINO (Base All)",
    "mm-gdino-large-all": "MM Grounding DINO (Large All)",
    "florence-2-base": "Florence-2 (Base)",
    "florence-2-large": "Florence-2 (Large)",
    "cogflorence-2.2-large": "CogFlorence 2.2 Large",
}

DETECTOR_METADATA: dict[str, dict[str, str]] = {
    "grounding-dino-tiny":  {"tier": "tiny",  "perf": "Fastest detector - great for quick object finding"},
    "grounding-dino-base":  {"tier": "medium","perf": "Accurate object detection with text prompts"},
    "mm-gdino-base-all":    {"tier": "medium","perf": "Improved Grounding DINO - trained on 11 datasets, +7.5 mAP over original"},
    "mm-gdino-large-all":   {"tier": "large", "perf": "Best accuracy Grounding DINO - 60.3 mAP on COCO, trained on 12 datasets"},
    "florence-2-base":      {"tier": "medium","perf": "Versatile - captioning + detection + OCR"},
    "florence-2-large":     {"tier": "large", "perf": "Most capable - best accuracy, needs GPU"},
    "cogflorence-2.2-large":{"tier": "large", "perf": "Fine-tuned Florence - enhanced grounding + captioning"},
}

HF_MODEL_IDS = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    "mm-gdino-base-all": "openmmlab-community/mm_grounding_dino_base_all",
    "mm-gdino-large-all": "openmmlab-community/mm_grounding_dino_large_all",
    "florence-2-base": "microsoft/Florence-2-base",
    "florence-2-large": "microsoft/Florence-2-large",
    "cogflorence-2.2-large": "thwri/CogFlorence-2.2-Large",
}

HF_MODEL_REVISIONS = {
    "grounding-dino-tiny": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "grounding-dino-base": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    "mm-gdino-base-all": "b32293f50a56f7218d9f73ff940ffb9eba84d63c",
    "mm-gdino-large-all": "0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0",
    "florence-2-base": "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
    "florence-2-large": "21a599d414c4d928c9032694c424fb94458e3594",
    "cogflorence-2.2-large": "19f2c614fdfd18ba49c81d567d70d3a68313e0bb",
}

TRUSTED_HF_IDS = set(HF_MODEL_IDS.values())

_FLORENCE_MODELS = {"florence-2-base", "florence-2-large", "cogflorence-2.2-large"}

_WEIGHTS_MIN_BYTES = 50 * 1024 * 1024
_WEIGHTS_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def hf_weights_downloaded(
    hf_id: str,
    cache_root: Any = None,
    revision: str | None = None,
) -> bool:
    from pathlib import Path

    root = Path(cache_root) if cache_root else MODEL_WEIGHTS_DIR / "hf_cache"
    snapshots = root / "hub" / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    if revision:
        snapshots = snapshots / revision
    if not snapshots.exists():
        return False
    for entry in snapshots.rglob("*"):
        try:
            if (
                entry.is_file()
                and entry.name.lower().endswith(_WEIGHTS_SUFFIXES)
                and entry.stat().st_size >= _WEIGHTS_MIN_BYTES
            ):
                return True
        except OSError:
            continue
    return False


class _NullIO:
    def write(self, *_a) -> int:
        return 0

    def flush(self) -> None:
        pass


_WEIGHT_RE = re.compile(r"\.(safetensors|bin|pt|pth|onnx)$", re.IGNORECASE)


def _emit(model_name: str, **fields: Any) -> None:
    payload = {"name": model_name, "phase": "download"}
    payload.update(fields)
    LOGGER.info(
        "%s: %.0f%% - files %s/%s",
        model_name,
        payload.get("pct") or 0,
        payload.get("files_done"),
        payload.get("files_total"),
        extra={"dl": payload},
    )


def _snapshot_fetch(
    hf_id: str,
    cache_dir: Any,
    model_name: str,
    revision: str | None = None,
) -> bool:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from huggingface_hub.utils.tqdm import tqdm as hf_tqdm
        import sys as _sys
        _fd = _sys.modules["huggingface_hub.file_download"]
        _ut = _sys.modules["huggingface_hub.utils.tqdm"]
    except Exception:
        return False

    hub_dir = str(Path(cache_dir) / "hub")
    state = {"files_done": 0, "bytes": 0}
    state_lock = threading.Lock()

    def emit_combined(pct: int = 0, done_bytes: int = 0, total_bytes: int = 0) -> None:
        with state_lock:
            fd, ft = state["files_done"], state["files_total"]
        _emit(
            model_name,
            file=state.get("main_file"),
            pct=pct,
            done_bytes=done_bytes,
            total_bytes=total_bytes,
            files_done=fd,
            files_total=ft,
        )

    try:
        info = HfApi().model_info(repo_id=hf_id, revision=revision, files_metadata=True)
        files = [(s.rfilename, int(s.size or 0)) for s in info.siblings if (s.size or 0) > 0]
    except Exception as e:
        LOGGER.debug("File listing failed for %s (%s)", hf_id, e)
        files = []
    if not files:
        try:
            snapshot_download(repo_id=hf_id, revision=revision, cache_dir=hub_dir)
            return True
        except Exception as e:
            LOGGER.warning("Snapshot fetch failed for %s (%s) - falling back", hf_id, e)
            return False

    weight_files = [(n, s) for n, s in files if _WEIGHT_RE.search(n)]
    main_name, main_size = max(weight_files or files, key=lambda t: t[1])
    state["files_total"] = len(files)
    state["main_file"] = Path(main_name).name

    class _ByteBar(hf_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._sp_total = int(kwargs.get("total") or 0)
            self._sp_initial = int(kwargs.get("initial") or 0)
            self._sp_desc = str(kwargs.get("desc") or "")
            self._sp_is_main = Path(self._sp_desc).name == state["main_file"]
            self._sp_last = 0.0
            kwargs["file"] = _NullIO()
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> None:
            if getattr(self, "disable", False):
                self.n = getattr(self, "n", 0) + n
            else:
                super().update(n)
            if not self._sp_is_main:
                return
            now = time.monotonic()
            if now - self._sp_last < 0.4 and self.n < self._sp_total:
                return
            self._sp_last = now
            done = min(self._sp_initial + int(self.n), self._sp_total)
            pct = round(done * 100 / self._sp_total) if self._sp_total else 0
            with state_lock:
                state["bytes"] = done
            emit_combined(pct=pct, done_bytes=done, total_bytes=self._sp_total)

        def close(self) -> None:
            try:
                if self._sp_is_main:
                    emit_combined(
                        pct=100,
                        done_bytes=self._sp_total,
                        total_bytes=self._sp_total,
                    )
            finally:
                super().close()

    orig_fd, orig_ut = _fd.tqdm, _ut.tqdm
    _fd.tqdm = _ByteBar
    _ut.tqdm = _ByteBar
    try:
        others = [n for n, _ in files if n != main_name]

        def fetch_one(fname: str) -> None:
            hf_hub_download(repo_id=hf_id, filename=fname, revision=revision, cache_dir=hub_dir)
            with state_lock:
                state["files_done"] += 1
            emit_combined()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as main_pool:
            main_fut = main_pool.submit(fetch_one, main_name)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(others))) as pool:
                futures = [pool.submit(fetch_one, n) for n in others]
                errors = [f.exception() for f in concurrent.futures.as_completed(futures)]
            main_fut.result()
        if any(e is not None for e in errors):
            raise RuntimeError(f"{errors[0]}")
        emit_combined(pct=100, done_bytes=main_size, total_bytes=main_size)
        return True
    except Exception as e:
        LOGGER.warning("Snapshot fetch failed for %s (%s) - falling back", hf_id, e)
        return False
    finally:
        _fd.tqdm, _ut.tqdm = orig_fd, orig_ut


def _is_trusted_local_path(hf_id: str) -> bool:
    import os

    path = os.path.abspath(hf_id)
    if not os.path.isdir(path):
        return False
    hub_root = os.path.abspath(str(MODEL_WEIGHTS_DIR / "hf_cache" / "hub"))
    if not (path.startswith(hub_root + os.sep)):
        return False
    parts = os.path.relpath(path, hub_root).split(os.sep)
    if len(parts) < 3 or parts[-2] not in ("snapshots", ".no_exist"):
        return False
    return any(parts[0] == "models--" + trusted.replace("/", "--") for trusted in TRUSTED_HF_IDS)


def _normalize_auto_map(auto_map: dict) -> dict:
    cleaned = {}
    for key, value in auto_map.items():
        if not isinstance(value, str):
            continue
        value = value.strip()
        if "--" in value:
            head, _, tail = value.partition("--")
            if "/" in head:
                value = tail
        cleaned[key] = value
    return cleaned


def _florence_config_paths(hf_id: str) -> list[str]:
    from huggingface_hub import scan_cache_dir

    out: list[str] = []
    try:
        repos = [r for r in scan_cache_dir().repos if r.repo_id == hf_id]
    except Exception:
        return out
    for r in repos:
        for rev in r.revisions:
            snap = Path(rev.snapshot_path)
            for name in ("config.json", "preprocessor_config.json"):
                p = snap / name
                if p.exists():
                    out.append(str(p))
    return out


def _florence_auto_map(hf_id: str) -> dict | None:
    import json

    for config_path in _florence_config_paths(hf_id):
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        auto_map = cfg.get("auto_map")
        if not isinstance(auto_map, dict):
            continue
        cleaned = _normalize_auto_map(auto_map)
        cleaned.pop("AutoModelForZeroShotObjectDetection", None)
        return cleaned
    return None


_FLORENCE_SDPA_BROKEN = [False]


def _florence_sdpa_broken() -> bool:
    return bool(_FLORENCE_SDPA_BROKEN[0])


def _load_florence_model(
    hf_id: str,
    dtype: Any,
    local_only: bool = False,
    revision: str | None = None,
) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = None
    auto_map = _florence_auto_map(hf_id)
    if auto_map is not None:
        try:
            with offline_guard(local_only):
                config = AutoConfig.from_pretrained(
                    hf_id,
                    trust_remote_code=True,
                    local_files_only=local_only,
                    revision=revision,
                )
            config.auto_map = auto_map
        except Exception as exc:
            LOGGER.debug("Florence: could not build a patched config: %s", exc)
            config = None

    last_err: Exception | None = None
    for impl in ("sdpa", "eager"):
        if impl == "sdpa" and _florence_sdpa_broken():
            LOGGER.debug("Florence: skipping sdpa (transformers build lacks it)")
            continue
        try:
            with offline_guard(local_only):
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    config=config,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation=impl,
                    local_files_only=local_only,
                    low_cpu_mem_usage=True,
                    revision=revision,
                )
            LOGGER.info("Florence: active attention implementation: %s", impl)
            return model
        except AttributeError as e:
            last_err = e
            if "_supports_sdpa" in str(e):
                _FLORENCE_SDPA_BROKEN[0] = True
                LOGGER.debug("Florence sdpa unavailable in this transformers build (%s)", e)
                continue
            LOGGER.warning(
                "Florence: attention implementation '%s' failed to load (%s: %s)",
                impl, type(e).__name__, e,
            )
        except Exception as e:
            last_err = e
            LOGGER.warning(
                "Florence: attention implementation '%s' failed to load (%s: %s)",
                impl, type(e).__name__, e,
            )
    raise RuntimeError(
        f"Florence-2 failed to load with both sdpa and eager attention: {last_err}"
    ) from last_err


def _florence_language_generation_config(model: Any, language_model: Any) -> Any:
    from transformers import GenerationConfig

    try:
        config = GenerationConfig.from_model_config(language_model.config)
    except Exception as e:
        LOGGER.warning("Florence: could not build inner generation config (%s)", e)
        return getattr(model, "generation_config", None)

    outer = getattr(model, "generation_config", None)
    if outer is not None:
        try:
            for key, value in outer.to_diff_dict().items():
                setattr(config, key, value)
        except Exception:
            pass
    return config


def _enable_florence_generation(model: Any) -> None:
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception as e:
        LOGGER.warning("Florence: GenerationMixin unavailable (%s)", e)
        return

    for attr in ("", "language_model"):
        obj = model if not attr else getattr(model, attr, None)
        if obj is None or issubclass(type(obj), GenerationMixin):
            continue
        obj.__class__ = type(f"{type(obj).__name__}__gen", (type(obj), GenerationMixin), {})
        LOGGER.debug("Florence: attached GenerationMixin to %s", obj.__class__.__name__)

    lm = getattr(model, "language_model", None)
    if lm is not None and getattr(lm, "generation_config", None) is None:
        lm.generation_config = _florence_language_generation_config(model, lm)


def _florence_processor_source(hf_id: str) -> str | None:
    from pathlib import Path
    import json
    import shutil
    import tempfile

    source = Path(hf_id)
    config_path = source / "preprocessor_config.json"
    if not source.is_dir() or not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict):
        return None
    normalized = _normalize_auto_map(auto_map)
    if normalized == auto_map:
        return None
    class_ref = normalized.get("AutoProcessor", "")
    module_name = class_ref.split(".", 1)[0]
    if not module_name or not (source / f"{module_name}.py").is_file():
        return None

    target = Path(tempfile.mkdtemp(prefix="spectra-florence-processor-"))
    try:
        for child in source.iterdir():
            if not child.is_file() or child.suffix.lower() in {
                ".bin", ".pth", ".pt", ".safetensors"
            }:
                continue
            shutil.copy2(child, target / child.name)
        config["auto_map"] = normalized
        (target / "preprocessor_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        return None
    return str(target)


def _load_florence_detector(
    hf_id: str,
    model_name: str,
    device: str,
    local_only: bool = False,
    revision: str | None = None,
) -> tuple[Any, Any, str]:
    if hf_id not in TRUSTED_HF_IDS and not _is_trusted_local_path(hf_id):
        raise ValueError(
            f"Refusing trust_remote_code load of untrusted model '{hf_id}'. "
            f"Allowed: {sorted(TRUSTED_HF_IDS)}"
        )

    import os

    dtype = torch.float32
    if device != "cpu" and os.environ.get("SPECTRA_DETECTOR_FP16") == "1":
        dtype = torch.float16

    LOGGER.debug("Florence: importing transformers...")
    from transformers import AutoProcessor
    LOGGER.debug("Florence: transformers imported OK")

    LOGGER.debug("Florence: loading processor from %s ...", hf_id)
    processor_source = _florence_processor_source(hf_id)
    old_stderr = sys.stderr
    sys.stderr = _StderrInterceptor("Florence-2")
    try:
        with offline_guard(local_only):
            processor = AutoProcessor.from_pretrained(
                processor_source or hf_id,
                trust_remote_code=True,
                local_files_only=local_only,
                revision=revision,
            )
    finally:
        sys.stderr = old_stderr
        if processor_source is not None:
            import shutil
            shutil.rmtree(processor_source, ignore_errors=True)
    LOGGER.debug("Florence: processor loaded")

    try:
        from unittest.mock import patch as _patch
        from transformers.dynamic_module_utils import get_imports

        def _safe_imports(filename: str) -> list[str]:
            if not str(filename).endswith("modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            imports = [i for i in imports if i != "flash_attn"]
            return imports

        with _patch("transformers.dynamic_module_utils.get_imports", _safe_imports):
            LOGGER.debug("Florence: loading model (with flash_attn patch)...")
            sys.stderr = _StderrInterceptor("Florence-2")
            try:
                model = _load_florence_model(hf_id, dtype, local_only, revision)
            finally:
                sys.stderr = old_stderr
    except Exception:
        LOGGER.debug("Florence: flash_attn patch failed, retrying without patch...")
        sys.stderr = _StderrInterceptor("Florence-2")
        try:
            model = _load_florence_model(hf_id, dtype, local_only)
        finally:
            sys.stderr = old_stderr

    LOGGER.debug("Florence: model loaded, moving to device %s ...", device)
    actual_device = device
    if device != "cpu":
        try:
            model = model.to(device)
        except Exception as e:
            if not cpu_fallback_enabled():
                raise
            LOGGER.warning("Device %s failed for Florence (%s), using CPU", device, e)
            actual_device = "cpu"
            model = model.to("cpu")
    if actual_device == "cpu" and dtype != torch.float32:
        model = model.float()
        LOGGER.info("Detector on CPU: using fp32 (CPU fp16 is emulated and slow)")

    model.eval()
    _enable_florence_generation(model)

    if actual_device != "cpu":
        try:
            LOGGER.debug("Florence: pre-warming on %s ...", actual_device)
            from PIL import Image
            _dummy = processor(text="<OD>", images=Image.new("RGB", (1024, 1024)), return_tensors="pt")
            _model_dtype = next(model.parameters()).dtype
            _dummy = {k: v.to(actual_device, dtype=_model_dtype) if v.is_floating_point() else v.to(actual_device) for k, v in _dummy.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                with torch.inference_mode():
                    model.generate(**_dummy, max_new_tokens=1, num_beams=1, do_sample=False, use_cache=False)
            LOGGER.debug("Florence: pre-warm complete")
        except Exception as e:
            LOGGER.debug("Florence: pre-warm skipped (%s)", e)

    LOGGER.debug("Loaded Florence model: %s on %s (dtype=%s)", model_name, actual_device, dtype)
    return processor, model, actual_device


def _load_grounding_dino(
    hf_id: str,
    model_name: str,
    device: str,
    local_only: bool = False,
    revision: str | None = None,
) -> tuple[Any, Any, str]:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    import os

    dtype = torch.float32
    if device != "cpu" and os.environ.get("SPECTRA_DETECTOR_FP16") == "1":
        dtype = torch.float16

    old_stderr = sys.stderr
    sys.stderr = _StderrInterceptor("GroundingDINO")
    try:
        with offline_guard(local_only):
            processor = AutoProcessor.from_pretrained(
                hf_id,
                local_files_only=local_only,
                revision=revision,
            )
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                local_files_only=local_only,
                low_cpu_mem_usage=True,
                revision=revision,
            )
    finally:
        sys.stderr = old_stderr

    actual_device = device
    if device != "cpu":
        try:
            model = model.to(device)
        except Exception as e:
            if not cpu_fallback_enabled():
                raise
            LOGGER.warning("Device %s failed for detector (%s), using CPU", device, e)
            actual_device = "cpu"
            model = model.to("cpu")
    if actual_device == "cpu" and dtype != torch.float32:
        model = model.float()
        LOGGER.info("Detector on CPU: using fp32 (CPU fp16 is emulated and slow)")

    model.eval()

    if actual_device != "cpu":
        try:
            LOGGER.debug("GroundingDINO: pre-warming on %s ...", actual_device)
            from PIL import Image
            _dummy = processor(images=Image.new("RGB", (800, 800)), text=[["test"]], return_tensors="pt")
            _model_dtype = next(model.parameters()).dtype
            _dummy = {k: v.to(actual_device, dtype=_model_dtype) if v.is_floating_point() else v.to(actual_device) for k, v in _dummy.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(_model_dtype == torch.float16)):
                with torch.inference_mode():
                    model(**_dummy)
            LOGGER.debug("GroundingDINO: pre-warm complete")
        except Exception as e:
            LOGGER.debug("GroundingDINO: pre-warm skipped (%s)", e)

    LOGGER.debug("Loaded Grounding DINO: %s on %s (dtype=%s)", model_name, actual_device, dtype)
    return processor, model, actual_device


def _is_oom(exc: Exception) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _limit_image(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _scale_detections(
    detections: list[Detection], scale: float, width: int, height: int
) -> list[Detection]:
    output: list[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        output.append(
            Detection(
                bbox=[
                    min(max(x1 / scale, 0.0), float(width)),
                    min(max(y1 / scale, 0.0), float(height)),
                    min(max(x2 / scale, 0.0), float(width)),
                    min(max(y2 / scale, 0.0), float(height)),
                ],
                score=det.score,
                label=det.label,
                mask=det.mask,
            )
        )
    return output


class GroundingDetector(DetectorBackend):
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._processor: Any = None
        self._model: Any = None
        self._is_florence: bool = False
        self._actual_device: str = device
        self._max_side = max(800, int(os.environ.get("SPECTRA_DETECT_MAX_SIDE", "1280")))
        self._active_side = self._max_side
        self._allow_cpu_fallback = cpu_fallback_enabled()
        self._infer_lock = threading.Lock()

    def load_model(self, model_path: str | None = None) -> None:
        model_name = model_path or "grounding-dino-tiny"
        if model_name not in GROUNDING_DETECTORS:
            raise ValueError(
                f"Unknown grounding detector '{model_name}'. "
                f"Available: {list(GROUNDING_DETECTORS.keys())}"
            )

        import os
        from pathlib import Path

        hf_id = HF_MODEL_IDS[model_name]
        revision = HF_MODEL_REVISIONS[model_name]
        cache_dir = MODEL_WEIGHTS_DIR / "hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)

        model_cache = Path(cache_dir) / "hub" / f"models--{hf_id.replace('/', '--')}"
        no_exist_dir = model_cache / ".no_exist"
        snapshots_dir = model_cache / "snapshots"
        pinned_snapshot = snapshots_dir / revision
        if no_exist_dir.exists() and not snapshots_dir.exists():
            try:
                no_exist_dir.rename(snapshots_dir)
                LOGGER.info("Fixed HF cache: renamed .no_exist -> snapshots for %s", hf_id)
            except OSError:
                pass

        effective_device = self._device if torch.cuda.is_available() else "cpu"

        is_florence = model_name in _FLORENCE_MODELS

        processor = None
        model = None
        actual_device = None
        loaded = False
        try:
            if pinned_snapshot.is_dir():
                local_path = str(pinned_snapshot)
                LOGGER.info("Loading %s (cached, pinned %s)", hf_id, revision[:12])
                try:
                    if is_florence:
                        processor, model, actual_device = _load_florence_detector(
                            local_path, model_name, effective_device, local_only=True
                        )
                    else:
                        processor, model, actual_device = _load_grounding_dino(
                            local_path, model_name, effective_device, local_only=True
                        )
                    loaded = True
                except Exception as e:
                    LOGGER.debug(
                        "Pinned cache load failed for %s: %s - trying network", hf_id, e
                    )
                    processor = model = actual_device = None

            if not loaded:
                LOGGER.info("Not cached - downloading %s", model_name)
                from ..utils.net_check import is_connected
                if not is_connected():
                    raise RuntimeError(
                        f"Model '{hf_id}' not cached and no internet connection available.\n"
                        f"  Connect to the internet to download, or place the model in:\n"
                        f"  {cache_dir / 'hub'}"
                    ) from None
                fetched = _snapshot_fetch(hf_id, cache_dir, model_name, revision)
                if fetched and pinned_snapshot.is_dir():
                    local_path = str(pinned_snapshot)
                    LOGGER.info("Downloaded %s - loading from %s", model_name, hf_id)
                    LOGGER.info(
                        "%s: download complete",
                        model_name,
                        extra={"dl": {"name": model_name, "phase": "done",
                                      "files_done": None, "files_total": None, "pct": 100}},
                    )
                    if is_florence:
                        processor, model, actual_device = _load_florence_detector(
                            local_path, model_name, effective_device, local_only=True
                        )
                    else:
                        processor, model, actual_device = _load_grounding_dino(
                            local_path, model_name, effective_device, local_only=True
                        )
                    loaded = True
                if not loaded:
                    LOGGER.info("Downloading %s via direct load", model_name)
                    disable_progress_bars = None
                    try:
                        from huggingface_hub.utils import enable_progress_bars, disable_progress_bars
                        enable_progress_bars()
                    except Exception:
                        pass
                    try:
                        if is_florence:
                            processor, model, actual_device = _load_florence_detector(
                                hf_id, model_name, effective_device, local_only=False, revision=revision
                            )
                        else:
                            processor, model, actual_device = _load_grounding_dino(
                                hf_id, model_name, effective_device, local_only=False, revision=revision
                            )
                        loaded = True
                    finally:
                        if disable_progress_bars:
                            disable_progress_bars()
        except Exception:
            LOGGER.warning(
                "Failed to load grounding detector: %s (%s)",
                model_name,
                hf_id,
                exc_info=True,
            )
            raise

        self._processor = processor
        self._model = model
        self._actual_device = actual_device
        self._is_florence = is_florence
        self._active_side = self._max_side
        self._set_processor_size(self._active_side)
        release_gpu_memory()
        LOGGER.debug("Loaded grounding detector: %s (%s)", model_name, hf_id)

    def unload_model(self) -> None:
        self._processor = None
        self._model = None
        self._is_florence = False
        release_gpu_memory()

    def _set_processor_size(self, longest_side: int) -> None:
        """Make the image processor honour the detector limit.

        Grounding DINO otherwise rescales every input to its 1333 px default,
        which made a 1280 -> 1024 retry use the same tensor and fail twice.
        """
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is None:
            return
        shortest = max(384, int(round(longest_side * 0.6)))
        try:
            image_processor.size = {
                "shortest_edge": shortest,
                "longest_edge": longest_side,
            }
        except Exception as exc:
            LOGGER.debug("Could not set detector input size to %d px: %s", longest_side, exc)

    def detect(
        self,
        image: np.ndarray,
        query: str,
        confidence: float | None = None,
        max_detections: int = 10,
    ) -> list[Detection]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Detector not loaded. Call load_model() first.")

        with self._infer_lock:
            sides: list[int] = []
            for side in (self._active_side, min(self._max_side, 1280), 1024, 800):
                side = min(side, self._max_side)
                if side not in sides:
                    sides.append(side)

            last_oom: Exception | None = None
            last_limited = image
            last_side = sides[-1]
            last_scale = 1.0
            for index, side in enumerate(sides):
                limited, scale = _limit_image(image, side)
                self._set_processor_size(side)
                try:
                    detections = self._run_detect(limited, query, confidence, max_detections)
                except Exception as exc:
                    if not _is_oom(exc) or self._actual_device == "cpu":
                        raise
                    last_oom = exc
                    last_limited = limited
                    last_side = side
                    last_scale = scale
                    # Drop the failed forward's frame locals (input/output
                    # tensors) before asking the allocator to release VRAM.
                    if exc.__traceback__ is not None:
                        traceback.clear_frames(exc.__traceback__)
                    release_gpu_memory()
                    if index < len(sides) - 1:
                        next_side = sides[index + 1]
                        LOGGER.warning(
                            "Detector ran out of VRAM on %s at %d px; retrying at %d px on the GPU",
                            self._actual_device,
                            side,
                            next_side,
                            extra={
                                "phase": "vram",
                                "reason": "oom-retry",
                                "max_side": next_side,
                            },
                        )
                        continue
                    break

                self._active_side = side
                if scale != 1.0:
                    LOGGER.info(
                        "Detection image limited to %d px (scale %.3f) to fit the AMD profile",
                        side,
                        scale,
                        extra={"phase": "detect", "max_side": side, "scale": round(scale, 4)},
                    )
                    return _scale_detections(detections, scale, image.shape[1], image.shape[0])
                return detections

            if last_oom is None:
                raise RuntimeError("Detector failed without an error")
            if self._allow_cpu_fallback:
                LOGGER.warning(
                    "Detector exhausted VRAM on %s at %d px; falling back to CPU",
                    self._actual_device,
                    last_side,
                    extra={"phase": "vram", "reason": "oom-cpu-opt-in"},
                )
                self._fallback_cpu()
                self._set_processor_size(last_side)
                detections = self._run_detect(
                    last_limited, query, confidence, max_detections
                )
                if last_scale != 1.0:
                    return _scale_detections(
                        detections, last_scale, image.shape[1], image.shape[0]
                    )
                return detections
            LOGGER.error(
                "Detector exhausted VRAM on %s and automatic CPU fallback is disabled",
                self._actual_device,
                extra={"phase": "vram", "reason": "oom-no-fallback"},
            )
            raise torch.OutOfMemoryError(str(last_oom)) from last_oom

    def _run_detect(
        self,
        image: np.ndarray,
        query: str,
        confidence: float | None,
        max_detections: int,
    ) -> list[Detection]:
        if self._is_florence:
            return self._detect_florence(image, query, max_detections, confidence)
        return self._detect_grounding_dino(image, query, confidence, max_detections)

    def _fallback_cpu(self) -> None:
        model = self._model
        self._actual_device = "cpu"
        if model is not None:
            try:
                model = model.float()
            except Exception:
                pass
            try:
                model = model.to("cpu")
            except Exception as exc:
                LOGGER.warning("Detector could not move to CPU: %s", exc)
        self._model = model
        release_gpu_memory()

    def _move_inputs_to_device(self, inputs: dict) -> dict:
        if self._actual_device != "cpu":
            model_dtype = next(self._model.parameters()).dtype
            for k, v in inputs.items():
                if v.is_floating_point():
                    inputs[k] = v.to(device=self._actual_device, dtype=model_dtype)
                else:
                    inputs[k] = v.to(device=self._actual_device)
        else:
            inputs = {k: v.to(self._actual_device) for k, v in inputs.items()}
        return inputs

    def _detect_florence(
        self,
        image: np.ndarray,
        query: str,
        max_detections: int,
        confidence: float | None = None,
    ) -> list[Detection]:
        from PIL import Image

        rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb)

        labels = [q.strip() for q in query.split(",") if q.strip()]
        text_input = ", ".join(labels)

        if text_input:
            task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{text_input}"
            task_tag = "<CAPTION_TO_PHRASE_GROUNDING>"
        else:
            task_prompt = "<OD>"
            task_tag = "<OD>"

        LOGGER.debug("Florence detect: task_prompt=%s", task_prompt)

        _enable_florence_generation(self._model)

        inputs = self._processor(text=task_prompt, images=pil_image, return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs)

        is_half = next(self._model.parameters()).dtype == torch.float16
        autocast_device = "cpu" if self._actual_device == "cpu" else "cuda"
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=is_half):
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False,
                )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed = self._processor.post_process_generation(
            generated_text,
            task=task_tag,
            image_size=pil_image.size,
        )

        detections = []
        od_result = parsed.get(task_tag, {})
        bboxes = od_result.get("bboxes", [])
        od_labels = od_result.get("labels", [])

        LOGGER.debug("Florence raw: %d boxes, text=%s", len(bboxes), generated_text[:200])

        for bbox, label in zip(bboxes, od_labels):
            x1, y1, x2, y2 = bbox
            if x1 >= x2 or y1 >= y2:
                continue
            detections.append(
                Detection(bbox=[x1, y1, x2, y2], score=1.0, label=str(label))
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        LOGGER.debug(
            "Florence detect returning %d detections (Florence reports no per-box score)",
            len(detections),
        )

        return self._nms(detections, max_detections)

    def _detect_grounding_dino(
        self,
        image: np.ndarray,
        query: str,
        confidence: float | None,
        max_detections: int,
    ) -> list[Detection]:
        from PIL import Image

        rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb)
        img_w, img_h = pil_image.size

        labels = [q.strip() for q in query.split(",") if q.strip()]
        text_labels = [labels]

        LOGGER.debug("GroundingDINO detect: text_labels=%s, conf=%s", text_labels, confidence)

        inputs = self._processor(images=pil_image, text=text_labels, return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs)

        is_half = next(self._model.parameters()).dtype == torch.float16
        autocast_device = "cpu" if self._actual_device == "cpu" else "cuda"
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=is_half):
            with torch.inference_mode():
                outputs = self._model(**inputs)

        post_kwargs = dict(
            target_sizes=[(img_h, img_w)],
        )
        if confidence is not None:
            post_kwargs["threshold"] = confidence

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            **post_kwargs,
        )

        detections = []
        if results:
            for res in results:
                boxes = res.get("boxes", [])
                scores = res.get("scores", [])
                labels = res.get("text_labels", res.get("labels", []))
                LOGGER.debug("GroundingDINO raw: %d boxes", len(boxes))
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else list(box)
                    score_val = score.item() if hasattr(score, 'item') else float(score)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    detections.append(
                        Detection(bbox=[x1, y1, x2, y2], score=score_val, label=str(label))
                    )

        detections.sort(key=lambda d: d.score, reverse=True)
        LOGGER.debug("GroundingDINO detect returning %d detections", len(detections))

        return self._nms(detections, max_detections)

    @staticmethod
    def _nms(detections: list[Detection], max_detections: int, iou_threshold: float = 0.5) -> list[Detection]:
        if len(detections) <= 1:
            return detections[:max_detections]
        try:
            import supervision as sv

            sv_dets = sv.Detections(
                xyxy=np.asarray([d.bbox for d in detections], dtype=np.float32),
                confidence=np.asarray([d.score for d in detections], dtype=np.float32),
                data={"source_index": np.arange(len(detections))},
            )
            kept = sv_dets.with_nms(threshold=iou_threshold, class_agnostic=True)
            keep_ids = [int(i) for i in kept.data["source_index"]]
            return [detections[i] for i in keep_ids][:max_detections]
        except Exception:
            keep = []
            suppressed = [False] * len(detections)
            for i in range(len(detections)):
                if suppressed[i]:
                    continue
                keep.append(detections[i])
                if len(keep) >= max_detections:
                    break
                xi1, yi1, xi2, yi2 = detections[i].bbox
                area_i = (xi2 - xi1) * (yi2 - yi1)
                for j in range(i + 1, len(detections)):
                    if suppressed[j]:
                        continue
                    xj1, yj1, xj2, yj2 = detections[j].bbox
                    inter_x1 = max(xi1, xj1)
                    inter_y1 = max(yi1, yj1)
                    inter_x2 = min(xi2, xj2)
                    inter_y2 = min(yi2, yj2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
                    area_j = (xj2 - xj1) * (yj2 - yj1)
                    iou = inter_area / (area_i + area_j - inter_area) if (area_i + area_j - inter_area) > 0 else 0
                    if iou > iou_threshold:
                        suppressed[j] = True
            return keep[:max_detections]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
