from __future__ import annotations

import concurrent.futures
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .detector_base import Detection, DetectorBackend
from ..utils.device import MODEL_WEIGHTS_DIR, release_gpu_memory
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
    "grounding-dino-tiny":  {"tier": "tiny",  "perf": "Fastest detector — great for quick object finding"},
    "grounding-dino-base":  {"tier": "medium","perf": "Accurate object detection with text prompts"},
    "mm-gdino-base-all":    {"tier": "medium","perf": "Improved Grounding DINO — trained on 11 datasets, +7.5 mAP over original"},
    "mm-gdino-large-all":   {"tier": "large", "perf": "Best accuracy Grounding DINO — 60.3 mAP on COCO, trained on 12 datasets"},
    "florence-2-base":      {"tier": "medium","perf": "Versatile — captioning + detection + OCR"},
    "florence-2-large":     {"tier": "large", "perf": "Most capable — best accuracy, needs GPU"},
    "cogflorence-2.2-large":{"tier": "large", "perf": "Fine-tuned Florence — enhanced grounding + captioning"},
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

TRUSTED_HF_IDS = set(HF_MODEL_IDS.values())

_FLORENCE_MODELS = {"florence-2-base", "florence-2-large", "cogflorence-2.2-large"}

_WEIGHTS_MIN_BYTES = 50 * 1024 * 1024
_WEIGHTS_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def hf_weights_downloaded(hf_id: str, cache_root: Any = None) -> bool:
    from pathlib import Path

    root = Path(cache_root) if cache_root else MODEL_WEIGHTS_DIR / "hf_cache"
    snapshots = root / "hub" / f"models--{hf_id.replace('/', '--')}" / "snapshots"
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
        "%s: %.0f%% · files %s/%s",
        model_name,
        payload.get("pct") or 0,
        payload.get("files_done"),
        payload.get("files_total"),
        extra={"dl": payload},
    )


def _snapshot_fetch(hf_id: str, cache_dir: Any, model_name: str) -> bool:
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

    def emit_combined(pct: int = 0, done_bytes: int = 0, total_bytes: int = 0, force: bool = False) -> None:
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
        info = HfApi().model_info(repo_id=hf_id, files_metadata=True)
        files = [(s.rfilename, int(s.size or 0)) for s in info.siblings if (s.size or 0) > 0]
    except Exception as e:
        LOGGER.debug("File listing failed for %s (%s)", hf_id, e)
        files = []
    if not files:
        try:
            snapshot_download(repo_id=hf_id, cache_dir=hub_dir)
            return True
        except Exception as e:
            LOGGER.warning("Snapshot fetch failed for %s (%s) — falling back", hf_id, e)
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
                        force=True,
                    )
            finally:
                super().close()

    orig_fd, orig_ut = _fd.tqdm, _ut.tqdm
    _fd.tqdm = _ByteBar
    _ut.tqdm = _ByteBar
    try:
        others = [n for n, _ in files if n != main_name]

        def fetch_one(fname: str) -> None:
            hf_hub_download(repo_id=hf_id, filename=fname, cache_dir=hub_dir)
            with state_lock:
                state["files_done"] += 1
            emit_combined(force=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as main_pool:
            main_fut = main_pool.submit(fetch_one, main_name)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(others))) as pool:
                futures = [pool.submit(fetch_one, n) for n in others]
                errors = [f.exception() for f in concurrent.futures.as_completed(futures)]
            main_fut.result()
        if any(e is not None for e in errors):
            raise RuntimeError(f"{errors[0]}")
        emit_combined(pct=100, done_bytes=main_size, total_bytes=main_size, force=True)
        return True
    except Exception as e:
        LOGGER.warning("Snapshot fetch failed for %s (%s) — falling back", hf_id, e)
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


def _normalize_auto_map(auto_map: dict, local_repo: str) -> dict:
    """Drop cross-repo pointers from an `auto_map`.

    Some Florence repos point `AutoProcessor` at a *different* repo, e.g.
    `microsoft/Florence-2-large--processing_florence2.Florence2Processor`.
    transformers then tries to fetch that module from the other repo, which is
    not in the local cache, so loading fails even though this repo contains
    its own `processing_florence2.py`. Keep only the module path.
    """
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
    """Local paths for a repo's config files, resolved from the snapshot we already have.

    `hf_hub_download(cache_dir=...)` resolves differently across
    huggingface_hub versions and fails to find a repo that `_snapshot_fetch`
    already downloaded. Resolving from the snapshot directory is exact.
    """
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


def _clean_florence_config(cache_dir, hf_id: str, local_only: bool = False) -> None:
    import json

    for config_path in _florence_config_paths(hf_id):
        config_path = Path(config_path)
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except Exception:
            continue

        auto_map = cfg.get("auto_map") or {}
        if not isinstance(auto_map, dict):
            continue
        cleaned_map = _normalize_auto_map(auto_map, hf_id)
        changed = cleaned_map != auto_map

        bad_keys = [k for k in ("AutoModelForZeroShotObjectDetection",) if k in cleaned_map]
        if bad_keys:
            for k in bad_keys:
                del cleaned_map[k]
            changed = True

        if not changed:
            continue

        if cleaned_map:
            cfg["auto_map"] = cleaned_map
        else:
            cfg.pop("auto_map", None)

        try:
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            LOGGER.debug("Cleaned %s auto_map at %s", config_path.name, config_path)
        except OSError as e:
            LOGGER.warning("Could not rewrite Florence %s: %s", config_path.name, e)


_FLORENCE_SDPA_BROKEN = [False]


def _florence_sdpa_broken() -> bool:
    return bool(_FLORENCE_SDPA_BROKEN[0])


def _load_florence_model(hf_id: str, dtype: Any, local_only: bool = False) -> Any:
    from transformers import AutoModelForCausalLM

    last_err: Exception | None = None
    for impl in ("sdpa", "eager"):
        if impl == "sdpa" and _florence_sdpa_broken():
            LOGGER.debug("Florence: skipping sdpa (transformers build lacks it)")
            continue
        try:
            with offline_guard(local_only):
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation=impl,
                    local_files_only=local_only,
                    low_cpu_mem_usage=True,
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
    """Build a generation config for Florence's inner language model.

    The outer config ships `num_beams`/`early_stopping` in its
    `generation_config.json` but carries none of the special-token ids, so copying
    it straight onto the language model leaves `decoder_start_token_id` and
    `bos_token_id` unset and `generate()` raises "`decoder_start_token_id` or
    `bos_token_id` has to be defined". The inner language config does have them,
    so build from it and then overlay the outer's non-default settings.
    """
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
    """Give Florence's remote-code model working `.generate()` on transformers >= 4.50.

    transformers 4.50 dropped `GenerationMixin` from `PreTrainedModel`. The
    Florence-2 / CogFlorence remote code predates that change: the outer
    `Florence2ForConditionalGeneration.generate()` merges the image and text
    embeddings and then delegates to `self.language_model.generate(...)`. The
    language model never defined its own `generate` and relied on inheriting it,
    so the delegated call raises `AttributeError`.

    Re-basing the loaded instances onto a subclass that also inherits
    `GenerationMixin` restores the inherited methods without touching the
    read-only model cache. The language model needs a generation config with the
    special-token ids, which this also installs.
    """
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


def _load_florence_detector(hf_id: str, model_name: str, device: str, local_only: bool = False) -> tuple[Any, Any, str]:
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
    old_stderr = sys.stderr
    sys.stderr = _StderrInterceptor("Florence-2")
    try:
        with offline_guard(local_only):
            processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, local_files_only=local_only)
    finally:
        sys.stderr = old_stderr
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
                model = _load_florence_model(hf_id, dtype, local_only)
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


def _load_grounding_dino(hf_id: str, model_name: str, device: str, local_only: bool = False) -> tuple[Any, Any, str]:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    import os

    dtype = torch.float32
    if device != "cpu" and os.environ.get("SPECTRA_DETECTOR_FP16") == "1":
        dtype = torch.float16

    old_stderr = sys.stderr
    sys.stderr = _StderrInterceptor("GroundingDINO")
    try:
        with offline_guard(local_only):
            processor = AutoProcessor.from_pretrained(hf_id, local_files_only=local_only)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                local_files_only=local_only,
                low_cpu_mem_usage=True,
            )
    finally:
        sys.stderr = old_stderr

    actual_device = device
    if device != "cpu":
        try:
            model = model.to(device)
        except Exception as e:
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


class GroundingDetector(DetectorBackend):
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._processor: Any = None
        self._model: Any = None
        self._model_name: str | None = None
        self._is_florence: bool = False
        self._actual_device: str = device

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
        cache_dir = MODEL_WEIGHTS_DIR / "hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)

        model_cache = Path(cache_dir) / "hub" / f"models--{hf_id.replace('/', '--')}"
        no_exist_dir = model_cache / ".no_exist"
        snapshots_dir = model_cache / "snapshots"
        if no_exist_dir.exists() and not snapshots_dir.exists():
            try:
                no_exist_dir.rename(snapshots_dir)
                LOGGER.info("Fixed HF cache: renamed .no_exist → snapshots for %s", hf_id)
            except OSError:
                pass

        effective_device = self._device if torch.cuda.is_available() else "cpu"

        is_florence = model_name in _FLORENCE_MODELS

        processor = None
        model = None
        actual_device = None
        loaded = False
        try:
            if snapshots_dir.exists():
                snapshot_dirs = sorted(
                    (d for d in snapshots_dir.iterdir() if d.is_dir()),
                    key=lambda p: p.stat().st_mtime, reverse=True
                )
                if snapshot_dirs:
                    local_path = str(snapshot_dirs[0])
                    LOGGER.info("Loading %s (cached)", hf_id)
                    try:
                        if is_florence:
                            _clean_florence_config(cache_dir, hf_id, local_only=True)
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
                            "Local cache load failed for %s: %s — trying network", hf_id, e
                        )
                        processor = model = actual_device = None

            if not loaded:
                LOGGER.info("Not cached — downloading %s", model_name)
                from backend.utils.net_check import is_connected
                if not is_connected():
                    raise RuntimeError(
                        f"Model '{hf_id}' not cached and no internet connection available.\n"
                        f"  Connect to the internet to download, or place the model in:\n"
                        f"  {cache_dir / 'hub'}"
                    ) from None
                fetched = _snapshot_fetch(hf_id, cache_dir, model_name)
                if fetched and snapshots_dir.exists():
                    snapshot_dirs = sorted(
                        (d for d in snapshots_dir.iterdir() if d.is_dir()),
                        key=lambda p: p.stat().st_mtime, reverse=True
                    )
                    if snapshot_dirs:
                        local_path = str(snapshot_dirs[0])
                        LOGGER.info("Downloaded %s — loading from %s", model_name, hf_id)
                        LOGGER.info(
                            "%s: download complete",
                            model_name,
                            extra={"dl": {"name": model_name, "phase": "done",
                                          "files_done": None, "files_total": None, "pct": 100}},
                        )
                        if is_florence:
                            _clean_florence_config(cache_dir, hf_id, local_only=True)
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
                            _clean_florence_config(cache_dir, hf_id, local_only=False)
                            processor, model, actual_device = _load_florence_detector(
                                hf_id, model_name, effective_device, local_only=False
                            )
                        else:
                            processor, model, actual_device = _load_grounding_dino(
                                hf_id, model_name, effective_device, local_only=False
                            )
                        loaded = True
                    finally:
                        if disable_progress_bars:
                            disable_progress_bars()
        except Exception:
            self._model_name = model_name
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
        self._model_name = model_name
        LOGGER.debug("Loaded grounding detector: %s (%s)", model_name, hf_id)

    def unload_model(self) -> None:
        self._processor = None
        self._model = None
        self._model_name = None
        self._is_florence = False
        release_gpu_memory()

    def detect(
        self,
        image: np.ndarray,
        query: str,
        confidence: float | None = None,
        max_detections: int = 10,
    ) -> list[Detection]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Detector not loaded. Call load_model() first.")

        if self._is_florence:
            return self._detect_florence(image, query, max_detections, confidence)
        return self._detect_grounding_dino(image, query, confidence, max_detections)

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
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_half):
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
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_half):
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
