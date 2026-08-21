from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import torch

from .detector_base import Detection, DetectorBackend
from ..utils.device import MODEL_WEIGHTS_DIR, release_gpu_memory
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


def _clean_florence_config(cache_dir, hf_id: str, local_only: bool = False) -> None:
    import json
    from pathlib import Path

    config_path = None
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(
            repo_id=hf_id,
            filename="config.json",
            cache_dir=str(cache_dir),
            local_files_only=local_only,
        )
    except Exception as e:
        LOGGER.debug("Could not download Florence config for cleaning: %s", e)
        return

    config_path = Path(config_path)
    if not config_path.exists():
        return

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        return

    auto_map = cfg.get("auto_map", {})
    bad_keys = [k for k in ("AutoProcessor", "AutoModelForZeroShotObjectDetection") if k in auto_map]
    if not bad_keys:
        return

    for k in bad_keys:
        del auto_map[k]
    if not auto_map:
        cfg.pop("auto_map", None)

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    LOGGER.debug("Cleaned Florence config.json auto_map at %s (removed %s)", config_path, bad_keys)


def _load_florence_model(hf_id: str, dtype: Any, local_only: bool = False) -> Any:
    from transformers import AutoModelForCausalLM

    last_err: Exception | None = None
    for impl in ("sdpa", "eager"):
        try:
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
        except Exception as e:
            last_err = e
            LOGGER.warning(
                "Florence: attention implementation '%s' failed to load (%s: %s)",
                impl, type(e).__name__, e,
            )
    raise RuntimeError(
        f"Florence-2 failed to load with both sdpa and eager attention: {last_err}"
    ) from last_err


def _load_florence_detector(hf_id: str, model_name: str, device: str, local_only: bool = False) -> tuple[Any, Any, str]:
    if hf_id not in TRUSTED_HF_IDS and not _is_trusted_local_path(hf_id):
        raise ValueError(
            f"Refusing trust_remote_code load of untrusted model '{hf_id}'. "
            f"Allowed: {sorted(TRUSTED_HF_IDS)}"
        )

    dtype = torch.float32

    LOGGER.debug("Florence: importing transformers...")
    from transformers import AutoProcessor
    LOGGER.debug("Florence: transformers imported OK")

    LOGGER.debug("Florence: loading processor from %s ...", hf_id)
    old_stderr = sys.stderr
    sys.stderr = _StderrInterceptor("Florence-2")
    try:
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

    model.eval()

    if actual_device != "cpu":
        try:
            LOGGER.debug("Florence: pre-warming on %s ...", actual_device)
            from PIL import Image
            _dummy = processor(text="<OD>", images=Image.new("RGB", (64, 64)), return_tensors="pt")
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

    dtype = torch.float32

    old_stderr = sys.stderr
    sys.stderr = _StderrInterceptor("GroundingDINO")
    try:
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

    model.eval()

    if actual_device != "cpu":
        try:
            LOGGER.debug("GroundingDINO: pre-warming on %s ...", actual_device)
            from PIL import Image
            _dummy = processor(images=Image.new("RGB", (64, 64)), text=[["test"]], return_tensors="pt")
            _model_dtype = next(model.parameters()).dtype
            _dummy = {k: v.to(actual_device, dtype=_model_dtype) if v.is_floating_point() else v.to(actual_device) for k, v in _dummy.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
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
        self._is_florence = is_florence

        loaded = False
        if snapshots_dir.exists():
            snapshot_dirs = sorted(
                (d for d in snapshots_dir.iterdir() if d.is_dir()),
                key=lambda p: p.stat().st_mtime, reverse=True
            )
            if snapshot_dirs:
                local_path = str(snapshot_dirs[0])
                LOGGER.info("Loading %s from local cache: %s", hf_id, local_path)
                try:
                    if is_florence:
                        self._processor, self._model, self._actual_device = _load_florence_detector(
                            local_path, model_name, effective_device, local_only=True
                        )
                    else:
                        self._processor, self._model, self._actual_device = _load_grounding_dino(
                            local_path, model_name, effective_device, local_only=True
                        )
                    loaded = True
                except Exception as e:
                    LOGGER.debug(
                        "Local cache load failed for %s: %s — trying network", hf_id, e
                    )

        if not loaded:
            LOGGER.info("Cache miss for '%s' — trying network download", hf_id)
            from backend.utils.net_check import is_connected
            if not is_connected():
                raise RuntimeError(
                    f"Model '{hf_id}' not cached and no internet connection available.\n"
                    f"  Connect to the internet to download, or place the model in:\n"
                    f"  {cache_dir / 'hub'}"
                ) from None
            if is_florence:
                _clean_florence_config(cache_dir, hf_id, local_only=False)
                self._processor, self._model, self._actual_device = _load_florence_detector(
                    hf_id, model_name, effective_device, local_only=False
                )
            else:
                self._processor, self._model, self._actual_device = _load_grounding_dino(
                    hf_id, model_name, effective_device, local_only=False
                )

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
            return self._detect_florence(image, query, max_detections)
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

        inputs = self._processor(text=task_prompt, images=pil_image, return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
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
            score = 1.0
            detections.append(
                Detection(bbox=[x1, y1, x2, y2], score=score, label=str(label))
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        LOGGER.debug("Florence detect returning %d detections", len(detections))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
