from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import cv2

from .detector_base import Detection, DetectorBackend
from ..utils.device import MODEL_WEIGHTS_DIR, release_gpu_memory
from ..utils.stderr_progress import StderrInterceptor as _StderrInterceptor

LOGGER = logging.getLogger(__name__)

YOLOE_MODELS = {
    "yoloe-26l-seg": "YOLOE-26 Large (Segmentation)",
    "yoloe-26m-seg": "YOLOE-26 Medium (Segmentation)",
    "yoloe-26s-seg": "YOLOE-26 Small (Segmentation)",
    "yoloe-26n-seg": "YOLOE-26 Nano (Segmentation)",
    "yoloe-11l-seg": "YOLOE-11 Large (Segmentation)",
    "yoloe-11m-seg": "YOLOE-11 Medium (Segmentation)",
    "yoloe-11s-seg": "YOLOE-11 Small (Segmentation)",
}

YOLOE_METADATA: dict[str, dict[str, str]] = {
    "yoloe-26l-seg": {"tier": "large",  "perf": "Best accuracy — 36.8% LVIS mAP, 161 FPS on T4"},
    "yoloe-26m-seg": {"tier": "medium", "perf": "Balanced — 35.4% LVIS mAP"},
    "yoloe-26s-seg": {"tier": "small",  "perf": "Fast — 29.9% LVIS mAP"},
    "yoloe-26n-seg": {"tier": "tiny",   "perf": "Fastest — 23.7% LVIS mAP"},
    "yoloe-11l-seg": {"tier": "large",  "perf": "YOLO11-based — 35.2% LVIS mAP"},
    "yoloe-11m-seg": {"tier": "medium", "perf": "YOLO11-based — balanced"},
    "yoloe-11s-seg": {"tier": "small",  "perf": "YOLO11-based — fast"},
}

_YOLOE_WEIGHTS: dict[str, str] = {
    "yoloe-26l-seg": "yoloe-26l-seg.pt",
    "yoloe-26m-seg": "yoloe-26m-seg.pt",
    "yoloe-26s-seg": "yoloe-26s-seg.pt",
    "yoloe-26n-seg": "yoloe-26n-seg.pt",
    "yoloe-11l-seg": "yoloe-11l-seg.pt",
    "yoloe-11m-seg": "yoloe-11m-seg.pt",
    "yoloe-11s-seg": "yoloe-11s-seg.pt",
}


class YOLEDetector(DetectorBackend):
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model: Any = None
        self._model_name: str | None = None
        self._current_classes: list[str] = []

    def load_model(self, model_path: str | None = None) -> None:
        model_name = model_path or "yoloe-26l-seg"
        if model_name not in YOLOE_MODELS:
            raise ValueError(
                f"Unknown YOLOE model '{model_name}'. "
                f"Available: {list(YOLOE_MODELS.keys())}"
            )

        weight_file = _YOLOE_WEIGHTS[model_name]

        model_dir = MODEL_WEIGHTS_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        local_path = model_dir / weight_file

        encoder_file = "mobileclip2_b.ts" if model_name.startswith("yoloe-26") else "mobileclip_blt.ts"
        encoder_path = model_dir / encoder_file

        missing = [p for p in (local_path, encoder_path) if not p.exists()]
        if missing:
            from backend_amd_gpu.utils.net_check import is_connected
            if not is_connected():
                raise RuntimeError(
                    f"No internet connection and {', '.join(p.name for p in missing)} not found locally.\n"
                    f"  Connect to the internet to download, or place the files in:\n"
                    f"  {model_dir}"
                )
            from backend_amd_gpu.utils.model_integrity import download_model_file
            for p in missing:
                download_model_file(p.name, p)

        import os
        import types
        os.environ["ULTRALYTICS_HOME"] = str(model_dir)

        try:
            from ultralytics.utils import SETTINGS
            SETTINGS["weights_dir"] = str(model_dir)
        except Exception as e:
            LOGGER.warning("YOLOE: could not set SETTINGS weights_dir: %s", e)

        LOGGER.debug("YOLOE: importing ultralytics...")
        from ultralytics import YOLOE

        import ultralytics.nn.text_model as _text_model_module
        _orig_build_text_model = _text_model_module.build_text_model

        def _build_text_model(variant, device=None):
            if variant == "mobileclip2:b":
                return _text_model_module.MobileCLIPTS(device, weight=str(model_dir / "mobileclip2_b.ts"))
            if variant == "mobileclip:blt":
                return _text_model_module.MobileCLIPTS(device, weight=str(model_dir / "mobileclip_blt.ts"))
            return _orig_build_text_model(variant, device)

        _text_model_module.build_text_model = _build_text_model

        old_stderr = sys.stderr
        old_cwd = os.getcwd()
        sys.stderr = _StderrInterceptor("YOLOE")
        os.chdir(str(model_dir))
        try:
            from backend_amd_gpu.utils.model_integrity import verify_model
            if not verify_model(local_path, weight_file):
                raise RuntimeError(
                    f"Integrity check failed for {weight_file}. "
                    f"Delete it and re-download: {local_path}"
                )
            LOGGER.debug("YOLOE: loading from local path %s", local_path)
            self._model = YOLOE(str(local_path))
        finally:
            sys.stderr = old_stderr
            os.chdir(old_cwd)

        _orig_get_text_pe = self._model.model.get_text_pe

        def _cached_get_text_pe(self, text, *args, **kwargs):
            kwargs["cache_clip_model"] = True
            return _orig_get_text_pe(text, *args, **kwargs)

        self._model.model.get_text_pe = types.MethodType(_cached_get_text_pe, self._model.model)

        self._model_name = model_name
        self._current_classes = []
        LOGGER.info("Loaded YOLOE detector: %s", model_name)

    def unload_model(self) -> None:
        self._model = None
        self._model_name = None
        self._current_classes = []
        release_gpu_memory()

    def detect(
        self,
        image: np.ndarray,
        query: str,
        confidence: float | None = None,
        max_detections: int = 10,
    ) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("YOLOE model not loaded. Call load_model() first.")

        return self._detect_standard(image, query, confidence, max_detections)

    def _detect_standard(
        self,
        image: np.ndarray,
        query: str,
        confidence: float | None,
        max_detections: int,
    ) -> list[Detection]:
        labels = [q.strip() for q in query.split(",") if q.strip()]
        if not labels:
            return []

        LOGGER.debug("YOLOE: setting classes to %s", labels)
        self._model.set_classes(labels)
        self._current_classes = labels

        predict_kwargs: dict[str, Any] = {"conf": confidence if confidence is not None else 0.05}
        LOGGER.debug("YOLOE detect: labels=%s, conf=%s", labels, predict_kwargs["conf"])

        old_stderr = sys.stderr
        sys.stderr = _StderrInterceptor("YOLOE")
        try:
            results = self._model.predict(image, **predict_kwargs)
        finally:
            sys.stderr = old_stderr

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            masks_data = None
            if hasattr(result, "masks") and result.masks is not None and result.masks.data is not None:
                masks_data = result.masks.data.cpu().numpy()
            img_h, img_w = image.shape[:2]

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                names = result.names if hasattr(result, 'names') else {}
                classes = self._current_classes
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    score = float(confs[i])
                    cls_idx = int(clss[i])
                    if cls_idx in names:
                        label = names[cls_idx]
                    elif classes and cls_idx < len(classes):
                        label = classes[cls_idx]
                    else:
                        label = str(cls_idx)
                    if x1 >= x2 or y1 >= y2:
                        continue

                    det_mask = None
                    if masks_data is not None and i < len(masks_data):
                        raw_mask = masks_data[i]
                        resized = cv2.resize(raw_mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                        det_mask = (resized > 0.5).astype(np.bool_)

                    detections.append(
                        Detection(bbox=[x1, y1, x2, y2], score=score, label=label, mask=det_mask)
                    )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:max_detections]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
