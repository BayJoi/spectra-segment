from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import torch

from .base import SegmentationBackend
from ..utils.device import MODEL_WEIGHTS_DIR, release_gpu_memory
from ..utils.stderr_progress import StderrInterceptor as _StderrInterceptor
from ..utils.torch_threads import cpu_threads, cudnn_disabled, resolve_threads

LOGGER = logging.getLogger(__name__)
_MODEL_LOAD_LOCK = threading.Lock()

DEFAULT_ENCODE_DIM = 1024
ENCODE_DIM_MAX = 1500
MAX_ASPECT_RATIO = 1.6

_DEFAULT_THREADS = resolve_threads("SAM3_THREADS")

SAM3_MODEL = "sam3.pt"
SAM3_MODEL_DISPLAY = "SAM 3"

_SAM3_OVERRIDES: dict[str, Any] = {
    "task": "segment",
    "mode": "predict",
    "imgsz": 1024,
    "conf": 0.25,
    "iou": 0.7,
    "agnostic_nms": False,
}


class SAM3Backend(SegmentationBackend):

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._predictor: Any = None
        self._model_path: str | None = None
        self._image_set: bool = False
        self._src_shape: tuple[int, int] | None = None
        self._encode_dim: int = DEFAULT_ENCODE_DIM
        self._infer_lock = threading.Lock()
        self._orig_shape: tuple[int, int] | None = None
        self._proc_shape: tuple[int, int] | None = None
        self._content_shape: tuple[int, int] | None = None
        self._pad_top = 0
        self._pad_left = 0
        self._pad_scale = 1.0

    def load_model(self, model_path: str) -> None:
        if model_path != SAM3_MODEL:
            raise ValueError(f"Unknown SAM3 model '{model_path}'.")

        model_dir = MODEL_WEIGHTS_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        os.environ["ULTRALYTICS_HOME"] = str(model_dir)
        os.environ["YOLO_CONFIG_DIR"] = str(model_dir)

        try:
            import clip
        except ImportError:
            raise RuntimeError(
                "SAM 3 requires the 'clip' text-encoder package, which is missing.\n"
                "  Re-run install_amd.bat to install it."
            ) from None

        abs_model = str(model_dir / model_path)
        if not Path(abs_model).exists():
            from backend_amd_gpu.utils.net_check import is_connected
            if not is_connected():
                raise RuntimeError(
                    f"No internet connection and '{model_path}' not found locally.\n"
                    f"  Connect to the internet to download, or place the file in:\n"
                    f"  {model_dir}"
                )
            from backend_amd_gpu.utils.model_integrity import download_model_file, PINNED_MODEL_URLS
            url = PINNED_MODEL_URLS.get(model_path)
            if url:
                LOGGER.debug("Not cached: %s", model_path)
                download_model_file(model_path, Path(abs_model), url=url)
            else:
                LOGGER.debug("Not cached: %s", model_path)

        with _MODEL_LOAD_LOCK:
            _orig = torch.load

            def _safe(*a, **kw):
                kw.setdefault("weights_only", True)
                return _orig(*a, **kw)

            torch.load = _safe
            _interceptor = _StderrInterceptor(model_path)
            _old_stderr = sys.stderr
            sys.stderr = _interceptor
            try:
                from ultralytics.utils import SETTINGS
                SETTINGS["weights_dir"] = str(model_dir)
                try:
                    with cpu_threads(_DEFAULT_THREADS):
                        self._predictor = self._build_predictor(abs_model)
                except Exception as e:
                    if self._device != "cpu" and not (isinstance(e, ModuleNotFoundError) or "clip" in str(e).lower()):
                        LOGGER.warning("Device %s failed (%s), falling back to CPU", self._device, e)
                        self._device = "cpu"
                        with cpu_threads(_DEFAULT_THREADS):
                            self._predictor = self._build_predictor(abs_model)
                    else:
                        raise
                self._model_path = model_path
                LOGGER.debug("Loaded %s on %s", SAM3_MODEL_DISPLAY, self._device)
            finally:
                sys.stderr = _old_stderr
                torch.load = _orig

    def _build_predictor(self, abs_model: str):
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        overrides = dict(_SAM3_OVERRIDES)
        overrides.update({
            "model": abs_model,
            "device": self._device,
            "quantize": int(os.environ.get("SAM3_QUANTIZE", "32" if self._device == "cpu" else "16")),
        })
        predictor = SAM3SemanticPredictor(overrides=overrides)
        predictor.setup_model()
        predictor.imgsz = [self._encode_dim, self._encode_dim]
        if os.environ.get("SAM3_CHANNELS_LAST", "0") == "1":
            try:
                model = predictor.model
                if hasattr(model, "eval"):
                    model.eval()
                model = model.to(memory_format=torch.channels_last)
                predictor.model = model
            except Exception:
                LOGGER.exception("SAM3 channels_last failed; keeping default memory format")
        if os.environ.get("SAM3_COMPILE", "0") == "1":
            try:
                model = predictor.model
                if hasattr(model, "eval"):
                    model.eval()
                predictor.model = torch.compile(model, dynamic=True)
            except Exception:
                LOGGER.exception("SAM3 torch.compile failed; running uncompiled")
        return predictor

    def unload_model(self) -> None:
        with self._infer_lock:
            self.reset_image()
            self._predictor = None
            self._model_path = None
            release_gpu_memory()

    def _prepare_input(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        oh, ow = image.shape[:2]
        self._orig_shape = (oh, ow)
        self._pad_top = 0
        self._pad_left = 0
        self._pad_scale = 1.0

        scale = 1.0
        longest = max(oh, ow)
        if longest > self._encode_dim:
            scale = self._encode_dim / longest
        self._pad_scale = scale

        sw, sh = int(round(ow * scale)), int(round(oh * scale))
        if sh == 0 or sw == 0:
            raise ValueError(f"Image too small after scaling ({sw}x{sh}).")
        self._content_shape = (sh, sw)

        proc = image
        if scale < 1.0:
            proc = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)

        aspect = max(sh, sw) / max(1, min(sh, sw))
        if aspect <= MAX_ASPECT_RATIO:
            self._proc_shape = (sh, sw)
            return proc, (sh, sw)

        side = max(sh, sw)
        top = (side - sh) // 2
        left = (side - sw) // 2
        padded = np.full((side, side, 3), 114, dtype=proc.dtype)
        padded[top:top + sh, left:left + sw] = proc
        self._proc_shape = (side, side)
        self._pad_top = top
        self._pad_left = left
        return padded, (side, side)

    def set_encoding_resolution(self, encode_dim: int) -> None:
        if not 384 <= encode_dim <= ENCODE_DIM_MAX:
            raise ValueError(f"encode_dim must be within 384-{ENCODE_DIM_MAX}, got {encode_dim}")
        if encode_dim == self._encode_dim and self._image_set:
            return
        LOGGER.info("SAM3 encoding resolution set to %dx%d", encode_dim, encode_dim)
        with self._infer_lock:
            if self._predictor is not None:
                self._predictor.imgsz = [encode_dim, encode_dim]
                self._predictor.reset_image()
            self._encode_dim = encode_dim
            self._src_shape = None
            self._image_set = False

    def set_image(self, image: np.ndarray) -> None:
        if self._predictor is None:
            raise RuntimeError("Model not loaded.")
        self._src_shape = (image.shape[0], image.shape[1])
        try:
            self._encode_image(image)
        except torch.OutOfMemoryError:
            if self._device == "cpu":
                raise
            self._fallback_cpu_and_reencode(image)
        except RuntimeError as e:
            if self._device != "cpu" and "out of memory" in str(e).lower():
                self._fallback_cpu_and_reencode(image)
            else:
                raise

    def _fallback_cpu_and_reencode(self, image: np.ndarray) -> None:
        LOGGER.warning(
            "set_image: VRAM out of memory on %s - falling back to CPU (slower)",
            self._device,
        )
        model_path = self._model_path
        self.unload_model()
        self._device = "cpu"
        self.load_model(model_path)
        self._src_shape = (image.shape[0], image.shape[1])
        LOGGER.info("set_image: re-encoding %dx%d on CPU after VRAM fallback", image.shape[1], image.shape[0])
        self._encode_image(image)

    def _encode_image(self, image: np.ndarray) -> None:
        if self._predictor is None:
            raise RuntimeError("Model not loaded.")
        self._src_shape = (image.shape[0], image.shape[1])
        proc, proc_shape = self._prepare_input(image)
        LOGGER.debug(
            "set_image: encoding %dx%d (processed %dx%d, pad=%s) with %s ...",
            image.shape[1], image.shape[0], proc_shape[1], proc_shape[0],
            (self._pad_top, self._pad_left) if self._proc_shape != self._src_shape else "none",
            self._model_path or "?",
        )
        with self._infer_lock:
            self._predictor.reset_image()
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                with cpu_threads(_DEFAULT_THREADS), cudnn_disabled(), torch.inference_mode():
                    self._predictor.set_image(proc)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                LOGGER.error("set_image: encoding failed: %s", e)
                raise
        LOGGER.debug("set_image: encoding complete")
        self._image_set = True

    def predict_text(self, text: str, confidence: float | None = None) -> dict[str, Any]:
        if not self._image_set:
            raise RuntimeError("No image set.")
        with self._infer_lock:
            if self._predictor is None:
                raise RuntimeError("Model not loaded.")
            features = getattr(self._predictor, "features", None)
            if features is None:
                raise RuntimeError("No cached features.")
            with cpu_threads(_DEFAULT_THREADS if self._device == "cpu" else 1), \
                    cudnn_disabled(), torch.inference_mode():
                pred_masks, pred_boxes = self._predictor.inference_features(
                    features,
                    self._proc_shape or self._src_shape,
                    text=[text],
                )
        return self._format_result(pred_masks, pred_boxes, confidence=confidence)

    def predict(
        self,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        bboxes: list[float] | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> dict[str, Any]:
        if not self._image_set:
            raise RuntimeError("No image set.")
        if points is not None and len(points) > 0:
            raise NotImplementedError("SAM3 semantic mode does not support point prompts")
        with self._infer_lock:
            if self._predictor is None:
                raise RuntimeError("Model not loaded.")
            features = getattr(self._predictor, "features", None)
            if features is None:
                raise RuntimeError("No cached features.")
            with cpu_threads(_DEFAULT_THREADS if self._device == "cpu" else 1), \
                    cudnn_disabled(), torch.inference_mode():
                pred_masks, pred_boxes = self._predictor.inference_features(
                    features,
                    self._proc_shape or self._src_shape,
                    bboxes=bboxes,
                    labels=labels,
                )
        return self._format_result(pred_masks, pred_boxes)

    def _restore_outputs(self, pred_masks, pred_boxes):
        orig_shape = self._orig_shape or self._src_shape
        proc_shape = self._proc_shape or self._src_shape
        if orig_shape == proc_shape and self._pad_top == 0 and self._pad_left == 0:
            return pred_masks, pred_boxes

        oh, ow = orig_shape
        ch, cw = self._content_shape or (oh, ow)
        if pred_masks is not None:
            import torch.nn.functional as F

            band = pred_masks[:, self._pad_top:self._pad_top + ch, self._pad_left:self._pad_left + cw]
            band = band.float()[None]
            if band.shape[-2:] != (oh, ow):
                band = F.interpolate(band, (oh, ow), mode="bilinear")
            band = band[0]
            pred_masks = band > 0.5

        if pred_boxes is not None and pred_boxes.shape[0] > 0:
            b = pred_boxes[:, :4].clone().float()
            b[:, 0] = (b[:, 0] - self._pad_left) / self._pad_scale
            b[:, 1] = (b[:, 1] - self._pad_top) / self._pad_scale
            b[:, 2] = (b[:, 2] - self._pad_left) / self._pad_scale
            b[:, 3] = (b[:, 3] - self._pad_top) / self._pad_scale
            b[:, 0::2] = b[:, 0::2].clamp(0, ow)
            b[:, 1::2] = b[:, 1::2].clamp(0, oh)
            pred_boxes = torch.cat([b, pred_boxes[:, 4:]], dim=-1)

        return pred_masks, pred_boxes

    def _format_result(
        self,
        pred_masks,
        pred_boxes,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        h, w = self._src_shape or (0, 0)
        if pred_masks is None:
            return {
                "masks": np.zeros((0, h, w), dtype=bool),
                "scores": np.array([], dtype=float),
                "bboxes": np.zeros((0, 4), dtype=float),
                "low_res_masks": None,
            }
        pred_masks, pred_boxes = self._restore_outputs(pred_masks, pred_boxes)
        masks = pred_masks.detach().cpu().numpy()
        if pred_boxes is None:
            scores = np.zeros(masks.shape[0], dtype=np.float64)
            boxes = np.zeros((masks.shape[0], 4), dtype=np.float64)
        else:
            boxes = pred_boxes.detach().cpu().numpy()
            scores = boxes[:, 4].astype(np.float64) if boxes.shape[1] > 4 else np.zeros(boxes.shape[0])
        if confidence is not None and boxes.shape[0] > 0:
            keep = scores >= confidence
            masks = masks[keep]
            scores = scores[keep]
            boxes = boxes[keep]
        return {
            "masks": masks,
            "scores": scores,
            "bboxes": boxes[:, :4],
            "low_res_masks": None,
        }

    def reset_image(self) -> None:
        if self._predictor is not None:
            self._predictor.reset_image()
        self._image_set = False
        self._src_shape = None
        self._orig_shape = None
        self._proc_shape = None
        self._content_shape = None
        self._pad_top = 0
        self._pad_left = 0
        self._pad_scale = 1.0

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def has_image(self) -> bool:
        return self._image_set
