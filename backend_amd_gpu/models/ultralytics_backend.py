from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import SegmentationBackend
from ..utils.device import MODEL_WEIGHTS_DIR, release_gpu_memory
from ..utils.stderr_progress import StderrInterceptor as _StderrInterceptor
from ..utils.torch_threads import cpu_threads, resolve_threads, cudnn_disabled

LOGGER = logging.getLogger(__name__)
_MODEL_LOAD_LOCK = threading.Lock()

_DEFAULT_THREADS = resolve_threads("SAM2_THREADS")


def _move_to_device(obj: Any, device: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj

ULTRALYTICS_MODELS = {
    "sam2.1_b.pt": "SAM 2.1 Base", "sam2.1_l.pt": "SAM 2.1 Large",
    "sam2.1_s.pt": "SAM 2.1 Small", "sam2.1_t.pt": "SAM 2.1 Tiny",
    "sam2_b.pt": "SAM 2 Base", "sam2_l.pt": "SAM 2 Large",
    "sam2_s.pt": "SAM 2 Small", "sam2_t.pt": "SAM 2 Tiny",
}

MODEL_METADATA: dict[str, dict[str, str]] = {
    "sam2.1_t.pt":        {"tier": "tiny",  "perf": "Very fast — good for real-time workflows"},
    "sam2_t.pt":          {"tier": "tiny",  "perf": "Very fast — lightweight SAM2 variant"},
    "sam2.1_s.pt":        {"tier": "small", "perf": "Balanced speed/quality — great for most tasks"},
    "sam2_s.pt":          {"tier": "small", "perf": "Balanced speed/quality — SAM2 small"},
    "sam2.1_b.pt":        {"tier": "medium","perf": "Higher accuracy — needs GPU for best speed"},
    "sam2_b.pt":          {"tier": "medium","perf": "Higher accuracy — SAM2 base"},
    "sam2.1_l.pt":        {"tier": "large", "perf": "Best accuracy — requires GPU, slower"},
    "sam2_l.pt":          {"tier": "large", "perf": "Best accuracy — SAM2 large"},
}

DENYLIST = {"FastSAM-s.pt", "FastSAM-x.pt", "sam3_b.pt", "sam3_l.pt"}


class UltralyticsBackend(SegmentationBackend):
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._predictor: Any = None
        self._model_path: str | None = None
        self._image_set: bool = False
        self._src_shape: tuple[int, int] | None = None
        self._infer_lock = threading.RLock()
        self._image_pe: Any = None
        self._encoder_on_cpu: bool = False

    def load_model(self, model_path: str) -> None:
        if model_path in DENYLIST:
            raise ValueError(f"Model '{model_path}' is explicitly blocked.")
        if model_path not in ULTRALYTICS_MODELS:
            raise ValueError(f"Unknown Ultralytics model '{model_path}'.")

        model_dir = MODEL_WEIGHTS_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        os.environ["ULTRALYTICS_HOME"] = str(model_dir)
        os.environ["YOLO_CONFIG_DIR"] = str(model_dir)

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
            import sys
            _orig = torch.load
            def _safe(*a, **kw): kw.setdefault("weights_only", True); return _orig(*a, **kw)
            torch.load = _safe
            _interceptor = _StderrInterceptor(model_path)
            _old_stderr = sys.stderr
            sys.stderr = _interceptor
            try:
                from ultralytics.models.sam.predict import SAM2Predictor
                from ultralytics.utils import SETTINGS
                SETTINGS["weights_dir"] = str(model_dir)
                abs_model = str(model_dir / model_path)
                try:
                    self._predictor = SAM2Predictor(overrides={"model": abs_model, "imgsz": 1024, "device": self._device, "quantize": int(os.environ.get("SAM2_QUANTIZE", "32" if self._device == "cpu" else "16"))})
                    self._predictor.setup_model()
                except Exception as e:
                    if self._device != "cpu":
                        LOGGER.warning("Device %s failed (%s), falling back to CPU", self._device, e)
                        self._device = "cpu"
                        self._predictor = SAM2Predictor(overrides={"model": abs_model, "imgsz": 1024, "device": "cpu", "quantize": int(os.environ.get("SAM2_QUANTIZE", "32"))})
                        self._predictor.setup_model()
                    else:
                        raise
                self._model_path = model_path
                self._encoder_on_cpu = False
                if (
                    os.environ.get("SAM2_CHANNELS_LAST") == "1"
                    and self._predictor.model is not None
                ):
                    try:
                        self._predictor.model = self._predictor.model.to(memory_format=torch.channels_last)
                        LOGGER.debug("SAM2_CHANNELS_LAST=1")
                    except Exception as e:
                        LOGGER.warning("Could not enable channels_last: %s", e)
                if (
                    os.environ.get("SAM2_COMPILE") == "1"
                    and self._device != "cpu"
                    and self._predictor.model is not None
                ):
                    try:
                        self._predictor.model = torch.compile(self._predictor.model)
                        LOGGER.debug("SAM2_COMPILE=1")
                    except Exception as e:
                        LOGGER.warning("torch.compile failed for SAM2: %s", e)
                LOGGER.debug("Loaded %s on %s", ULTRALYTICS_MODELS[model_path], self._device)
            finally:
                sys.stderr = _old_stderr
                torch.load = _orig

    def unload_model(self) -> None:
        with self._infer_lock:
            self.reset_image()
            self._predictor = None
            self._model_path = None
            self._encoder_on_cpu = False
        release_gpu_memory()

    def set_image(self, image: np.ndarray) -> None:
        if self._predictor is None:
            raise RuntimeError("Model not loaded.")
        self._src_shape = (image.shape[0], image.shape[1])
        LOGGER.debug("set_image: encoding %dx%d with %s ...", image.shape[1], image.shape[0], self._model_path or "?")
        try:
            self._encode_image(image)
        except torch.OutOfMemoryError:
            if self._device == "cpu":
                raise
            self._fallback_cpu_encoder_reencode(image)
        except RuntimeError as e:
            if self._device != "cpu" and "out of memory" in str(e).lower():
                self._fallback_cpu_encoder_reencode(image)
            else:
                raise

    def _fallback_cpu_encoder_reencode(self, image: np.ndarray) -> None:
        try:
            self._cpu_encoder_reencode(image)
        except Exception as e:
            LOGGER.warning("CPU-encoder re-encode failed (%s); falling back to full CPU", e)
            self._fallback_full_cpu(image)

    def _cpu_encoder_reencode(self, image: np.ndarray) -> None:
        if self._predictor is None:
            raise RuntimeError("Model not loaded.")
        predictor = self._predictor
        model = predictor.model
        gpu = predictor.device
        LOGGER.warning(
            "set_image: encode OOM on %s - running image encoder on CPU "
            "(one-time); decoder stays on %s",
            self._device, gpu,
        )
        release_gpu_memory()
        self._src_shape = (image.shape[0], image.shape[1])
        predictor.im = None
        predictor.features = None
        with self._infer_lock:
            model.cpu()
            predictor.device = torch.device("cpu")
            if predictor.mean is not None:
                predictor.mean = predictor.mean.cpu()
            if predictor.std is not None:
                predictor.std = predictor.std.cpu()
            try:
                with torch.inference_mode(), cpu_threads(_DEFAULT_THREADS):
                    predictor.imgsz = [1024, 1024]
                    predictor.set_image(image)
            finally:
                model.to(gpu)
                predictor.device = gpu
                if predictor.mean is not None:
                    predictor.mean = predictor.mean.to(gpu)
                if predictor.std is not None:
                    predictor.std = predictor.std.to(gpu)
            if predictor.features is not None:
                predictor.features = _move_to_device(predictor.features, gpu)
            self._image_pe = model.sam_prompt_encoder.get_dense_pe()
            self._offload_encoder()
            self._warm_decoder()
            self._image_set = True
        LOGGER.warning("set_image: CPU-encoder re-encode complete, decoder active on %s", gpu)

    def _fallback_full_cpu(self, image: np.ndarray) -> None:
        LOGGER.warning(
            "set_image: falling back to full CPU model (%s) (slower)",
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
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            with self._infer_lock:
                self._restore_encoder()
                if self._device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._predictor.imgsz = [1024, 1024]
                with cpu_threads(_DEFAULT_THREADS), cudnn_disabled(), torch.inference_mode():
                    self._predictor.set_image(image)
                self._image_pe = self._predictor.model.sam_prompt_encoder.get_dense_pe()
                self._warm_decoder()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._offload_encoder()
                torch.cuda.empty_cache()
        except Exception as e:
            LOGGER.error("set_image: encoding failed: %s", e)
            raise
        LOGGER.debug("set_image: encoding complete")
        self._image_set = True

    def _model_dtype(self) -> torch.dtype:
        predictor = self._predictor
        if predictor is not None:
            dtype = getattr(predictor, "torch_dtype", None)
            if dtype is not None:
                return dtype
            model = getattr(predictor, "model", None)
            if model is not None:
                p = next(model.parameters(), None)
                if p is not None:
                    return p.dtype
        return torch.float32

    def _warm_decoder(self) -> None:
        if (
            self._device == "cpu"
            or not torch.cuda.is_available()
            or self._predictor is None
            or self._image_pe is None
            or self._src_shape is None
        ):
            return
        features = self._predictor.features
        if features is None:
            return
        src_h, src_w = self._src_shape
        dtype = self._model_dtype()
        r = min(1024 / src_h, 1024 / src_w)
        pts = torch.tensor([[[src_w / 2 * r, src_h / 2 * r]]], dtype=dtype, device=self._predictor.device)
        lbs = torch.tensor([[1]], dtype=torch.int32, device=self._predictor.device)
        try:
            with torch.inference_mode(), cpu_threads(_DEFAULT_THREADS if self._device == "cpu" else 1):
                se, de = self._predictor.model.sam_prompt_encoder(points=(pts, lbs), boxes=None, masks=None)
                self._predictor.model.sam_mask_decoder(
                    image_embeddings=features["image_embed"],
                    image_pe=self._image_pe,
                    sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                    multimask_output=False,
                    repeat_image=False, high_res_features=features["high_res_feats"],
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            LOGGER.debug("SAM2 decoder warmed up")
        except Exception as e:
            LOGGER.debug("SAM2 decoder warm-up skipped (%s)", e)

    def _restore_encoder(self) -> None:
        if not self._encoder_on_cpu or self._predictor is None:
            return
        model = self._predictor.model
        enc = getattr(model, "image_encoder", None)
        if enc is not None:
            enc.to(self._predictor.device)
        self._encoder_on_cpu = False
        LOGGER.debug("SAM2 image encoder moved back to %s", self._predictor.device)

    def _offload_encoder(self) -> None:
        if (
            self._device == "cpu"
            or not torch.cuda.is_available()
            or os.environ.get("SAM2_OFFLOAD_ENCODER") != "1"
            or self._predictor is None
        ):
            return
        model = self._predictor.model
        enc = getattr(model, "image_encoder", None)
        if enc is None or next(enc.parameters(), None) is None:
            return
        try:
            enc.cpu()
            self._encoder_on_cpu = True
            LOGGER.debug("SAM2 image encoder offloaded to CPU")
        except Exception as e:
            LOGGER.warning("Could not offload SAM2 image encoder: %s", e)

    def _features(self) -> Any:
        with self._infer_lock:
            if self._predictor is None:
                raise RuntimeError("Model not loaded.")
            if not self._image_set:
                raise RuntimeError("No image set.")
            features = self._predictor.features
        if features is None:
            raise RuntimeError("No cached features. Call set_image() first.")
        return features

    def predict(self, points=None, labels=None, bboxes=None, mask_input=None, multimask_output=False):
        if not self._image_set:
            raise RuntimeError("No image set.")
        with self._infer_lock:
            if self._predictor is None:
                raise RuntimeError("Model not loaded.")
            features = self._predictor.features
        if features is None:
            raise RuntimeError("No cached features.")

        src_h, src_w = self._src_shape
        dtype = self._model_dtype()

        r = min(1024 / src_h, 1024 / src_w)
        all_coords, all_lbl = [], []
        if bboxes is not None and len(bboxes) == 4:
            x1, y1, x2, y2 = (c * r for c in bboxes)
            all_coords.extend([[x1, y1], [x2, y2]]); all_lbl.extend([2, 3])
        if points is not None and labels is not None and len(points) > 0:
            all_coords.extend([[p[0] * r, p[1] * r] for p in points]); all_lbl.extend(labels)

        img_embed = features["image_embed"]
        high_res = features["high_res_feats"]
        with self._infer_lock:
            if self._predictor is None:
                raise RuntimeError("Model not loaded.")
            image_pe = self._image_pe
            if image_pe is None:
                image_pe = self._predictor.model.sam_prompt_encoder.get_dense_pe()
                self._image_pe = image_pe

        point_inputs = None
        if all_coords:
            pts = torch.tensor(all_coords, dtype=dtype, device=self._predictor.device).unsqueeze(0)
            lbs = torch.tensor(all_lbl, dtype=torch.int32, device=self._predictor.device).unsqueeze(0)
            point_inputs = (pts, lbs)

        mask_t = None
        if mask_input is not None:
            th, tw = self._predictor.model.sam_prompt_encoder.mask_input_size
            mask_t = torch.tensor(mask_input, dtype=dtype, device=self._predictor.device)
            if mask_t.ndim == 2: mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            elif mask_t.ndim == 3: mask_t = mask_t.unsqueeze(0)
            if mask_t.shape[-2:] != (th, tw):
                mask_t = torch.nn.functional.interpolate(mask_t, size=(th, tw), mode="bilinear", align_corners=False)

        with self._infer_lock, torch.inference_mode(), cpu_threads(_DEFAULT_THREADS if self._device == "cpu" else 1), cudnn_disabled():
            se, de = self._predictor.model.sam_prompt_encoder(points=point_inputs, boxes=None, masks=mask_t)
            pm, ps, _, _ = self._predictor.model.sam_mask_decoder(
                image_embeddings=img_embed,
                image_pe=image_pe,
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=multimask_output,
                repeat_image=False, high_res_features=high_res,
            )
        pm = pm.flatten(0, 1); ps = ps.flatten(0, 1)

        if pm is None or pm.shape[0] == 0:
            return {
                "masks": np.zeros((0, self._src_shape[0], self._src_shape[1]), dtype=bool)
                if self._src_shape else np.array([], dtype=bool),
                "scores": np.array([]),
                "low_res_masks": None,
            }

        raw = pm.detach().cpu().numpy()
        if self._src_shape:
            nh, nw = round(src_h * r), round(src_w * r)
            crop = pm[:, :int(nh/4.0), :int(nw/4.0)]
        else:
            crop = pm

        bn = crop.to(torch.float32)
        if crop.shape[-2:] != self._src_shape:
            bn = torch.nn.functional.interpolate(bn.unsqueeze(0), size=self._src_shape, mode="bilinear", align_corners=False).squeeze(0)
        bn = (bn > 0.0).to(torch.bool).detach().cpu().numpy()
        sn = ps.detach().cpu().numpy() if ps is not None else np.ones(bn.shape[0])

        if multimask_output and len(sn) > 1:
            bi = int(sn.argmax())
            return {"masks": bn[bi:bi+1], "scores": sn[bi:bi+1], "low_res_masks": raw[bi:bi+1]}
        return {"masks": bn, "scores": sn, "low_res_masks": raw}

    def predict_batch(self, bboxes: list[list[float]]) -> list[np.ndarray]:
        if not self._image_set:
            raise RuntimeError("No image set.")
        if not bboxes:
            return []
        try:
            return self._predict_batch_batched(bboxes)
        except Exception as e:
            LOGGER.debug("Batched SAM2 predict failed (%s); falling back to per-box", e)
            masks: list[np.ndarray] = []
            for bbox in bboxes:
                r = self.predict(bboxes=bbox)
                masks.append(r["masks"][0])
            return masks

    def _predict_batch_batched(self, bboxes: list[list[float]]) -> list[np.ndarray]:
        features = self._predictor.features
        if features is None:
            raise RuntimeError("No cached features.")
        src_h, src_w = self._src_shape
        dtype = self._model_dtype()

        r = min(1024 / src_h, 1024 / src_w)
        b = len(bboxes)
        coords = []
        for box in bboxes:
            x1, y1, x2, y2 = (c * r for c in box)
            coords.extend([[x1, y1], [x2, y2]])
        pts = torch.tensor(coords, dtype=dtype, device=self._predictor.device).reshape(b, 2, 2)
        lbs = torch.full((b, 2), 3, dtype=torch.int32, device=self._predictor.device)
        lbs[:, 0] = 2

        img_embed = features["image_embed"]
        high_res = features["high_res_feats"]
        with self._infer_lock:
            if self._predictor is None:
                raise RuntimeError("Model not loaded.")
            image_pe = self._image_pe
            if image_pe is None:
                image_pe = self._predictor.model.sam_prompt_encoder.get_dense_pe()
                self._image_pe = image_pe
        point_inputs = (pts, lbs)

        with self._infer_lock, torch.inference_mode(), cpu_threads(_DEFAULT_THREADS if self._device == "cpu" else 1), cudnn_disabled():
            se, de = self._predictor.model.sam_prompt_encoder(points=point_inputs, boxes=None, masks=None)
            pm, _, _, _ = self._predictor.model.sam_mask_decoder(
                image_embeddings=img_embed.expand(b, -1, -1, -1),
                image_pe=image_pe,
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False,
                repeat_image=False, high_res_features=high_res,
            )
        pm = pm.flatten(0, 1)

        if pm is None or pm.shape[0] == 0:
            return [np.zeros((src_h, src_w), dtype=bool) for _ in range(b)]

        if self._src_shape:
            nh, nw = round(src_h * r), round(src_w * r)
            crop = pm[:, :int(nh/4.0), :int(nw/4.0)]
        else:
            crop = pm

        bn = crop.to(torch.float32)
        if crop.shape[-2:] != self._src_shape:
            bn = torch.nn.functional.interpolate(bn.unsqueeze(0), size=self._src_shape, mode="bilinear", align_corners=False).squeeze(0)
        bn = (bn > 0.0).to(torch.bool).detach().cpu().numpy()
        return [bn[i] for i in range(b)]

    def reset_image(self) -> None:
        if self._predictor is not None:
            with self._infer_lock:
                self._predictor.reset_image()
        self._image_pe = None
        self._image_set = False
        self._src_shape = None

    @property
    def is_loaded(self) -> bool: return self._predictor is not None

    @property
    def has_image(self) -> bool: return self._image_set
