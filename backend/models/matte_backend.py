from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from ..utils.device import MODEL_WEIGHTS_DIR, cpu_fallback_enabled, release_gpu_memory
from ..utils.stderr_progress import StderrInterceptor
from ..utils.torch_threads import cpu_threads, resolve_threads

LOGGER = logging.getLogger(__name__)

_LOAD_LOCK = threading.Lock()
_DEFAULT_THREADS = resolve_threads("SAM2_THREADS")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class MatteModelSpec:
    name: str
    repo_id: str
    revision: str
    display_name: str
    input_size: int
    tier: str
    perf: str
    soft_alpha: bool
    weight_mb: int


MATTE_MODELS: dict[str, MatteModelSpec] = {
    "birefnet-general": MatteModelSpec(
        name="birefnet-general",
        repo_id="ZhengPeng7/BiRefNet",
        revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
        display_name="BiRefNet General",
        input_size=1024,
        tier="medium",
        perf="Medium - best all-round cutout",
        soft_alpha=True,
        weight_mb=423,
    ),
    "birefnet-lite": MatteModelSpec(
        name="birefnet-lite",
        repo_id="ZhengPeng7/BiRefNet_lite",
        revision="aa62cd87eafb9cc43056d08ef3615a14628b831d",
        display_name="BiRefNet Lite",
        input_size=1024,
        tier="tiny",
        perf="Light - fastest",
        soft_alpha=True,
        weight_mb=169,
    ),
    "birefnet-matting": MatteModelSpec(
        name="birefnet-matting",
        repo_id="ZhengPeng7/BiRefNet-matting",
        revision="eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9",
        display_name="BiRefNet Matting",
        input_size=1024,
        tier="large",
        perf="Heavy - hair and fine edges",
        soft_alpha=True,
        weight_mb=843,
    ),
    "birefnet-dynamic": MatteModelSpec(
        name="birefnet-dynamic",
        repo_id="ZhengPeng7/BiRefNet_dynamic",
        revision="280306042f57b7a33854319da62fd86aaa89ec4c",
        display_name="BiRefNet Dynamic",
        input_size=1024,
        tier="medium",
        perf="Medium - dynamic-resolution training",
        soft_alpha=True,
        weight_mb=424,
    ),
    "birefnet-dynamic-matting": MatteModelSpec(
        name="birefnet-dynamic-matting",
        repo_id="ZhengPeng7/BiRefNet_dynamic-matting",
        revision="074df545be87034e74a96bf71566ecbbc4c15f0a",
        display_name="BiRefNet Dynamic Matting",
        input_size=1024,
        tier="large",
        perf="Heavy - soft edges across resolutions",
        soft_alpha=True,
        weight_mb=424,
    ),
    "birefnet-lite-matting": MatteModelSpec(
        name="birefnet-lite-matting",
        repo_id="ZhengPeng7/BiRefNet_lite-matting",
        revision="99c33412e3f58e1f33187abdc8c435c645243690",
        display_name="BiRefNet Lite Matting",
        input_size=1024,
        tier="tiny",
        perf="Light - fast soft edges",
        soft_alpha=True,
        weight_mb=85,
    ),
    "birefnet-hr": MatteModelSpec(
        name="birefnet-hr",
        repo_id="ZhengPeng7/BiRefNet_HR",
        revision="a7a562f6fd16021180f2f4348f4de003a2d3d1e1",
        display_name="BiRefNet HR",
        input_size=2048,
        tier="large",
        perf="Heavy - 2048px detail",
        soft_alpha=True,
        weight_mb=424,
    ),
    "birefnet-hr-matting": MatteModelSpec(
        name="birefnet-hr-matting",
        repo_id="ZhengPeng7/BiRefNet_HR-matting",
        revision="5d6b6f8adcb5b417c871b1d84ceaae9871355b7f",
        display_name="BiRefNet HR Matting",
        input_size=2048,
        tier="large",
        perf="Heavy - 2048px hair and glass",
        soft_alpha=True,
        weight_mb=424,
    ),
    "birefnet-portrait": MatteModelSpec(
        name="birefnet-portrait",
        repo_id="ZhengPeng7/BiRefNet-portrait",
        revision="b6561965a70070d9143fd9e558f6ca3c481510db",
        display_name="BiRefNet Portrait",
        input_size=1024,
        tier="medium",
        perf="Medium - people and portrait edges",
        soft_alpha=True,
        weight_mb=844,
    ),
    "birefnet-512": MatteModelSpec(
        name="birefnet-512",
        repo_id="ZhengPeng7/BiRefNet_512x512",
        revision="678c37f9158280a2b3000b566ecd12bfbfd1a09c",
        display_name="BiRefNet 512 Draft",
        input_size=512,
        tier="tiny",
        perf="Light - 512px draft - fastest",
        soft_alpha=True,
        weight_mb=424,
    ),
}

_FP16 = torch.cuda.is_available() and os.environ.get("SPECTRA_MATTE_FP16", "1") == "1"


def _is_oom_error(exc: Exception) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


MAX_GUIDED_BOXES = 4
MAX_STROKE_POINTS = 4096


def matte_model_choices() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "display_name": spec.display_name,
            "input_size": spec.input_size,
            "tier": spec.tier,
            "perf": spec.perf,
            "soft_alpha": spec.soft_alpha,
            "weight_mb": spec.weight_mb,
            "downloaded": _is_downloaded(spec),
        }
        for spec in MATTE_MODELS.values()
    ]


def _hub_cache_dir():
    return MODEL_WEIGHTS_DIR / "hf_cache" / "hub"


def _is_downloaded(spec: MatteModelSpec) -> bool:
    cache = _hub_cache_dir()
    repo_dir = cache / ("models--" + spec.repo_id.replace("/", "--"))
    if not repo_dir.exists():
        return False
    snapshots = repo_dir / "snapshots" / spec.revision
    return snapshots.exists() and any(snapshots.glob("*.safetensors"))


class BiRefNetBackend:
    def __init__(self, spec: MatteModelSpec, device: str = "cpu") -> None:
        self._spec = spec
        self._device = device
        self._model: Any = None
        self._input_size_override: int | None = None
        self._infer_lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def spec(self) -> MatteModelSpec:
        return self._spec

    def load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForImageSegmentation

        display = self._spec.display_name
        downloaded = _is_downloaded(self._spec)
        LOGGER.info(
            "Downloading matte model %s" if not downloaded else "Loading matte model %s",
            display,
            extra={"phase": "matte-load", "model": self._spec.name},
        )
        interceptor = StderrInterceptor(display)
        old_stderr = sys.stderr
        sys.stderr = interceptor
        progress = None
        try:
            from huggingface_hub.utils import enable_progress_bars, disable_progress_bars
            enable_progress_bars()
            progress = disable_progress_bars
        except Exception:
            progress = None
        try:
            with _LOAD_LOCK:
                load_kwargs = dict(
                    revision=self._spec.revision,
                    trust_remote_code=True,
                    cache_dir=str(_hub_cache_dir()),
                )
                model = None
                if downloaded:
                    try:
                        model = AutoModelForImageSegmentation.from_pretrained(
                            self._spec.repo_id,
                            local_files_only=True,
                            **load_kwargs,
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "Cached matte model %s could not load offline (%s); retrying the pinned download",
                            display,
                            exc,
                        )
                if model is None:
                    model = AutoModelForImageSegmentation.from_pretrained(
                        self._spec.repo_id,
                        local_files_only=False,
                        **load_kwargs,
                    )
                model.eval()
                actual_device = self._device
                if actual_device != "cpu":
                    if not torch.cuda.is_available():
                        actual_device = "cpu"
                    else:
                        try:
                            model.to(actual_device)
                        except Exception as exc:
                            if not cpu_fallback_enabled():
                                raise RuntimeError(
                                    f"Could not place matte model on {actual_device}: {exc}"
                                ) from exc
                            LOGGER.warning(
                                "Matte device %s failed (%s), falling back to CPU",
                                actual_device,
                                exc,
                            )
                            actual_device = "cpu"
                self._device = actual_device
                self._model = model
        finally:
            sys.stderr = old_stderr
            interceptor.flush()
            if progress is not None:
                progress()
        LOGGER.info(
            "Matte model %s is ready on %s",
            display,
            self._device,
            extra={"phase": "residency", "model": self._spec.name},
        )

    def unload_model(self) -> None:
        with self._infer_lock:
            self._model = None
        release_gpu_memory()

    def _fallback_cpu(self) -> None:
        self.unload_model()
        self._device = "cpu"
        self.load_model()

    def _input_size(self) -> int:
        if self._input_size_override is not None:
            return max(512, min(self._input_size_override, self._spec.input_size))
        raw = os.environ.get("SPECTRA_MATTE_INPUT_SIZE", "").strip()
        try:
            requested = int(raw) if raw else self._spec.input_size
        except ValueError:
            requested = self._spec.input_size
        return max(512, min(requested, self._spec.input_size))

    def _retry_lower_input(self, crops: list[np.ndarray]) -> list[np.ndarray] | None:
        current = self._input_size()
        for size in (768, 512):
            if size >= current:
                continue
            LOGGER.warning(
                "Matte inference ran out of VRAM on %s at %d px; retrying at %d px",
                self._device,
                current,
                size,
                extra={"phase": "vram", "reason": "oom-retry", "model": self._spec.name},
            )
            self._input_size_override = size
            release_gpu_memory()
            try:
                return self._forward(crops)
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                current = size
                release_gpu_memory()
        return None

    def _forward(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        model = self._model
        if model is None:
            raise RuntimeError("Matte model not loaded.")
        size = self._input_size()
        batch = np.empty((len(crops), size, size, 3), dtype=np.float32)
        for i, crop in enumerate(crops):
            resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            batch[i] = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        if self._device != "cpu" and torch.cuda.is_available():
            tensor = tensor.to(self._device, non_blocking=True)

        try:
            with self._infer_lock, torch.inference_mode():
                use_gpu = self._device != "cpu" and torch.cuda.is_available()
                with cpu_threads(_DEFAULT_THREADS):
                    if use_gpu and _FP16:
                        with torch.autocast("cuda", dtype=torch.float16):
                            output = model(tensor)
                    else:
                        output = model(tensor)
        except RuntimeError as exc:
            if self._device == "cpu" or not _is_oom_error(exc):
                raise
            if not cpu_fallback_enabled():
                LOGGER.error(
                    "Matte inference exhausted VRAM on %s; automatic CPU fallback is disabled",
                    self._device,
                    extra={"phase": "vram", "reason": "oom-no-fallback", "model": self._spec.name},
                )
                raise
            retried = self._retry_lower_input(crops)
            if retried is not None:
                return retried
            LOGGER.warning(
                "Matte inference exhausted VRAM on %s; falling back to CPU",
                self._device,
                extra={"phase": "vram", "reason": "oom-cpu-opt-in", "model": self._spec.name},
            )
            self._fallback_cpu()
            return self._forward(crops)

        logits = output[-1] if isinstance(output, (list, tuple)) else output
        if hasattr(logits, "logits"):
            logits = logits.logits
        alpha = logits.float().sigmoid()
        if alpha.ndim == 3:
            alpha = alpha.unsqueeze(1)

        out: list[np.ndarray] = []
        for i, crop in enumerate(crops):
            ch, cw = crop.shape[:2]
            single = torch.nn.functional.interpolate(
                alpha[i : i + 1], size=(ch, cw), mode="bilinear", align_corners=False
            )
            out.append(single[0, 0].cpu().numpy().astype(np.float32, copy=False))
        return out

    def matte(
        self,
        image_rgb: np.ndarray,
        boxes: list[list[float]] | None = None,
        crop_margin: float = 0.08,
    ) -> np.ndarray:
        if boxes:
            height, width = image_rgb.shape[:2]
            cap = self._input_size()
            if cap < self._spec.input_size and max(height, width) > cap * 3 // 2:
                return self._crop_matte(image_rgb, boxes, crop_margin)
        alpha = self._forward([image_rgb])[0]
        if not boxes:
            return np.clip(alpha, 0.0, 1.0)

        keep = _roi_component_mask(alpha, boxes)
        if keep is not None:
            return np.clip(alpha * keep, 0.0, 1.0)
        return self._crop_matte(image_rgb, boxes, crop_margin)

    def _crop_matte(
        self,
        image_rgb: np.ndarray,
        boxes: list[list[float]],
        crop_margin: float,
    ) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        full = np.zeros((h, w), dtype=np.float32)
        for box in boxes[:MAX_GUIDED_BOXES]:
            x1, y1, x2, y2 = _context_box(box, w, h, crop_margin)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            alpha = self._forward([image_rgb[y1:y2, x1:x2]])[0]
            region = full[y1:y2, x1:x2]
            np.maximum(region, alpha, out=region)
        return np.clip(full, 0.0, 1.0)


def _roi_component_mask(
    alpha: np.ndarray, boxes: list[list[float]]
) -> np.ndarray | None:
    h, w = alpha.shape[:2]
    count, labels = cv2.connectedComponents((alpha > 0.5).astype(np.uint8), connectivity=8)
    if count <= 1:
        return None

    box_mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes[:MAX_GUIDED_BOXES]:
        x1, y1, x2, y2 = _expanded_box(box, w, h, 0.02)
        box_mask[y1:y2, x1:x2] = 1

    flat = labels.ravel()
    overlap = np.bincount(flat, weights=box_mask.ravel(), minlength=count)
    keep_ids = [i for i in range(1, count) if overlap[i] > 0]
    if not keep_ids:
        return None

    keep = np.isin(labels, keep_ids).astype(np.uint8) * 255
    keep = cv2.dilate(keep, np.ones((9, 9), np.uint8))
    keep = cv2.GaussianBlur(keep.astype(np.float32) / 255.0, (0, 0), 3.0)
    return np.clip(keep * 1.35, 0.0, 1.0)


def _context_box(
    box: list[float], width: int, height: int, margin: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    span = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * max(0.15, min(0.5, margin)))
    span = max(span, min(320.0, float(min(width, height))))
    half = span / 2.0
    return (
        max(0, int(center_x - half)),
        max(0, int(center_y - half)),
        min(width, int(center_x + half)),
        min(height, int(center_y + half)),
    )


def _expanded_box(
    box: list[float], width: int, height: int, margin: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    pad_x = max(0.0, min(0.5, margin)) * max(1.0, x2 - x1)
    pad_y = max(0.0, min(0.5, margin)) * max(1.0, y2 - y1)
    return (
        max(0, int(np.floor(x1 - pad_x))),
        max(0, int(np.floor(y1 - pad_y))),
        min(width, int(np.ceil(x2 + pad_x))),
        min(height, int(np.ceil(y2 + pad_y))),
    )


def points_bbox(points: list[list[float]], labels: list[int] | None) -> list[float] | None:
    positives = [
        p for i, p in enumerate(points) if labels is None or i >= len(labels) or labels[i] == 1
    ]
    if not positives:
        return None
    xs = [p[0] for p in positives]
    ys = [p[1] for p in positives]
    return [min(xs), min(ys), max(xs), max(ys)]


def sample_points(
    points: list[list[float]], labels: list[int] | None
) -> tuple[list[list[float]], list[int] | None]:
    if len(points) <= MAX_STROKE_POINTS:
        return points, labels
    keep = sorted(
        set(np.round(np.linspace(0, len(points) - 1, MAX_STROKE_POINTS)).astype(int).tolist())
    )
    sampled = [points[i] for i in keep]
    sampled_labels = (
        [labels[i] for i in keep] if labels is not None and len(labels) == len(points) else labels
    )
    return sampled, sampled_labels


def subtract_negative_strokes(
    alpha: np.ndarray,
    points: list[list[float]],
    labels: list[int] | None,
    radius: int,
) -> np.ndarray:
    if not points or labels is None:
        return alpha
    negatives = [p for p, lb in zip(points, labels) if lb == 0]
    if not negatives:
        return alpha

    mask = np.zeros(alpha.shape, dtype=np.uint8)
    thickness = max(1, int(radius) * 2)
    for i in range(1, len(points)):
        if labels[i] != 0 or labels[i - 1] != 0:
            continue
        cv2.line(
            mask,
            (int(points[i - 1][0]), int(points[i - 1][1])),
            (int(points[i][0]), int(points[i][1])),
            255,
            thickness,
            cv2.LINE_AA,
        )
    for x, y in negatives:
        cv2.circle(mask, (int(x), int(y)), max(1, int(radius)), 255, -1, cv2.LINE_AA)

    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), max(1.0, radius / 2))
    return alpha * (1.0 - np.clip(soft, 0.0, 1.0))


def apply_alpha_controls(
    alpha: np.ndarray,
    edge_shift: int = 0,
    edge_feather: int = 0,
    invert: bool = False,
) -> np.ndarray:
    out = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    if edge_shift:
        out = np.clip(out + float(edge_shift) / 100.0, 0.0, 1.0)
    if invert:
        out = 1.0 - out
    if edge_feather > 0:
        radius = max(0, int(edge_feather))
        out = cv2.GaussianBlur(out, (0, 0), max(0.5, radius / 2.0))
    return np.clip(out, 0.0, 1.0)


def alpha_to_png_b64(alpha: np.ndarray) -> str:
    import base64

    data = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", data)
    if not ok or buf is None:
        raise ValueError("Failed to encode alpha mask")
    return base64.b64encode(buf.tobytes()).decode("ascii")
