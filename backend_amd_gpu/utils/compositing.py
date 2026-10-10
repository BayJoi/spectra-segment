from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image


def mask_to_png_b64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", _to_mask_u8(mask))
    if not ok or buf is None:
        raise ValueError("cv2.imencode failed to encode the mask")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _is_unit_scale(arr: np.ndarray) -> bool:
    finite = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    return float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0


def _to_mask_f32(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.floating):
        if _is_unit_scale(arr):
            return arr.astype(np.float32)
        return (np.clip(np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0) / 255.0).astype(np.float32)
    return (np.clip(arr.astype(np.float64), 0.0, 255.0) / 255.0).astype(np.float32)


def _to_mask_u8(mask: np.ndarray) -> np.ndarray:
    return np.clip(_to_mask_f32(mask) * 255.0, 0, 255).astype(np.uint8)


def feather_mask_edge(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    mask_f = _to_mask_f32(mask)
    if radius < 1:
        return mask_f
    ksize = radius * 2 + 1
    padded = np.pad(mask_f, radius, mode="constant", constant_values=0.0)
    blurred = cv2.GaussianBlur(padded, (ksize, ksize), 0)
    h, w = mask_f.shape
    return blurred[radius : radius + h, radius : radius + w]


def composite_rgba(image_rgb: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    alpha = np.clip(_to_mask_f32(soft_mask) * 255.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([image_rgb, alpha])
    return rgba


def composite_background(
    image_rgb: np.ndarray,
    soft_mask: np.ndarray,
    background_color: tuple[int, int, int] | None = None,
    bg_image: np.ndarray | None = None,
) -> np.ndarray:
    bg = bg_image if bg_image is not None else np.full_like(image_rgb, background_color or (255, 255, 255))
    alpha = _to_mask_f32(soft_mask)[..., None]
    result = (
        image_rgb.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)
    return result


def composite_and_encode(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    output_format: str = "png",
    background_color: tuple[int, int, int] | None = None,
    feather_radius: int = 3,
    bg_image: np.ndarray | None = None,
) -> bytes:
    soft = feather_mask_edge(mask, feather_radius)
    buf = io.BytesIO()

    if background_color is not None:
        result = composite_background(image_rgb, soft, background_color=background_color, bg_image=bg_image)
        pil_img = Image.fromarray(result, "RGB")
        if output_format in ("jpg", "jpeg"):
            pil_img.save(buf, format="JPEG", quality=95)
        else:
            pil_img.save(buf, format="PNG")
    else:
        result = composite_rgba(image_rgb, soft)
        pil_img = Image.fromarray(result, "RGBA")
        pil_img.save(buf, format="PNG")

    return buf.getvalue()
