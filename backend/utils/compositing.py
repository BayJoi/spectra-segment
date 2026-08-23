from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image


def mask_to_png_b64(mask: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", (mask.astype(np.uint8) * 255))
    return base64.b64encode(buf.tobytes()).decode("ascii")


def feather_mask_edge(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    if radius < 1:
        return mask_f
    ksize = radius * 2 + 1
    padded = np.pad(mask_f, radius, mode="constant", constant_values=0.0)
    blurred = cv2.GaussianBlur(padded, (ksize, ksize), 0)
    h, w = mask_f.shape
    return blurred[radius : radius + h, radius : radius + w]


def composite_rgba(image_rgb: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    alpha = (soft_mask * 255).clip(0, 255).astype(np.uint8)
    rgba = np.dstack([image_rgb, alpha])
    return rgba


def composite_background(
    image_rgb: np.ndarray,
    soft_mask: np.ndarray,
    background_color: tuple[int, int, int] | None = None,
    bg_image: np.ndarray | None = None,
) -> np.ndarray:
    bg = bg_image if bg_image is not None else np.full_like(image_rgb, background_color or (255, 255, 255))
    alpha = soft_mask[..., None]
    result = (image_rgb * alpha + bg * (1 - alpha)).clip(0, 255).astype(np.uint8)
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
