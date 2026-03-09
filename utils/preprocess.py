"""
utils/preprocess.py
===================

Image preprocessing utilities for Brahmi OCR.

Goals:
1) Keep glyph geometry intact with aspect-ratio-safe square padding.
2) Provide optional step-by-step diagnostics for backend/Web UI display.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if method == "median":
        return cv2.medianBlur(image, 5)
    if method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    raise ValueError(f"Unknown denoising method: {method}")


def apply_threshold(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if method == "simple":
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return binary
    raise ValueError(f"Unknown thresholding method: {method}")


def enhance_contrast(
    image: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def resize_with_padding(
    image: np.ndarray, target_size: tuple = (384, 384), background: int = 255
) -> tuple[np.ndarray, dict]:
    """
    Resize while preserving aspect ratio, then center-pad to target size.
    Returns both the resized image and geometry metadata.
    """
    tw, th = target_size
    h, w = image.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Invalid image shape for resize_with_padding")

    scale = min(tw / w, th / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    if len(image.shape) == 2:
        canvas = np.full((th, tw), background, dtype=np.uint8)
    else:
        channels = image.shape[2]
        canvas = np.full((th, tw, channels), background, dtype=np.uint8)

    x_off = (tw - new_w) // 2
    y_off = (th - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    meta = {
        "original_width": int(w),
        "original_height": int(h),
        "target_width": int(tw),
        "target_height": int(th),
        "resized_width": int(new_w),
        "resized_height": int(new_h),
        "x_offset": int(x_off),
        "y_offset": int(y_off),
        "scale": float(scale),
        "aspect_ratio_preserved": True,
    }
    return canvas, meta


def _pil_resampling_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def letterbox_pil(
    image: Image.Image,
    target_size: tuple = (384, 384),
    fill_color: tuple = (255, 255, 255),
    return_meta: bool = False,
):
    """
    PIL letterbox utility used by training/inference to avoid image squishing.
    """
    image = image.convert("RGB")
    tw, th = target_size
    ow, oh = image.size
    if ow <= 0 or oh <= 0:
        raise ValueError("Invalid PIL image size for letterbox_pil")

    scale = min(tw / ow, th / oh)
    nw = max(1, int(round(ow * scale)))
    nh = max(1, int(round(oh * scale)))
    resized = image.resize((nw, nh), _pil_resampling_filter())

    canvas = Image.new("RGB", (tw, th), fill_color)
    x_off = (tw - nw) // 2
    y_off = (th - nh) // 2
    canvas.paste(resized, (x_off, y_off))

    if not return_meta:
        return canvas

    meta = {
        "original_width": int(ow),
        "original_height": int(oh),
        "target_width": int(tw),
        "target_height": int(th),
        "resized_width": int(nw),
        "resized_height": int(nh),
        "x_offset": int(x_off),
        "y_offset": int(y_off),
        "scale": float(scale),
        "aspect_ratio_preserved": True,
    }
    return canvas, meta


def _array_stats(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def preprocess_image(
    path: str,
    target_size: tuple = (384, 384),
    threshold_method: str | None = None,
    return_debug: bool = False,
):
    """
    Full preprocessing pipeline.

    Steps:
      1) load image
      2) grayscale
      3) denoise
      4) contrast enhancement
      5) optional threshold
      6) resize with aspect-ratio-preserving square padding
      7) convert to RGB PIL image

    Returns:
      - PIL.Image if return_debug=False
      - (PIL.Image, debug_dict) if return_debug=True
    """
    img = load_image(path)
    gray = to_grayscale(img)
    denoised = remove_noise(gray, method="gaussian")
    enhanced = enhance_contrast(denoised)
    processed = enhanced

    debug = {
        "input_path": path,
        "target_size": {"width": int(target_size[0]), "height": int(target_size[1])},
        "steps": [],
        "padding": {},
    }

    if return_debug:
        debug["steps"].append({"name": "loaded_bgr", "stats": _array_stats(img)})
        debug["steps"].append({"name": "grayscale", "stats": _array_stats(gray)})
        debug["steps"].append({"name": "denoise_gaussian", "stats": _array_stats(denoised)})
        debug["steps"].append({"name": "clahe_contrast", "stats": _array_stats(enhanced)})

    if threshold_method:
        processed = apply_threshold(processed, method=threshold_method)
        if return_debug:
            debug["steps"].append(
                {
                    "name": f"threshold_{threshold_method}",
                    "stats": _array_stats(processed),
                }
            )

    resized, pad_meta = resize_with_padding(processed, target_size=target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(rgb)

    if return_debug:
        debug["steps"].append({"name": "resize_with_padding", "stats": _array_stats(resized)})
        debug["steps"].append({"name": "final_rgb", "stats": _array_stats(rgb)})
        debug["padding"] = pad_meta
        return pil_img, debug

    return pil_img
