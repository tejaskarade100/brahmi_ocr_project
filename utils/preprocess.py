"""
utils/preprocess.py — Image Preprocessing Utilities for Brahmi OCR
===================================================================

Provide helper functions that clean and enhance Brahmi script images
before they are fed to the TrOCR model.  Stone inscriptions and old
manuscripts often have noise, uneven lighting, and low contrast.

Preprocessing patterns are informed by the Capstone Brahmi Inscriptions
project (Gaussian blur, Otsu threshold, morphological operations).

DEPENDENCIES:
    opencv-python (cv2), Pillow, numpy
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk as a BGR numpy array.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        image: BGR numpy array.

    Raises:
        FileNotFoundError: if the image cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR colour image to single-channel grayscale.

    If the image is already single-channel it is returned unchanged.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """
    Remove noise from a grayscale image.

    Args:
        image:  Grayscale image.
        method: 'gaussian' | 'median' | 'bilateral' | 'nlm'

    Returns:
        Denoised image.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(image, 5)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == "nlm":
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def apply_threshold(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Binarise the image using thresholding.

    Args:
        image:  Grayscale (or denoised) image.
        method: 'adaptive' | 'otsu' | 'simple'

    Returns:
        Binary (black & white) image.
    """
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
    elif method == "otsu":
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return binary
    elif method == "simple":
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return binary
    else:
        raise ValueError(f"Unknown thresholding method: {method}")


def binarize_image(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Convenience wrapper: grayscale conversion + thresholding.

    Args:
        image:  BGR or grayscale image.
        method: Threshold method passed to apply_threshold().

    Returns:
        Binarized single-channel image.
    """
    gray = to_grayscale(image)
    return apply_threshold(gray, method=method)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0,
                     grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram
    Equalisation).

    Args:
        image:      Grayscale image.
        clip_limit: Threshold for contrast limiting.
        grid_size:  Size of the local region for histogram equalisation.

    Returns:
        Contrast-enhanced image.
    """
    if len(image.shape) == 3:
        image = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def resize_image(image: np.ndarray,
                 target_size: tuple = (384, 384)) -> np.ndarray:
    """
    Resize the image to *target_size* while preserving aspect ratio.
    The shorter dimension is scaled to fit, then the result is centred
    on a white canvas of the exact target size.

    Args:
        image:       Input image (grayscale or BGR).
        target_size: (width, height).

    Returns:
        Resized image.
    """
    tw, th = target_size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a white canvas and paste the resized image in the centre
    if len(image.shape) == 2:
        canvas = np.full((th, tw), 255, dtype=np.uint8)
    else:
        canvas = np.full((th, tw, image.shape[2]), 255, dtype=np.uint8)

    y_off = (th - new_h) // 2
    x_off = (tw - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def preprocess_array(
    image: np.ndarray,
    target_size: tuple = (384, 384),
    noise_method: str = "gaussian",
    threshold_method: Optional[str] = None,
) -> np.ndarray:
    """
    Preprocess an in-memory image and return RGB array ready for PIL conversion.

    Args:
        image:            Input image (BGR or grayscale).
        target_size:      Output size (width, height).
        noise_method:     Denoising method for remove_noise().
        threshold_method: Optional threshold mode; if None, keep grayscale.

    Returns:
        RGB numpy array.
    """
    gray = to_grayscale(image)
    denoised = remove_noise(gray, method=noise_method)
    enhanced = enhance_contrast(denoised)

    processed = enhanced
    if threshold_method:
        processed = apply_threshold(enhanced, method=threshold_method)

    resized = resize_image(processed, target_size)
    if len(resized.shape) == 2:
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def preprocess_image(path: str, target_size: tuple = (384, 384)) -> Image.Image:
    """
    Full preprocessing pipeline.

    Steps:
        1. Load image from path.
        2. Convert to grayscale.
        3. Remove noise (Gaussian blur).
        4. Enhance contrast (CLAHE).
        5. Resize to model input size.

    Args:
        path:        Path to the raw image file.
        target_size: (width, height) for the output.

    Returns:
        Preprocessed image as a PIL RGB Image, ready for TrOCR processor.
    """
    img = load_image(path)
    rgb = preprocess_array(
        img,
        target_size=target_size,
        noise_method="gaussian",
        threshold_method=None,
    )
    return Image.fromarray(rgb)
