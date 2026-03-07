"""
inference/predict.py — Run OCR Prediction on a Brahmi Script Image
===================================================================

Load a fine-tuned TrOCR model from  model/brahmi_trocr/  and predict
Brahmi Unicode text from an input image.

USAGE:
    python inference/predict.py --image path/to/image.png
    python inference/predict.py --image path/to/image.png --model_dir model/brahmi_trocr
    python inference/predict.py --image path/to/image.png --preprocess
"""

import os
import sys
import argparse
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_trained_model(model_dir: str, device: str = None):
    """
    Load a fine-tuned TrOCR model and processor from disk.

    The saved tokenizer already contains the added Brahmi tokens,
    so no need to re-add them here.

    Args:
        model_dir: Path to the saved model directory.
        device:    'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        processor, model, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained(model_dir, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Ensure generation config has the required params
    if not hasattr(model.generation_config, 'decoder_start_token_id') or \
       model.generation_config.decoder_start_token_id is None:
        model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    if not hasattr(model.generation_config, 'eos_token_id') or \
       model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    if not hasattr(model.generation_config, 'pad_token_id') or \
       model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    return processor, model, device


def predict(image_path: str, processor, model, device: str,
            preprocess: bool = False) -> str:
    """
    Run OCR prediction on a single image.

    Args:
        image_path: Path to the input image.
        processor:  TrOCRProcessor.
        model:      Fine-tuned VisionEncoderDecoderModel.
        device:     'cuda' or 'cpu'.
        preprocess: If True, apply utils.preprocess pipeline first.

    Returns:
        Predicted Brahmi Unicode text.
    """
    import cv2
    import numpy as np

    def upscale_small_image(image: Image.Image, min_short_side: int = 128) -> Image.Image:
        w, h = image.size
        short_side = min(w, h)
        if short_side >= min_short_side:
            return image
        scale = min_short_side / max(short_side, 1)
        new_w = max(int(round(w * scale)), 1)
        new_h = max(int(round(h * scale)), 1)
        return image.resize((new_w, new_h), resample=Image.BICUBIC)

    def merge_strings(s1: str, s2: str, max_overlap: int = 12) -> str:
        s1 = s1.strip()
        s2 = s2.strip()
        if not s1:
            return s2
        if not s2:
            return s1

        overlap = min(len(s1), len(s2), max_overlap)
        for i in range(overlap, 0, -1):
            if s1[-i:] == s2[:i]:
                return (s1 + s2[i:]).strip()
        return f"{s1} {s2}".strip()

    def prepare_image(image: Image.Image) -> Image.Image:
        image = upscale_small_image(image)
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if preprocess:
            from utils.preprocess import preprocess_array
            image_rgb = preprocess_array(
                image_bgr,
                target_size=(384, 384),
                noise_method="gaussian",
                threshold_method="adaptive",
            )
        else:
            from utils.preprocess import resize_image
            image_bgr = resize_image(image_bgr, target_size=(384, 384))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)

    def decode_image(image: Image.Image) -> str:
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=256,
                num_beams=1,
                length_penalty=1.0,
            )
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Load raw image first; chunking must happen before any 384x384 resize.
    full_image = Image.open(image_path).convert("RGB")
    full_image = upscale_small_image(full_image)
    width, height = full_image.size
    aspect_ratio = width / max(height, 1)

    # Chunk wide images to avoid destructive horizontal squishing.
    if aspect_ratio >= 2.2:
        chunk_span = max(int(height * 2.2), height)
        stride = max(int(height * 1.8), 1)  # overlap preserves boundary glyphs
        left = 0
        merged_text = ""

        while left < width:
            right = min(left + chunk_span, width)
            chunk = full_image.crop((left, 0, right, height))
            piece = decode_image(prepare_image(chunk))
            merged_text = merge_strings(merged_text, piece)
            if right >= width:
                break
            left += stride

        return merged_text.strip()

    # Short/normal image path
    processed = prepare_image(full_image)
    return decode_image(processed).strip()


def main():
    parser = argparse.ArgumentParser(description="Brahmi OCR — Inference")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--model_dir", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "model", "brahmi_trocr"),
                        help="Path to the saved model directory")
    parser.add_argument("--preprocess", action="store_true",
                        help="Apply image preprocessing before OCR")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_dir}")
    processor, model, device = load_trained_model(args.model_dir)

    print(f"Running OCR on: {args.image}")
    text = predict(args.image, processor, model, device, args.preprocess)

    print(f"\n{'='*50}")
    print(f"Predicted Brahmi text:")
    print(f"  {text}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
