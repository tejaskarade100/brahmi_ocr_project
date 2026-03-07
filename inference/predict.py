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
    import math
    import cv2
    import numpy as np
    
    # Load raw image first before preprocessing squishes it
    full_image = Image.open(image_path).convert("RGB")
    width, height = full_image.size
    aspect_ratio = width / height

    def merge_strings(s1, s2):
        if not s1: return s2
        if not s2: return s1
        # Check overlaps from 6 chars down to 1 (accounts for chunk boundary overlaps)
        max_overlap = min(len(s1), len(s2), 6)
        for i in range(max_overlap, 0, -1):
            if s1[-i:] == s2[:i]:
                return s1 + s2[i:]
        return s1 + " " + s2

    # If the image is a wide phrase, chunk it horizontally to prevent 384x384 squishing!
    if aspect_ratio > 3.0:
        num_chunks = int(math.ceil(aspect_ratio / 2.0))  # Each chunk is twice as wide as it is tall
        chunk_width = width // num_chunks
        predicted_text = ""
        
        for i in range(num_chunks):
            left = i * chunk_width
            right = min(left + chunk_width, width)
            # Add a small overlap
            if i > 0: left = max(left - int(height * 0.2), 0)
            if i < num_chunks - 1: right = min(right + int(height * 0.2), width)
                
            chunk = full_image.crop((left, 0, right, height))
            
            if preprocess:
                from utils.preprocess import binarize_image, enhance_contrast, resize_image
                chunk_cv = cv2.cvtColor(np.array(chunk), cv2.COLOR_RGB2BGR)
                chunk_cv = binarize_image(chunk_cv, method='adaptive')
                chunk_cv = enhance_contrast(chunk_cv)
                chunk_cv = resize_image(chunk_cv, target_size=(384, 384))
                chunk = Image.fromarray(cv2.cvtColor(chunk_cv, cv2.COLOR_BGR2RGB))
                
            pixel_values = processor(chunk, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=256,
                    num_beams=1,   # Greedy decoding to stop repeating hallucinations
                    length_penalty=1.0,
                )
            piece = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predicted_text = merge_strings(predicted_text, piece)
            
        return predicted_text.strip()
    
    # Otherwise, it is a normal short word or character
    if preprocess:
        from utils.preprocess import preprocess_image
        # preprocess_image already reads from path
        chunk_cv = preprocess_image(image_path)
        full_image = Image.fromarray(cv2.cvtColor(chunk_cv, cv2.COLOR_BGR2RGB))
        
    pixel_values = processor(full_image, return_tensors="pt").pixel_values.to(device)

    # Generate prediction with simpler greedy parameters
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=256,
            num_beams=1,
            length_penalty=1.0,
        )

    # Decode tokens → text
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_text


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
