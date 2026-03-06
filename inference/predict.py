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

    processor = TrOCRProcessor.from_pretrained(model_dir)
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
    if preprocess:
        from utils.preprocess import preprocess_image
        image = preprocess_image(image_path)
    else:
        image = Image.open(image_path).convert("RGB")

    # Encode the image
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate prediction with explicit parameters
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
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
