"""
inference/predict.py
====================

Run OCR prediction on a Brahmi image and return either:
1) plain text output, or
2) detailed structured output for backend/UI use.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import unicodedata
from typing import Dict, List

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import letterbox_pil, preprocess_image


def load_trained_model(model_dir: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained(model_dir, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    if model.generation_config.decoder_start_token_id is None:
        model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    return processor, model, device


def _category_guess(text: str) -> str:
    words = [w for w in text.split(" ") if w]
    word_count = len(words)
    char_count = len(text.replace(" ", ""))
    if word_count <= 1 and char_count <= 2:
        return "character_or_ngram"
    if word_count <= 1:
        return "word"
    if word_count <= 4:
        return "phrase"
    return "long_sentence"


def _character_trace(text: str) -> List[Dict]:
    out = []
    for idx, ch in enumerate(text):
        cp = f"U+{ord(ch):05X}"
        try:
            uname = unicodedata.name(ch)
        except ValueError:
            uname = "UNKNOWN"
        out.append(
            {
                "index": idx,
                "char": ch,
                "codepoint": cp,
                "unicode_name": uname,
                "is_space": ch == " ",
            }
        )
    return out


def _token_trace(
    generated_ids: torch.Tensor,
    transition_scores: torch.Tensor | None,
    tokenizer,
) -> List[Dict]:
    seq = generated_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(seq)
    trace = []
    for idx, (tok_id, token) in enumerate(zip(seq, tokens)):
        confidence = None
        if transition_scores is not None and idx > 0:
            step_idx = idx - 1
            if step_idx < transition_scores.shape[1]:
                logp = float(transition_scores[0, step_idx].item())
                if math.isfinite(logp):
                    confidence = float(math.exp(logp))

        trace.append(
            {
                "index": idx,
                "token_id": int(tok_id),
                "token": str(token),
                "is_special": bool(tok_id in tokenizer.all_special_ids),
                "confidence": confidence,
            }
        )
    return trace


def _text_breakdown(text: str) -> Dict:
    words = [w for w in text.split(" ") if w]
    return {
        "character_count": len(text.replace(" ", "")),
        "word_count": len(words),
        "space_count": text.count(" "),
        "word_char_counts": [len(w) for w in words],
        "category_guess": _category_guess(text),
    }


def predict(
    image_path: str,
    processor,
    model,
    device: str,
    preprocess: bool = False,
    image_size: int = 384,
    threshold_method: str | None = None,
    debug: bool = False,
) -> Dict:
    preprocess_info = {}

    if preprocess:
        if debug:
            image, preprocess_info = preprocess_image(
                image_path,
                target_size=(image_size, image_size),
                threshold_method=threshold_method,
                return_debug=True,
            )
        else:
            image = preprocess_image(
                image_path,
                target_size=(image_size, image_size),
                threshold_method=threshold_method,
                return_debug=False,
            )
            preprocess_info = {"pipeline": "preprocess_image", "debug": False}
    else:
        image = Image.open(image_path).convert("RGB")
        if debug:
            image, pad_meta = letterbox_pil(
                image, target_size=(image_size, image_size), return_meta=True
            )
            preprocess_info = {
                "pipeline": "letterbox_only",
                "padding": pad_meta,
                "debug": True,
            }
        else:
            image = letterbox_pil(image, target_size=(image_size, image_size))
            preprocess_info = {"pipeline": "letterbox_only", "debug": False}

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        gen_out = model.generate(
            pixel_values,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            return_dict_in_generate=True,
            output_scores=debug,
        )

    generated_ids = gen_out.sequences
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    transition_scores = None
    if debug and hasattr(gen_out, "scores") and gen_out.scores:
        beam_indices = getattr(gen_out, "beam_indices", None)
        try:
            transition_scores = model.compute_transition_scores(
                generated_ids,
                gen_out.scores,
                beam_indices=beam_indices,
                normalize_logits=True,
            )
        except TypeError:
            # Compatibility fallback for older transformers versions.
            transition_scores = model.compute_transition_scores(
                generated_ids, gen_out.scores, beam_indices=beam_indices
            )

    result = {
        "image_path": image_path,
        "predicted_text": predicted_text,
        "preprocess": preprocess_info,
        "text_breakdown": _text_breakdown(predicted_text),
    }

    if debug:
        tokenizer = processor.tokenizer
        result["token_trace"] = _token_trace(generated_ids, transition_scores, tokenizer)
        result["character_trace"] = _character_trace(predicted_text)
        result["generation"] = {
            "sequence_length": int(generated_ids.shape[1]),
            "decoder_start_token_id": int(model.generation_config.decoder_start_token_id),
            "eos_token_id": int(model.generation_config.eos_token_id),
            "pad_token_id": int(model.generation_config.pad_token_id),
            "eos_emitted": int(model.generation_config.eos_token_id)
            in [int(x) for x in generated_ids[0].tolist()],
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Brahmi OCR inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "brahmi_trocr"
        ),
        help="Path to trained model directory",
    )
    parser.add_argument("--preprocess", action="store_true", help="Run preprocess pipeline")
    parser.add_argument(
        "--threshold_method",
        type=str,
        default=None,
        choices=["adaptive", "otsu", "simple"],
        help="Optional thresholding method when --preprocess is enabled",
    )
    parser.add_argument("--image_size", type=int, default=384, help="Square target size")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include token/character/preprocess trace for backend/UI",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default="",
        help="Optional path to write full prediction JSON",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model_dir}")
    processor, model, device = load_trained_model(args.model_dir)
    print(f"Running OCR on: {args.image}")

    result = predict(
        image_path=args.image,
        processor=processor,
        model=model,
        device=device,
        preprocess=args.preprocess,
        image_size=args.image_size,
        threshold_method=args.threshold_method,
        debug=args.debug,
    )

    print("\n==================================================")
    print("Predicted Brahmi text:")
    print(f"  {result['predicted_text']}")
    print("==================================================")

    if args.debug:
        stats = result.get("text_breakdown", {})
        print("\nBreakdown:")
        print(f"  Category guess : {stats.get('category_guess')}")
        print(f"  Characters     : {stats.get('character_count')}")
        print(f"  Words          : {stats.get('word_count')}")
        print(f"  Spaces         : {stats.get('space_count')}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed JSON written to: {args.json_out}")
    elif args.debug:
        print("\nDetailed JSON:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
