"""
inference/predict.py
====================

Run OCR prediction on a Brahmi image and return either:
1) plain text output, or
2) detailed structured output for backend/UI use.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import unicodedata
from typing import Dict, List, Tuple

import cv2
import numpy as np
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
    line_count = len([line for line in text.splitlines() if line.strip()])
    words = [w for w in text.split() if w]
    word_count = len(words)
    char_count = len(text.replace(" ", "").replace("\n", ""))
    if line_count > 1:
        return "multiline"
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
    words = [w for w in text.split() if w]
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "character_count": len(text.replace(" ", "").replace("\n", "")),
        "word_count": len(words),
        "line_count": len(lines) if lines else 1,
        "line_break_count": text.count("\n"),
        "space_count": text.count(" "),
        "word_char_counts": [len(w) for w in words],
        "category_guess": _category_guess(text),
    }


def _segment_lines(image: Image.Image, min_line_height: int = 10, min_gap: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    Given a PIL Image, converts to grayscale, applies a binary threshold,
    and calculates horizontal projection profiles to segment text lines.
    Returns a list of bounding boxes: (x_min, y_min, x_max, y_max)
    """
    cv_img = np.array(image.convert("L"))
    _, thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_activity = np.count_nonzero(thresh, axis=1)
    active_threshold = max(2, int(thresh.shape[1] * 0.01))

    lines = []
    in_line = False
    start_y = 0

    for y, val in enumerate(row_activity):
        if not in_line and val >= active_threshold:
            in_line = True
            start_y = y
        elif in_line and val < active_threshold:
            in_line = False
            if y - start_y >= min_line_height:
                pad_y = max(0, start_y - min_gap)
                pad_bot = min(cv_img.shape[0], y + min_gap)

                line_slice = thresh[pad_y:pad_bot, :]
                col_activity = np.count_nonzero(line_slice, axis=0)
                nz = np.nonzero(col_activity)[0]
                if len(nz) > 0:
                    start_x = max(0, nz[0] - min_gap)
                    end_x = min(cv_img.shape[1], nz[-1] + min_gap + 1)
                    lines.append((int(start_x), int(pad_y), int(end_x), int(pad_bot)))
                else:
                    lines.append((0, int(pad_y), int(cv_img.shape[1]), int(pad_bot)))

    if in_line and len(row_activity) - start_y >= min_line_height:
        pad_y = max(0, start_y - min_gap)
        line_slice = thresh[pad_y:, :]
        col_activity = np.count_nonzero(line_slice, axis=0)
        nz = np.nonzero(col_activity)[0]
        if len(nz) > 0:
            start_x = max(0, nz[0] - min_gap)
            end_x = min(cv_img.shape[1], nz[-1] + min_gap + 1)
            lines.append((int(start_x), int(pad_y), int(end_x), int(cv_img.shape[0])))
        else:
            lines.append((0, int(pad_y), int(cv_img.shape[1]), int(cv_img.shape[0])))

    if not lines:
        return [(0, 0, image.width, image.height)]

    return lines


def _pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def predict(
    image_path: str,
    processor,
    model,
    device: str,
    preprocess: bool = False,
    image_size: int = 384,
    threshold_method: str | None = None,
    debug: bool = False,
    multiline: bool = False,
    return_base64: bool = False,
) -> Dict:
    preprocess_info = {}
    original_image = Image.open(image_path).convert("RGB")

    if preprocess:
        if debug:
            base_image, preprocess_info = preprocess_image(
                image_path,
                target_size=None,
                threshold_method=threshold_method,
                return_debug=True,
            )
        else:
            base_image = preprocess_image(
                image_path,
                target_size=None,
                threshold_method=threshold_method,
                return_debug=False,
            )
            preprocess_info = {
                "pipeline": "preprocess_image",
                "debug": False,
                "target_size": None,
            }
    else:
        base_image = original_image
        preprocess_info = {"pipeline": "none", "debug": False}

    if multiline:
        line_boxes = _segment_lines(base_image)
    else:
        line_boxes = [(0, 0, base_image.width, base_image.height)]

    all_line_results = []
    full_text = []
    token_trace_out = []
    character_trace_out = []

    for line_index, box in enumerate(line_boxes):
        line_crop = base_image.crop(box)

        if debug:
            model_input_img, pad_meta = letterbox_pil(
                line_crop, target_size=(image_size, image_size), return_meta=True
            )
            preprocess_info.setdefault("line_padding", []).append(
                {"line_index": line_index, **pad_meta}
            )
        else:
            model_input_img = letterbox_pil(line_crop, target_size=(image_size, image_size))

        pixel_values = processor(model_input_img, return_tensors="pt").pixel_values.to(device)

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
        line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        full_text.append(line_text)

        line_result = {
            "line_index": line_index,
            "bbox": {"x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3]},
            "text": line_text,
            "text_breakdown": _text_breakdown(line_text),
        }

        if debug:
            transition_scores = None
            if hasattr(gen_out, "scores") and gen_out.scores:
                beam_indices = getattr(gen_out, "beam_indices", None)
                try:
                    transition_scores = model.compute_transition_scores(
                        generated_ids,
                        gen_out.scores,
                        beam_indices=beam_indices,
                        normalize_logits=True,
                    )
                except TypeError:
                    transition_scores = model.compute_transition_scores(
                        generated_ids, gen_out.scores, beam_indices=beam_indices
                    )

            tokenizer = processor.tokenizer
            line_token_trace = _token_trace(generated_ids, transition_scores, tokenizer)
            line_character_trace = _character_trace(line_text)
            line_result["token_trace"] = line_token_trace
            line_result["character_trace"] = line_character_trace

            if len(line_boxes) == 1:
                token_trace_out = line_token_trace
                character_trace_out = line_character_trace
            else:
                token_trace_out.append({"line_index": line_index, "tokens": line_token_trace})
                character_trace_out.append({"line_index": line_index, "characters": line_character_trace})

        all_line_results.append(line_result)

    final_predicted_text = "\n".join(full_text) if multiline and len(full_text) > 1 else (full_text[0] if full_text else "")

    result = {
        "image_path": image_path,
        "predicted_text": final_predicted_text,
        "preprocess": preprocess_info,
        "text_breakdown": _text_breakdown(final_predicted_text),
        "lines": all_line_results,
    }

    if debug:
        result["token_trace"] = token_trace_out
        result["character_trace"] = character_trace_out

    if return_base64 and base_image is not None:
        result["base64_image"] = f"data:image/jpeg;base64,{_pil_to_base64(base_image)}"

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
        choices=["adaptive", "otsu", "simple", "auto"],
        help="Optional thresholding method when --preprocess is enabled (use 'auto' for heuristic detection)",
    )
    parser.add_argument("--image_size", type=int, default=384, help="Square target size")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include token/character/preprocess trace for backend/UI",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Attempt to segment the image into multiple lines before inference",
    )
    parser.add_argument(
        "--base64",
        action="store_true",
        help="Embed the base64 preprocessed image directly in the output JSON",
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
        multiline=args.multiline,
        return_base64=args.base64,
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
