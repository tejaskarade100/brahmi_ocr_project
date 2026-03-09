"""
training/train.py
=================

Fine-tune TrOCR on map.json-driven Brahmi OCR datasets.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, Iterable, List

import torch
import jiwer
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AddedToken, TrOCRProcessor, VisionEncoderDecoderModel, get_cosine_schedule_with_warmup

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset_loader import BrahmiDataset


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TrOCR on Brahmi OCR data (map.json)")
    p.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"),
        help="Path to primary dataset root (must contain map.json)",
    )
    p.add_argument(
        "--extra_data",
        type=str,
        nargs="*",
        default=[],
        help="Optional extra dataset roots, each with its own map.json",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="microsoft/trocr-small-printed",
        help="HuggingFace model identifier or local checkpoint path",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="model/brahmi_trocr",
        help="Directory to save the best model",
    )
    p.add_argument(
        "--drive_save_path",
        type=str,
        default="",
        help="Optional backup path for best model after each improvement",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_label_length", type=int, default=64)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    return p.parse_args()


def _sum_dict_counts(dicts: Iterable[Dict[int, int]]) -> Dict[int, int]:
    merged: Dict[int, int] = {}
    for d in dicts:
        for k, v in d.items():
            merged[int(k)] = merged.get(int(k), 0) + int(v)
    return dict(sorted(merged.items()))


def merge_summaries(summaries: List[Dict]) -> Dict:
    if not summaries:
        return {
            "total_samples": 0,
            "unique_labels": 0,
            "category_counts": {
                "characters_ngrams": 0,
                "words": 0,
                "phrases": 0,
                "long_sentences": 0,
            },
            "char_length_histogram": {},
            "word_count_histogram": {},
        }

    category_counts = {
        "characters_ngrams": 0,
        "words": 0,
        "phrases": 0,
        "long_sentences": 0,
    }
    total_samples = 0
    unique_labels = 0
    char_hists = []
    word_hists = []

    for s in summaries:
        total_samples += int(s.get("total_samples", 0))
        unique_labels += int(s.get("unique_labels", 0))
        cats = s.get("category_counts", {})
        for key in category_counts:
            category_counts[key] += int(cats.get(key, 0))
        char_hists.append(s.get("char_length_histogram", {}))
        word_hists.append(s.get("word_count_histogram", {}))

    return {
        "total_samples": total_samples,
        "unique_labels": unique_labels,
        "category_counts": category_counts,
        "char_length_histogram": _sum_dict_counts(char_hists),
        "word_count_histogram": _sum_dict_counts(word_hists),
    }


def print_dataset_summary(title: str, summary: Dict):
    total = max(int(summary.get("total_samples", 0)), 1)
    cats = summary.get("category_counts", {})

    c_chars = int(cats.get("characters_ngrams", 0))
    c_words = int(cats.get("words", 0))
    c_phrases = int(cats.get("phrases", 0))
    c_long = int(cats.get("long_sentences", 0))

    def pct(x: int) -> float:
        return 100.0 * x / total

    print(f"\n{title}")
    print(f"  Total samples         : {summary.get('total_samples', 0)}")
    print(f"  Unique text labels    : {summary.get('unique_labels', 0)}")
    print(f"  Characters/N-Grams    : {c_chars} ({pct(c_chars):.2f}%)")
    print(f"  Words                 : {c_words} ({pct(c_words):.2f}%)")
    print(f"  Phrases               : {c_phrases} ({pct(c_phrases):.2f}%)")
    print(f"  Long sentences        : {c_long} ({pct(c_long):.2f}%)")

    char_hist = summary.get("char_length_histogram", {})
    word_hist = summary.get("word_count_histogram", {})
    if char_hist:
        top_char_lengths = list(char_hist.items())[:12]
        print(f"  Char-length histogram : {top_char_lengths}")
    if word_hist:
        top_word_lengths = list(word_hist.items())[:12]
        print(f"  Word-count histogram  : {top_word_lengths}")


def load_model(model_name: str, processor, dataset_chars: List[str], max_new_tokens: int):
    """
    Load model, add dataset characters to tokenizer, and set generation config.
    """
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    existing_vocab = processor.tokenizer.get_vocab()
    new_tokens = [
        AddedToken(ch, normalized=False)
        for ch in dataset_chars
        if ch and ch not in existing_vocab
    ]

    if new_tokens:
        num_added = processor.tokenizer.add_tokens(new_tokens)
        print(
            f"  Added {num_added} dataset tokens to tokenizer "
            f"(vocab size -> {len(processor.tokenizer)})"
        )

    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.decoder.vocab_size = len(processor.tokenizer)
    model.config.vocab_size = len(processor.tokenizer)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    return model


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device, processor):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
            
            generated_ids = model.generate(pixel_values)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            labels[labels == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(preds)
            references.extend(refs)

    val_loss = total_loss / max(num_batches, 1)
    
    # Filter out empty references to avoid jiwer errors
    filtered_preds = []
    filtered_refs = []
    for p, r in zip(predictions, references):
        if r.strip():
            filtered_preds.append(p)
            filtered_refs.append(r)
            
    cer = jiwer.cer(filtered_refs, filtered_preds) if filtered_refs else 1.0
    wer = jiwer.wer(filtered_refs, filtered_preds) if filtered_refs else 1.0
    
    return val_loss, cer, wer


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    print(f"Device: {device} | FP16: {use_fp16}")

    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    roots = [args.data_dir] + list(args.extra_data or [])
    roots = [r for r in roots if r and os.path.isdir(r)]
    if not roots:
        raise ValueError("No valid dataset roots found. Check --data_dir / --extra_data.")

    # Auto-resume: prefer existing output checkpoint, then optional backup path.
    model_name_or_path = args.model_name
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"\n[RESUME] Found existing model at '{args.output_dir}'.")
        model_name_or_path = args.output_dir
    elif args.drive_save_path and os.path.exists(os.path.join(args.drive_save_path, "config.json")):
        print(f"\n[RESUME] Found existing model at '{args.drive_save_path}'.")
        model_name_or_path = args.drive_save_path

    print(f"Loading processor from: {model_name_or_path}")
    processor = TrOCRProcessor.from_pretrained(model_name_or_path, use_fast=False)

    train_parts = []
    val_parts = []
    train_summaries = []
    val_summaries = []
    dataset_chars = set()

    for root in roots:
        print(f"\nLoading dataset root: {root}")
        train_ds = BrahmiDataset(
            root,
            split="train",
            processor=processor,
            max_label_length=args.max_label_length,
            split_ratios=split_ratios,
            seed=args.seed,
            image_size=args.image_size,
        )
        val_ds = BrahmiDataset(
            root,
            split="val",
            processor=processor,
            max_label_length=args.max_label_length,
            split_ratios=split_ratios,
            seed=args.seed,
            image_size=args.image_size,
        )

        train_parts.append(train_ds)
        val_parts.append(val_ds)
        train_summaries.append(train_ds.summary)
        val_summaries.append(val_ds.summary)
        dataset_chars.update(train_ds.character_set)
        dataset_chars.update(val_ds.character_set)

    train_summary = merge_summaries(train_summaries)
    val_summary = merge_summaries(val_summaries)
    print_dataset_summary("Train Dataset Breakdown", train_summary)
    print_dataset_summary("Val Dataset Breakdown", val_summary)

    # Ensure whitespace token is included for phrase/sentence OCR.
    dataset_chars.add(" ")
    sorted_chars = sorted(ch for ch in dataset_chars if ch)
    print(f"\nTokenizer character inventory size: {len(sorted_chars)}")

    print(f"Loading model from: {model_name_or_path}")
    model = load_model(
        model_name_or_path,
        processor=processor,
        dataset_chars=sorted_chars,
        max_new_tokens=args.max_label_length,
    )
    model.to(device)

    if len(train_parts) == 1:
        train_ds = train_parts[0]
        val_ds = val_parts[0]
    else:
        train_ds = ConcatDataset(train_parts)
        val_ds = ConcatDataset(val_parts)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=BrahmiDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=BrahmiDataset.collate_fn,
    )

    print(f"\nTrain samples: {len(train_ds)}")
    print(f"Val samples  : {len(val_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_cer = float("inf")
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_loss, val_cer, val_wer = evaluate(model, val_loader, device, processor)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f}"
        )

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"  -> Best model saved to {args.output_dir} (CER: {val_cer:.4f})")

            if args.drive_save_path:
                try:
                    import shutil

                    if os.path.exists(args.drive_save_path):
                        shutil.rmtree(args.drive_save_path)
                    shutil.copytree(args.output_dir, args.drive_save_path)
                    print(f"  -> Backup saved to {args.drive_save_path}")
                except Exception as exc:
                    print(f"  -> Warning: backup failed: {exc}")
        else:
            patience_counter += 1
            print(f"  -> No improvement in CER. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\\nEarly stopping triggered at epoch {epoch}.")
                break

    print(f"\\nTraining complete. Best val CER: {best_val_cer:.4f}")
    print(f"Model saved at: {args.output_dir}")


if __name__ == "__main__":
    main()
