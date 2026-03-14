"""
training/train.py
=================

Fine-tune TrOCR on map.json-driven Brahmi OCR datasets.
Supports Pass 1 efficient training with balanced classes, 
gradient accumulation, length collapse diagnostics, and sequence category tracking.
"""

from __future__ import annotations

import argparse
import contextlib
import math
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

from training.dataset_loader import BrahmiDataset, create_weighted_sampler, summarize_samples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"))
    p.add_argument("--extra_data", type=str, nargs="*", default=[])
    p.add_argument("--model_name", type=str, default="microsoft/trocr-small-printed")
    p.add_argument("--output_dir", type=str, default="model/brahmi_trocr")
    p.add_argument("--drive_save_path", type=str, default="")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_label_length", type=int, default=64)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=min(os.cpu_count() or 2, 4))
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--balanced_sampling", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_fixed_per_class", type=int, default=80)
    p.add_argument("--max_words", type=int, default=10000)
    p.add_argument("--max_phrases", type=int, default=10000)
    p.add_argument("--max_long_sentences", type=int, default=8000)
    return p.parse_args()


def _sum_dict_counts(dicts: Iterable[Dict[int, int]]) -> Dict[int, int]:
    merged: Dict[int, int] = {}
    for d in dicts:
        for k, v in d.items():
            merged[int(k)] = merged.get(int(k), 0) + int(v)
    return dict(sorted(merged.items()))


def merge_summaries(summaries: List[Dict]) -> Dict:
    category_counts = {"characters_ngrams": 0, "words": 0, "phrases": 0, "long_sentences": 0}
    total_samples, unique_labels = 0, 0
    char_hists, word_hists = [], []

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

    pct = lambda x: 100.0 * x / total
    print(f"\n{title}")
    print(f"  Total samples         : {summary.get('total_samples', 0)}")
    print(f"  Characters/N-Grams    : {c_chars} ({pct(c_chars):.2f}%)")
    print(f"  Words                 : {c_words} ({pct(c_words):.2f}%)")
    print(f"  Phrases               : {c_phrases} ({pct(c_phrases):.2f}%)")
    print(f"  Long sentences        : {c_long} ({pct(c_long):.2f}%)")


def _looks_like_local_model_path(value: str) -> bool:
    if not value:
        return False

    expanded = os.path.expanduser(value)
    if os.path.exists(expanded) or os.path.isabs(expanded):
        return True

    if expanded.startswith((".", "~", os.path.sep)):
        return True

    drive, _ = os.path.splitdrive(expanded)
    return bool(drive)


def load_model(model_name: str, processor, dataset_chars: List[str], max_new_tokens: int):
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    existing_vocab = processor.tokenizer.get_vocab()
    new_tokens = [AddedToken(ch, normalized=False) for ch in dataset_chars if ch and ch not in existing_vocab]

    if new_tokens:
        num_added = processor.tokenizer.add_tokens(new_tokens)
        print(f"  Added {num_added} tokens (vocab size -> {len(processor.tokenizer)})")

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


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler, grad_accum_steps):
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda") if scaler else contextlib.nullcontext():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / grad_accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(dataloader):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device, processor):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    predictions, references, category_ids_list = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            cids = batch.get("category_ids", torch.zeros(len(labels)))
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
            
            generated_ids = model.generate(pixel_values)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            labels[labels == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(preds)
            references.extend(refs)
            category_ids_list.extend(cids.tolist())

    val_loss = total_loss / max(num_batches, 1)
    
    metrics = {
        0: {"preds": [], "refs": []},
        1: {"preds": [], "refs": []},
        2: {"preds": [], "refs": []},
        3: {"preds": [], "refs": []},
    }
    
    exact_match = 0
    first_char_acc = 0
    len_ratio_sum = 0
    total_valid = 0

    filtered_preds, filtered_refs = [], []

    for p, r, cid in zip(predictions, references, category_ids_list):
        if r.strip():
            ps, rs = p.strip(), r.strip()
            filtered_preds.append(ps)
            filtered_refs.append(rs)
            metrics[cid]["preds"].append(ps)
            metrics[cid]["refs"].append(rs)
            
            total_valid += 1
            if ps == rs: exact_match += 1
            if ps and ps[0] == rs[0]: first_char_acc += 1
            len_ratio_sum += len(ps) / max(len(rs), 1)
            
    cer = jiwer.cer(filtered_refs, filtered_preds) if filtered_refs else 1.0
    wer = jiwer.wer(filtered_refs, filtered_preds) if filtered_refs else 1.0
    
    results = {
        "loss": val_loss,
        "cer": cer,
        "wer": wer,
        "exact_match": exact_match / max(total_valid, 1),
        "first_char_acc": first_char_acc / max(total_valid, 1),
        "len_ratio": len_ratio_sum / max(total_valid, 1),
        "cat_metrics": {}
    }
    
    names = {0: "char", 1: "word", 2: "phrase", 3: "long"}
    for cid, data in metrics.items():
        if data["refs"]:
            results["cat_metrics"][names[cid]] = {
                "cer": jiwer.cer(data["refs"], data["preds"]),
                "wer": jiwer.wer(data["refs"], data["preds"])
            }
            
    return results


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

    model_name_or_path = args.model_name

    # Check for internal checkpoints (Output dir takes precedence for easy resume)
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        model_name_or_path = os.path.abspath(args.output_dir)
        print(f"🔄 Resuming from local output checkpoint: {model_name_or_path}")
    elif args.drive_save_path and os.path.exists(os.path.join(args.drive_save_path, "config.json")):
        model_name_or_path = os.path.abspath(args.drive_save_path)
        print(f"🔄 Resuming from Google Drive checkpoint: {model_name_or_path}")

    if _looks_like_local_model_path(model_name_or_path):
        model_name_or_path = os.path.abspath(os.path.expanduser(model_name_or_path))
        if not os.path.isdir(model_name_or_path):
            print(f"❌ Error: Model directory not found at {model_name_or_path}")
            sys.exit(1)
        print(f"🔍 Loading local model from: {model_name_or_path}")
    else:
        print(f"🌐 Loading remote model from Hub: {model_name_or_path}")

    processor = TrOCRProcessor.from_pretrained(model_name_or_path, use_fast=False)

    train_parts, val_parts = [], []
    train_summaries, val_summaries = [], []
    dataset_chars = set()

    for root in roots:
        train_ds = BrahmiDataset(root, "train", processor, args.max_label_length, "map.json", split_ratios, args.seed, args.image_size, args.max_fixed_per_class, args.max_words, args.max_phrases, args.max_long_sentences)
        val_ds = BrahmiDataset(root, "val", processor, args.max_label_length, "map.json", split_ratios, args.seed, args.image_size, args.max_fixed_per_class, args.max_words, args.max_phrases, args.max_long_sentences)

        if args.max_train_samples > 0:
            train_ds.samples = train_ds.samples[:args.max_train_samples]
            val_ds.samples = val_ds.samples[:args.max_train_samples]
            train_ds.summary = summarize_samples(train_ds.samples)
            val_ds.summary = summarize_samples(val_ds.samples)

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

    dataset_chars.add(" ")
    sorted_chars = sorted(ch for ch in dataset_chars if ch)

    model = load_model(model_name_or_path, processor, sorted_chars, args.max_label_length)
    model.to(device)

    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)

    sampler = create_weighted_sampler(train_ds) if args.balanced_sampling else None
    shuffle = sampler is None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, 
        collate_fn=BrahmiDataset.collate_fn, num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        collate_fn=BrahmiDataset.collate_fn, num_workers=args.num_workers, pin_memory=args.pin_memory
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1)))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_val_cer = float("inf")
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, args.gradient_accumulation_steps)
        metrics = evaluate(model, val_loader, device, processor)
        val_cer = metrics["cer"]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f}\n"
            f"  [Global] CER: {val_cer:.4f} | WER: {metrics['wer']:.4f}\n"
            f"  [Diagnostic] Exact: {metrics['exact_match']:.2f} | First Char: {metrics['first_char_acc']:.2f} | Len Ratio: {metrics['len_ratio']:.2f}"
        )
        for cat, scores in metrics["cat_metrics"].items():
            print(f"    - {cat}: CER {scores['cer']:.4f} | WER {scores['wer']:.4f}")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"  -> Best model saved to {args.output_dir}")
            
            if args.drive_save_path:
                try:
                    import shutil
                    os.makedirs(os.path.dirname(args.drive_save_path), exist_ok=True)
                    if os.path.exists(args.drive_save_path):
                        shutil.rmtree(args.drive_save_path)
                    shutil.copytree(args.output_dir, args.drive_save_path)
                    print(f"  -> 💾 Auto-synced backup to Google Drive!")
                except Exception as e:
                    print(f"  -> ⚠️ Could not sync to Drive: {e}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

if __name__ == "__main__":
    main()
