# Brahmi Script OCR — Final Year Project

## Overview

This project performs **Optical Character Recognition (OCR)** on ancient **Brahmi script** found on stone inscriptions, manuscripts, and historical documents. It uses a **TrOCR** (Transformer-based OCR) model fine-tuned on a curated, map-driven Brahmi dataset.

## Pipeline

```
Image → Preprocessing → OCR (Brahmi Unicode) → Transliteration (Devanagari) → Translation (Hindi / English)
```

| Stage | Status |
|-------|--------|
| Map-driven dataset loading | ✅ Implemented |
| Image preprocessing | ✅ Implemented |
| TrOCR fine-tuning | ✅ Implemented |
| Inference | ✅ Implemented |
| Transliteration | 🔜 Planned |
| Translation | 🔜 Planned |
| Web app | 🔜 Planned |

## Project Structure

```
brahmi_ocr_project/
│
├── dataset/
│   ├── map.json                  single source of truth for labels/folders
│   ├── 1Vowels/                  character folders
│   ├── 2Consonants/              character + matra folders
│   ├── 3Numbers/                 number folders
│   ├── 4Extras/                  punctuation/symbol folders
│   └── 5Words_Phrases/           mixed text images (+ labels file)
│
├── model/
│   └── brahmi_trocr/           ← saved fine-tuned model
│
├── training/
│   ├── train.py                ← TrOCR fine-tuning script
│   └── dataset_loader.py       ← PyTorch Dataset class
│
├── inference/
│   └── predict.py              ← run OCR on an image
│
├── utils/
│   └── preprocess.py           ← image preprocessing utilities
│
├── webapp/                     ← future frontend
├── Capstone_Brahmi_Inscriptions/   reference repo
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Set up the environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

### 2. Prepare the dataset

```bash
# Required: map.json + image folders
dataset/
  map.json
  1Vowels/
  2Consonants/
  3Numbers/
  4Extras/
  5Words_Phrases/
```

`map.json` entries support:
- Fixed label folder: every image in folder gets the mapped `char`
- Mixed folder (`char = "MIXED"`): labels come from `labels.txt` / `labels.tsv` / `labels.csv` / `annotations.*`

### 3. Train the model

```bash
python training/train.py
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `microsoft/trocr-small-printed` | Base model |
| `--epochs` | 10 | Training epochs |
| `--batch_size` | 2 | Batch size |
| `--lr` | 5e-5 | Learning rate |
| `--data_dir` | `dataset/` | Dataset directory |
| `--output_dir` | `model/brahmi_trocr/` | Model save directory |
| `--train_ratio` | 0.8 | Train split ratio |
| `--val_ratio` | 0.1 | Validation split ratio |
| `--test_ratio` | 0.1 | Test split ratio |
| `--image_size` | 384 | Square padded image size |

The script automatically uses **FP16** if a CUDA GPU is available.
It also prints dataset breakdown (characters/ngrams, words, phrases, long sentences).

### 4. Run inference

```bash
python inference/predict.py --image path/to/brahmi_image.png

# With preprocessing (recommended for real stone/manuscript images)
python inference/predict.py --image path/to/image.png --preprocess

# Full debug JSON for backend/UI
python inference/predict.py --image path/to/image.png --preprocess --debug --json_out result.json
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| OCR Model | TrOCR (`microsoft/trocr-small-printed`) |
| Training | PyTorch + HuggingFace Transformers |
| Dataset | Curated map.json-driven hierarchy |
| Image Processing | OpenCV, Pillow |
| Backend (future) | FastAPI |
| Frontend (future) | React |

## Dataset Format

- **`dataset/map.json`** — canonical label/folder mapping (single source of truth)
- **Fixed class folders** — labels taken directly from mapped `char`
- **Mixed text folder(s)** — labels read from `labels.*` / `annotations.*` manifest
- **Train/Val/Test** — generated at runtime by `training/dataset_loader.py` using configured split ratios

## Reference

The `Capstone_Brahmi_Inscriptions/` directory contains a reference project
with preprocessing, segmentation, and a CNN-based OCR approach. We reuse
insights from its image preprocessing pipeline but use a modern
Transformer architecture (TrOCR) instead.

## License

This project is part of an academic final-year submission.
