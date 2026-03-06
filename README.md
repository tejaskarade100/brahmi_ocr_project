# Brahmi Script OCR — Final Year Project

## Overview

This project performs **Optical Character Recognition (OCR)** on ancient **Brahmi script** found on stone inscriptions, manuscripts, and historical documents. It uses a **TrOCR** (Transformer-based OCR) model fine-tuned on a synthetically generated Brahmi dataset.

## Pipeline

```
Image → Preprocessing → OCR (Brahmi Unicode) → Transliteration (Devanagari) → Translation (Hindi / English)
```

| Stage | Status |
|-------|--------|
| Synthetic dataset generation | ✅ Implemented |
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
│   ├── generate_synthetic.py  ← synthetic dataset generator
│   ├── images/                   generated Brahmi text images
│   ├── labels.txt                image_name<TAB>brahmi_text
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
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
├── NotoSansBrahmi-Regular.ttf      Brahmi font for dataset generation
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

### 2. Generate the synthetic dataset

```bash
# Full dataset (~20,000 images)
python dataset/generate_synthetic.py

# Small test run (25 images)
python dataset/generate_synthetic.py --num_chars 10 --num_words 10 --num_phrases 5
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--num_chars` | 5000 | Number of single-character images |
| `--num_words` | 10000 | Number of word images (2–6 chars) |
| `--num_phrases` | 5000 | Number of phrase images (5–15 chars) |
| `--img_size` | 384 | Image width and height in pixels |
| `--font_size` | 64 | Base font size |
| `--seed` | 42 | Random seed for reproducibility |

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

The script automatically uses **FP16** if a CUDA GPU is available.

### 4. Run inference

```bash
python inference/predict.py --image path/to/brahmi_image.png

# With preprocessing (recommended for real stone/manuscript images)
python inference/predict.py --image path/to/image.png --preprocess
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| OCR Model | TrOCR (`microsoft/trocr-small-printed`) |
| Training | PyTorch + HuggingFace Transformers |
| Dataset | Synthetic (NotoSansBrahmi font) |
| Image Processing | OpenCV, Pillow |
| Backend (future) | FastAPI |
| Frontend (future) | React |

## Dataset Format

- **`dataset/images/`** — PNG images of Brahmi text
- **`dataset/labels.txt`** — Tab-separated: `img_000001.png\t𑀓𑀭`
- **`dataset/splits/`** — Train (80%) / Val (10%) / Test (10%)

## Reference

The `Capstone_Brahmi_Inscriptions/` directory contains a reference project
with preprocessing, segmentation, and a CNN-based OCR approach. We reuse
insights from its image preprocessing pipeline but use a modern
Transformer architecture (TrOCR) instead.

## License

This project is part of an academic final-year submission.