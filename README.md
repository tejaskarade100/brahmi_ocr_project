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


### 5. Training on Free GPU Cloud Platforms (Colab / Kaggle)

Because training TrOCR requires significant compute, you can use the provided Jupyter Notebooks to seamlessly train on free GPUs without losing your progress!

#### 🚀 Option A: Google Colab (`colab_training.ipynb`)
1. Upload your dataset ZIP to Google Drive.
2. Open `colab_training.ipynb` in Google Colab.
3. Turn on the **T4 GPU** (Runtime -> Change runtime type).
4. Run the notebook. It will automatically mount your Google Drive, unpack the dataset, train the model, and safely save the weights back into your Google Drive permanently!

#### 🚀 Option B: Kaggle (`kaggle_training.ipynb`)
Kaggle offers **30 hours per week of free T4x2 GPU time**.
1. Create a New Notebook in Kaggle.
2. Click **Add Data** -> Upload your `dataset.zip` (Kaggle stores this permanently off-disk).
3. Upload `kaggle_training.ipynb` to the notebook via File -> Import.
4. Turn on the **GPU T4x2** in the right-side Session menu.
5. Run the notebook. 
6. **To Save Your Model:** When training finishes, click the **Save Version** button in the top right corner and choose "Save & Run All (Commit)". Your trained model weights will be permanently saved in the notebook's `checkpoints` output directory for future use!

**Persistence:** Both platforms rely on `train.py`'s automatic checkpoint detection. If your notebook crashes or you switch accounts, just ensure your previous checkpoint folder is available and the script will resume exactly where it left off!

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
