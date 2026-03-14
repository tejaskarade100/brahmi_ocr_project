# Brahmi OCR Project

This repository is an end-to-end OCR system for Brahmi inscriptions. It takes an input image, cleans it, runs OCR with a fine-tuned TrOCR model, transliterates the Brahmi output into Devanagari and Latin, and then produces Hindi and English translations through a FastAPI backend and a React frontend.

The project is organized as a full pipeline rather than only a model checkpoint. It includes:

- a map-driven Brahmi dataset layout
- synthetic data generation for clean, manuscript-like, and stone-like images
- TrOCR fine-tuning code
- inference with preprocessing, line segmentation, and debug traces
- a backend API for OCR + transliteration + translation
- a frontend UI for upload, overlays, diagnostics, and output display

## What The Project Does

At a high level the flow is:

`image -> preprocessing -> OCR -> transliteration -> translation -> UI diagnostics`

In practical terms:

1. A user uploads an inscription image.
2. The backend preprocesses the image for OCR.
3. The OCR model predicts Brahmi Unicode text.
4. The Brahmi text is transliterated into Devanagari and Latin.
5. The Devanagari output is translated into Hindi and English.
6. The frontend shows the original image, processed image, detected lines, OCR text, token trace, character trace, and translations.

## Full End-To-End Pipeline

### 1. Dataset definition

The dataset is controlled by [`dataset/map.json`](dataset/map.json). This is the single source of truth for:

- the Brahmi character inventory
- which folder corresponds to which label
- which folders are fixed-label folders
- which folder is the mixed-text folder

The mapped class inventory in this repository is:

| Category | Count | Share of mapped entries |
| --- | ---: | ---: |
| Vowels | 14 | 2.35% |
| Consonant and matra combinations | 518 | 86.77% |
| Numbers | 30 | 5.03% |
| Extras and signs | 34 | 5.70% |
| Mixed text folder | 1 | 0.17% |
| Total mapped entries | 597 | 100% |

This means the project is not only recognizing isolated base letters. Most of the class space is made of consonant-plus-vowel-sign combinations, which is important for Brahmi OCR because historical text is not just independent characters.

### 2. Dataset strategy

The project is set up for a hybrid dataset strategy:

- folder-based fixed class definitions for characters and symbols
- generated synthetic samples to fill quotas and balance coverage
- mixed text samples for words, phrases, sentences, and multiline text

In the current repository snapshot, the packaged image inventory is dominated by generated data, but the folder structure and loaders are built so curated real samples can live alongside generated samples.

Current image inventory on disk under [`dataset/`](dataset/):

| Top-level folder | Images |
| --- | ---: |
| `1Vowels` | 1,400 |
| `2Consonants` | 51,800 |
| `3Numbers` | 3,000 |
| `4Extras` | 3,400 |
| `5Words_Phrases` | 28,000 |
| Total | 87,600 |

This gives the project broad class coverage while keeping a dedicated mixed-text bucket for actual reading behavior.

### 3. Target balancing

[`dataset/build_targets.py`](dataset/build_targets.py) scans the current dataset and writes a deterministic manifest at [`dataset/reports/targets_manifest.csv`](dataset/reports/targets_manifest.csv).

Its job is to answer:

- how many images already exist for each mapped folder
- how many more need to be generated
- how many of those should be clean, manuscript, and stone style

The style split used by the generator is fixed and explicit:

- clean: 40%
- manuscript: 35%
- stone: 25%

Those same percentages show up in the current generated dataset:

| Synthetic style | Images | Share |
| --- | ---: | ---: |
| Clean | 35,040 | 40.00% |
| Manuscript | 30,660 | 35.00% |
| Stone | 21,900 | 25.00% |
| Total synthetic images | 87,600 | 100% |

### 4. Synthetic generation

[`dataset/generate_synthetic.py`](dataset/generate_synthetic.py) renders Brahmi text using `NotoSansBrahmi-Regular.ttf` and applies style-specific perturbations.

The generator supports:

- fixed class generation for characters, signs, and numerals
- mixed-text generation for `word`, `phrase`, `sentence`, and `multiline`
- HarfBuzz shaping when available so dependent vowel signs attach correctly
- fallback rendering when HarfBuzz is unavailable

Style behavior:

- clean: mild rotation and slight perspective variation
- manuscript: stronger warp, blur, noise, erosion, vignette, contrast change
- stone: strongest warp, elastic distortion, dilation or erosion, noise, shadows, blur

Mixed-text generation behavior:

- words are built from random Brahmi syllable tokens
- phrases use 2 to 4 words
- sentences use 5 to 10 words
- multiline uses 2 to 4 lines and 6 to 15 words total

Important detail for vowels:

- the generator uses a separate independent vowel pool and consonant pool
- at the start of a generated word, an independent vowel is chosen with a 15% probability when available
- otherwise generation is intentionally consonant-dominant
- after the first position, the generator prefers consonant syllables with roughly 85% probability

So the project does not enforce one static global vowel percentage inside all generated text. Instead it uses a controlled bias that keeps sequences readable and historically plausible for OCR training.

The current mixed-text metadata in [`dataset/5Words_Phrases/labels.json`](dataset/5Words_Phrases/labels.json) is:

| Mixed sequence type | Images | Share |
| --- | ---: | ---: |
| Word | 10,000 | 35.71% |
| Phrase | 10,000 | 35.71% |
| Sentence | 4,000 | 14.29% |
| Multiline | 4,000 | 14.29% |
| Total mixed images | 28,000 | 100% |

### 5. Validation and post-checking

The dataset tooling includes:

- [`dataset/validate_dataset.py`](dataset/validate_dataset.py) for structure and composition checks
- [`dataset/postcheck.py`](dataset/postcheck.py) for perceptual deduplication and quota audits

This is meant to prevent:

- missing mapped folders
- broken label manifests
- duplicate generated images
- severe imbalance between class buckets

### 6. Training data loading

[`training/dataset_loader.py`](training/dataset_loader.py) converts the folder structure into training samples.

Key design choices:

- `map.json` is always the authority for label resolution
- fixed folders map directly to one label
- mixed folders read labels from `labels.json`, `labels.txt`, `labels.tsv`, `labels.csv`, or `annotations.*`
- runtime caps prevent huge class imbalance
- weighted sampling can rebalance characters, words, phrases, and long sequences

The loader classifies every sample into one of four buckets:

- `characters_ngrams`
- `words`
- `phrases`
- `long_sentences`

This is important because the project is trying to solve both isolated character OCR and longer sequence OCR in the same model.

### 7. Model loading and vocabulary expansion

[`training/train.py`](training/train.py) fine-tunes `microsoft/trocr-small-printed`.

Training steps:

1. Load the TrOCR processor and VisionEncoderDecoder model.
2. Build the Brahmi character set from the dataset.
3. Add any dataset characters not already in the tokenizer.
4. Resize decoder token embeddings to match the expanded tokenizer.
5. Set decoder start, end, and pad tokens.
6. Fine-tune on Brahmi images and text labels.

From the documented training run you provided:

- base model: `microsoft/trocr-small-printed`
- added tokenizer tokens: 109
- model save path: `model/brahmi_trocr`

The warning about missing `encoder.pooler.*` weights is not a problem for this OCR use case. The pooler is not central to sequence generation and those weights are newly initialized when absent.

### 8. Training loop

Training includes:

- AdamW optimizer
- cosine learning-rate schedule with warmup
- optional FP16 on CUDA
- gradient accumulation
- early stopping on validation CER
- per-category metrics

The training script reports:

- train loss
- validation loss
- CER
- WER
- exact-match ratio
- first-character accuracy
- prediction/label length ratio
- category-wise CER and WER for chars, words, phrases, and long sequences

### 9. Saved model

The best checkpoint is saved to [`model/brahmi_trocr`](model/brahmi_trocr). The training script can also resume from this folder automatically if it already contains a model config.

### 10. Inference

[`inference/predict.py`](inference/predict.py) supports both CLI use and backend use.

Inference flow:

1. Load the saved processor and model.
2. Optionally preprocess the image.
3. Optionally segment the image into lines.
4. Letterbox each crop to the target square size without distortion.
5. Decode with beam search.
6. Return plain text or a structured JSON result.

Structured inference output can include:

- predicted Brahmi text
- text statistics
- line bounding boxes
- per-line decoded text
- token trace
- character trace
- preprocessing metadata
- a base64-encoded processed image

### 11. Preprocessing

[`utils/preprocess.py`](utils/preprocess.py) is shared by inference and backend.

The preprocessing pipeline is:

1. load image
2. grayscale conversion
3. denoising
4. CLAHE contrast enhancement
5. optional thresholding
6. resize with aspect-ratio-preserving padding
7. RGB conversion for model input

Supported threshold modes:

- `adaptive`
- `otsu`
- `simple`
- `auto`

The `auto` mode uses a heuristic based on:

- grayscale contrast standard deviation
- Laplacian variance as a blur proxy

This lets the project skip aggressive binarization for already clean scans while still cleaning noisy photo captures.

### 12. Backend orchestration

[`backend/main.py`](backend/main.py) wires OCR, transliteration, and translation together into one API.

The backend:

- loads the trained OCR model at startup
- accepts image uploads through `/api/upload`
- runs OCR with preprocessing, multiline mode, and debug output enabled
- transliterates Brahmi to Devanagari and Latin
- translates the Devanagari output to Hindi and English
- returns everything in one JSON payload

### 13. Frontend visualization

The React app under [`webapp/ancient-insight-main`](webapp/ancient-insight-main) is not only a viewer. It is a diagnostics UI.

It lets a user:

- upload an inscription image
- switch between original and processed image views
- inspect detected line boxes
- click a line to inspect only that line
- read Brahmi, Latin, Devanagari, Hindi, and English outputs
- inspect token-level and character-level traces

## Reported Training Run Summary

The numbers below are best treated as an indicative project run, not a final benchmark claim.

Documented environment from your run:

- device: CUDA
- FP16: enabled
- base model download from Hugging Face Hub
- model initialized from `microsoft/trocr-small-printed`

Documented dataset split from your run:

| Split | Total | Chars/N-grams | Words | Phrases | Long |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 47,488 | 53.01% | 16.82% | 16.75% | 13.43% |
| Val | 5,936 | 51.79% | 16.81% | 17.45% | 13.95% |

Documented best metrics shown in your log by epoch 6:

- train loss: `0.6941`
- val loss: `0.4422`
- global CER: `0.1969`
- global WER: `0.4051`
- exact match: `0.77`
- first character accuracy: `0.96`
- length ratio: `1.03`

If you want to present these as easier-to-read project accuracy proxies in documentation, a safe wording is:

- approximate character-level accuracy proxy: about `80.3%` from `1 - CER`
- exact full-string match on the logged validation set: about `77%`
- first-character correctness on the logged validation set: about `96%`

Do not present `1 - WER` as a strict official word accuracy. WER is an edit-distance metric, not a simple accuracy percentage.

## Project Structure

```text
brahmi_ocr_project/
|-- backend/
|-- dataset/
|-- inference/
|-- model/
|-- training/
|-- utils/
|-- webapp/ancient-insight-main/
|-- brahmi.json
|-- NotoSansBrahmi-Regular.ttf
|-- colab_training.ipynb
|-- kaggle_training.ipynb
|-- requirements.txt
|-- README.md
```

## How To Run The Project

### Python environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd webapp/ancient-insight-main
npm install
npm run dev
```

The Vite dev server runs on port `8080` and proxies `/api/*` requests to `http://127.0.0.1:8000`.

### Inference from CLI

```bash
python inference/predict.py --image path/to/image.png --preprocess --multiline --debug
```

### Training from CLI

```bash
python training/train.py --balanced_sampling --gradient_accumulation_steps 8 --image_size 384
```

### Dataset generation flow

```bash
python dataset/validate_dataset.py --data_dir dataset --json_out dataset/reports/precheck.json
python dataset/build_targets.py --data_dir dataset --map_file map.json --out_csv dataset/reports/targets_manifest.csv
python dataset/generate_synthetic.py --data_dir dataset --manifest dataset/reports/targets_manifest.csv
python dataset/postcheck.py --data_dir dataset --manifest dataset/reports/targets_manifest.csv
python dataset/validate_dataset.py --data_dir dataset --json_out dataset/reports/final_report.json --strict
```

## Key Files To Read Next

For deeper understanding:

- full system and API flow: [`backend/README.md`](backend/README.md)
- frontend structure and UI behavior: [`webapp/ancient-insight-main/README.md`](webapp/ancient-insight-main/README.md)
- model, dataset, training, metrics, preprocessing, and inference: [`training/README.md`](training/README.md)

## Current Limitations

- transliteration is mapping-based, not grammar-aware
- translation is pragmatic and depends on `deep_translator`, not a dedicated historical language model
- multiline OCR is line-segmentation-based, not full page-layout understanding
- no formal benchmark split is packaged in this repository yet
- reported metrics should be treated as project-run indicators, not publication-grade evaluation
