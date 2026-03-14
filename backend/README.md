# Backend README

This backend is the orchestration layer of the project. It does not train the model. Its job is to expose the OCR pipeline as a single API that the frontend can call.

Core flow:

`upload image -> OCR -> transliteration -> translation -> return structured JSON`

## Backend Responsibilities

[`main.py`](main.py) is responsible for:

- starting a FastAPI app
- loading the trained OCR model once at startup
- accepting uploaded image files
- running OCR through [`inference/predict.py`](../inference/predict.py)
- transliterating Brahmi text through [`transliterator.py`](transliterator.py)
- translating through [`translator.py`](translator.py)
- returning debug metadata that the frontend can visualize

## Startup Behavior

On startup the backend tries to load:

- processor: from `model/brahmi_trocr`
- model: from `model/brahmi_trocr`

If loading fails, the backend still starts, but OCR requests fall back to a dummy Brahmi string. That keeps the UI usable for development, but obviously not for real inference.

## API Contract

### `GET /`

Health check endpoint.

Returns:

```json
{
  "message": "Brahmi OCR API is running"
}
```

### `POST /api/upload`

Accepts one uploaded image file.

Processing steps:

1. Save uploaded file into a temporary backend folder.
2. Run OCR with:
   - preprocessing enabled
   - debug output enabled
   - multiline segmentation enabled
   - processed image returned as base64
3. Transliterate Brahmi into:
   - Devanagari
   - Latin
4. Translate the Devanagari text into:
   - Hindi
   - English
5. Return a single response object.
6. Delete the temporary file.

Response model fields:

- `brahmi_text`
- `devanagari_text`
- `latin_text`
- `hindi_translation`
- `english_translation`
- `debug_info`
- `lines`
- `token_trace`
- `character_trace`
- `base64_image`

## OCR Stage Inside The Backend

The backend calls [`predict()`](../inference/predict.py) with the following effective behavior:

- `preprocess=True`
- `debug=True`
- `multiline=True`
- `return_base64=True`
- `image_size=384`

That means the backend returns not only text, but also:

- processed image diagnostics
- line boxes
- per-line text
- token-level traces
- character-level traces

This is why the frontend can highlight detected lines and inspect detailed OCR outputs.

## Transliteration Layer

[`transliterator.py`](transliterator.py) loads character mappings from [`brahmi.json`](../brahmi.json).

Important implementation detail:

- `brahmi.json` is not strict machine-clean JSON throughout
- the transliterator extracts per-character objects with a regex
- it builds two dictionaries:
  - Brahmi to Devanagari
  - Brahmi to Latin

The transliteration is direct character mapping. It is not a linguistic parser. That means:

- it is deterministic
- it is easy to audit
- it does not model context-sensitive grammar or historical spelling normalization

This is the right choice for a first OCR pipeline because it keeps the transformation transparent.

## Translation Layer

[`translator.py`](translator.py) uses `deep_translator.GoogleTranslator`.

Current translation strategy:

- Hindi output: translate Hindi to Hindi
- English output: translate Hindi to English

Why it works this way:

- the OCR output is first mapped into Devanagari
- the backend assumes the Devanagari text is close enough to a Hindi or Sanskrit-like string to pass into the translator
- Hindi-to-Hindi is used as a pragmatic normalization pass

This is not a historical philology engine. It is a practical demo layer that makes the OCR result easier to read in modern scripts and languages.

Fallback behavior:

- if translation fails, Hindi falls back to the input Devanagari text
- English returns `"Translation failed"`

## Request And Response Flow

The full backend flow is:

1. Frontend sends `multipart/form-data` to `/api/upload`.
2. Backend stores the image temporarily.
3. OCR model predicts Brahmi Unicode text.
4. Backend converts Brahmi to Devanagari.
5. Backend converts Brahmi to Latin.
6. Backend translates Devanagari to Hindi and English.
7. Backend returns one structured response.
8. Frontend renders the response across its left and right panels.

## CORS And Frontend Integration

The backend enables permissive CORS:

- origins: `*`
- methods: `*`
- headers: `*`

For local development this is fine. For deployment it should be restricted.

The frontend dev server is configured to proxy `/api` requests to `http://127.0.0.1:8000`, so the browser can call the backend without hardcoding a separate host inside the React code.

## Run The Backend

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Start the server:

```bash
python -m uvicorn main:app --reload --port 8000
```

## Backend Dependencies

[`requirements.txt`](requirements.txt) includes:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `deep-translator`
- `transformers`
- `torch`
- `Pillow`

## Current Limitations

- translation quality depends on external translation behavior rather than a domain-specific epigraphy model
- transliteration is character-level, not context-aware
- the API currently accepts single image uploads only
- character spatial boxes are not returned yet, only textual traces and line boxes
