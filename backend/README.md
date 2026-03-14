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

# Backend README

This document explains the backend in simple and detailed terms.

The backend is the service layer of the project. It takes an uploaded image, runs OCR, performs transliteration and translation, and sends one structured response back to the frontend.

If you want one short summary:

> The backend is the bridge between the OCR model and the web UI. It receives the uploaded image, runs the OCR pipeline, converts the Brahmi output into readable scripts and languages, and returns both results and diagnostics.

## 1. What The Backend Does

The backend does not train the model. Its job is to serve the trained system.

It is responsible for:

- starting the API server
- loading the trained OCR model
- accepting uploaded image files
- calling the OCR inference code
- transliterating Brahmi text
- translating the output
- returning data in a frontend-friendly format

Main backend files:

- [`main.py`](main.py)
- [`transliterator.py`](transliterator.py)
- [`translator.py`](translator.py)
- [`requirements.txt`](requirements.txt)

## 2. Main Backend Libraries Used

| Library | Why it is used |
| --- | --- |
| `fastapi` | API framework |
| `uvicorn` | ASGI server to run the FastAPI app |
| `python-multipart` | file upload support |
| `torch` | model execution |
| `transformers` | TrOCR model and processor loading |
| `Pillow` | image handling inside the OCR pipeline |
| `deep-translator` | translation layer |

### What each one does here

#### `fastapi`

Used to define:

- routes
- request handling
- response models

#### `uvicorn`

Used to run the backend server locally.

#### `python-multipart`

Needed because the frontend uploads images as `multipart/form-data`.

#### `transformers` and `torch`

Used to load and run the trained TrOCR model inside the backend startup and OCR request flow.

#### `deep-translator`

Used for the translation step after transliteration.

## 3. The Main Backend File

The central backend file is:

- [`main.py`](main.py)

This file creates the FastAPI app and wires together:

- OCR
- transliteration
- translation

It is the entry point used when you run:

```bash
python -m uvicorn main:app --reload --port 8000
```

## 4. Startup Behavior

When the backend starts, it tries to load:

- processor from `model/brahmi_trocr`
- model from `model/brahmi_trocr`

This happens during the FastAPI startup event.

If model loading succeeds:

- OCR requests use the real trained model

If model loading fails:

- the backend still starts
- OCR endpoints return dummy Brahmi text

That fallback is useful during UI development, but not for real OCR.

## 5. API Endpoints

### `GET /`

Simple health check.

Response:

```json
{
  "message": "Brahmi OCR API is running"
}
```

### `POST /api/upload`

This is the main endpoint used by the frontend.

It accepts:

- one uploaded image file

It returns:

- OCR output
- transliteration
- translation
- OCR debug metadata

## 6. What Happens During `/api/upload`

The upload flow is:

1. The image file is saved to a temporary folder inside `backend/`.
2. The OCR pipeline is called.
3. The OCR result is extracted.
4. The Brahmi output is transliterated into:
   - Devanagari
   - Latin
5. The Devanagari output is translated into:
   - Hindi
   - English
6. A structured JSON response is built.
7. The temporary file is deleted.

This endpoint is therefore doing orchestration, not heavy OCR logic itself. The actual OCR logic lives in the inference module.

## 7. OCR Inside The Backend

The backend calls:

- [`../inference/predict.py`](../inference/predict.py)

More specifically, it uses:

- `load_trained_model(...)`
- `predict(...)`

The backend calls prediction with these important settings:

- `preprocess=True`
- `debug=True`
- `multiline=True`
- `return_base64=True`
- `image_size=384`

### What those settings mean

#### `preprocess=True`

The image is cleaned before OCR.

#### `debug=True`

The OCR result includes:

- token trace
- character trace
- text breakdown
- preprocessing metadata

#### `multiline=True`

The image is segmented into line regions before decoding.

#### `return_base64=True`

The processed image is returned in base64 form so the frontend can display it.

## 8. What The Backend Returns

The backend response model includes:

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

This is why the frontend can do more than just show final text. It can also visualize OCR internals.

## 9. Transliteration Layer

Transliteration is implemented in:

- [`transliterator.py`](transliterator.py)

This file loads mappings from:

- [`../brahmi.json`](../brahmi.json)

### What it does

It converts the predicted Brahmi text into:

- Devanagari text
- Latin transliteration

### How it works

The transliterator:

1. reads `brahmi.json`
2. extracts mapping objects
3. builds:
   - Brahmi to Devanagari mapping
   - Brahmi to Latin mapping
4. transliterates character by character

### Important limitation

This is a direct mapping approach. It is:

- deterministic
- easy to understand
- easy to audit

But it is not:

- context-aware grammar processing
- historical language interpretation
- intelligent word normalization

That is acceptable for a first OCR pipeline because it keeps the script conversion clear and simple.

## 10. Translation Layer

Translation is implemented in:

- [`translator.py`](translator.py)

It uses:

- `deep_translator.GoogleTranslator`

### Current translation behavior

The backend assumes the Devanagari transliteration is the best text to pass into translation.

It currently does:

- Hindi to Hindi translation for a normalized Hindi-like output
- Hindi to English translation for English output

### Why it is designed this way

The OCR model predicts Brahmi script.

The translator cannot directly work on Brahmi in this pipeline, so the system first maps Brahmi into Devanagari and then passes that result into translation.

### If translation fails

Fallback behavior is:

- Hindi output becomes the original Devanagari text
- English output becomes `"Translation failed"`

## 11. Request And Response Flow

The full backend-side data flow is:

1. Browser uploads image.
2. FastAPI receives file.
3. File is saved temporarily.
4. OCR is run through the inference module.
5. Brahmi text is returned from OCR.
6. Transliteration converts it into Devanagari and Latin.
7. Translation converts Devanagari into Hindi and English.
8. Backend returns one JSON response.
9. Temporary file is removed.

## 12. How The Backend Connects To The Frontend

The frontend calls:

- `POST /api/upload`

The backend enables CORS with permissive settings:

- all origins
- all methods
- all headers

For local development this is convenient.

In the frontend development setup, Vite proxies `/api` requests to:

- `http://127.0.0.1:8000`

So the frontend does not need to hardcode a second host when running locally.

## 13. Temporary File Handling

The backend stores uploads in a temporary folder under `backend/` and deletes them after the request is complete.

This is important because:

- uploaded images should not keep accumulating
- OCR works on a real file path
- the server stays cleaner during development

## 14. How To Run The Backend

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Start the server:

```bash
python -m uvicorn main:app --reload --port 8000
```

Once running, the main API is available at:

- `http://127.0.0.1:8000`

## 15. Backend Dependency Summary

[`requirements.txt`](requirements.txt) includes:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `deep-translator`
- `transformers`
- `torch`
- `Pillow`

## 16. Good Short Description For Reports

You can use wording like this:

> The backend is implemented using FastAPI and acts as the orchestration layer of the OCR system. It receives uploaded images, runs the trained TrOCR model through the inference pipeline, transliterates the predicted Brahmi text into Devanagari and Latin, translates the Devanagari output into Hindi and English, and returns a structured JSON response containing both final outputs and OCR diagnostics.

## 17. Current Limitations

- translation quality depends on an external general translation service
- transliteration is character-mapped, not context-aware
- the backend currently processes one uploaded image per request
- character-level spatial boxes are not returned yet
- if the OCR model is missing, the backend falls back to dummy text
