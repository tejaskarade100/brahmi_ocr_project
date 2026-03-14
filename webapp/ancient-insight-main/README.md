# Frontend README

This frontend is the presentation and inspection layer for the Brahmi OCR system. It is a React + Vite + TypeScript application that sends uploaded inscription images to the FastAPI backend and renders both the final outputs and the OCR diagnostics.

It is not just a result page. It is designed as an OCR analysis UI.

## Frontend Purpose

The UI is split into two panels:

- left panel: source image inspection
- right panel: OCR analysis and translation output

The frontend helps a user understand:

- what image was uploaded
- what the backend-preprocessed image looks like
- which text lines were detected
- what each line decoded to
- how the model tokenized and decoded the selected line
- what the transliterated and translated outputs are

## Main Flow

1. User uploads an image.
2. Frontend sends the file to `/api/upload`.
3. Backend returns OCR data and diagnostics.
4. Frontend displays:
   - processed image
   - line overlays
   - Brahmi text
   - Latin transliteration
   - Devanagari transliteration
   - Hindi translation
   - English translation
   - token trace
   - character trace

## App Structure

Main entry points:

- [`src/App.tsx`](src/App.tsx)
- [`src/pages/Index.tsx`](src/pages/Index.tsx)

Important components:

- [`src/components/InputModule.tsx`](src/components/InputModule.tsx)
- [`src/components/OutputModule.tsx`](src/components/OutputModule.tsx)
- [`src/types/ocr.ts`](src/types/ocr.ts)

## Page-Level State

[`Index.tsx`](src/pages/Index.tsx) holds the shared application state:

- `isProcessing`
- `isComplete`
- `ocrResult`
- `errorMessage`
- `activeLineIndex`

This is the state that ties both panels together.

For example:

- when a line is selected on the image overlay in the left panel
- the diagnostics for that same line are shown on the right panel

## Left Panel: Input Module

[`InputModule.tsx`](src/components/InputModule.tsx) handles the full image-side experience.

It supports:

- drag-and-drop upload
- file picker upload
- image clearing and re-upload
- switching between original and processed image
- toggling line overlays
- selecting a line box by clicking it
- showing a processing animation during OCR

Key implementation details:

- uploaded files are previewed locally through `URL.createObjectURL`
- the component measures the rendered image frame so line boxes scale correctly
- overlays are drawn only on the processed image view because the backend line boxes correspond to the processed image returned by OCR

The left panel therefore acts like a lightweight visual debugger for preprocessing and line segmentation.

## Right Panel: Output Module

[`OutputModule.tsx`](src/components/OutputModule.tsx) is the text and diagnostics panel.

It renders:

- transliteration section
  - Brahmi
  - Latin
  - Devanagari
- translation section
  - English
  - Hindi
- global OCR breakdown badges
- line-by-line OCR breakdown
- selected line diagnostics
- token trace pills
- character trace pills
- technical analysis details

Behavior worth noting:

- English translation is animated with a typewriter-style reveal
- selecting a line changes the diagnostics shown below
- grouped token traces and character traces are resolved automatically from either global or line-level payloads

## OCR Payload Type System

[`src/types/ocr.ts`](src/types/ocr.ts) defines the frontend contract for the backend response.

Important payload shapes:

- `OCRResponse`
- `OCRLineResult`
- `OCRBoundingBox`
- `OCRTextBreakdown`
- `OCRTokenTraceEntry`
- `OCRCharacterTraceEntry`

This keeps the UI implementation explicit about what the backend returns and prevents the OCR response from turning into an untyped blob.

## Backend Communication

The frontend sends uploads with:

- `fetch("/api/upload", { method: "POST", body: formData })`

There is no hardcoded backend host in the app code. Development routing is handled by Vite proxy configuration in [`vite.config.ts`](vite.config.ts).

Current proxy:

- `/api/* -> http://127.0.0.1:8000`

The frontend dev server runs on:

- `http://localhost:8080`

## Testing

There is a UI test suite in [`src/test/ocr-ui.test.tsx`](src/test/ocr-ui.test.tsx).

The tests cover:

- rendering translated output and line diagnostics
- switching between processed and original views
- showing backend error states when upload fails

This is useful because the frontend depends heavily on a fairly complex OCR response shape.

## Run The Frontend

Install dependencies:

```bash
cd webapp/ancient-insight-main
npm install
```

Start development server:

```bash
npm run dev
```

Run tests:

```bash
npm test
```

Build for production:

```bash
npm run build
```

## Tech Stack

Core stack:

- React
- TypeScript
- Vite
- Tailwind CSS
- Radix UI
- Framer Motion
- TanStack Query

## Current Limitations

- the frontend depends on backend line boxes and traces already being correct
- no per-character spatial bounding boxes are drawn yet
- no history view or batch-processing view exists yet
- OCR quality still depends on the model and the backend translation heuristics
