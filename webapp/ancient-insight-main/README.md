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

# Frontend README

This document explains the frontend in easy and detailed language.

The frontend is the user-facing part of the Brahmi OCR system. It is where users:

- upload an image
- inspect OCR processing
- read transliteration and translation
- examine line-by-line OCR diagnostics

If you want one short summary:

> The frontend is a React-based OCR analysis interface that uploads inscription images, receives structured OCR results from the backend, and shows both the final outputs and the internal debugging information in a clear two-panel layout.

## 1. What The Frontend Does

The frontend is not only a form that uploads an image and prints a result.

It also shows:

- original image
- processed image
- detected line overlays
- Brahmi output
- Latin transliteration
- Devanagari transliteration
- Hindi translation
- English translation
- token trace
- character trace

So the frontend acts both as:

- a product UI
- an OCR inspection tool

## 2. Main Frontend Technologies Used

The frontend uses:

- React
- TypeScript
- Vite
- Tailwind CSS
- Radix UI
- Framer Motion
- TanStack Query

### What each technology is doing

#### React

Used for:

- component-based UI
- state management through hooks
- rendering OCR results dynamically

#### TypeScript

Used for:

- typed OCR response objects
- safer component props
- cleaner frontend logic

#### Vite

Used for:

- local development server
- fast rebuilds
- proxying backend API requests

#### Tailwind CSS

Used for:

- utility-first styling
- layout and spacing
- quick responsive UI design

#### Radix UI

Used through UI primitives and supporting components for cleaner interface behavior.

#### Framer Motion

Used for:

- transitions
- animated loading states
- subtle reveal effects

#### TanStack Query

Included in the app structure for data/query support, though the upload flow itself is currently implemented directly with `fetch`.

## 3. Frontend Folder Structure

Important frontend files:

- [`src/App.tsx`](src/App.tsx)
- [`src/pages/Index.tsx`](src/pages/Index.tsx)
- [`src/components/InputModule.tsx`](src/components/InputModule.tsx)
- [`src/components/OutputModule.tsx`](src/components/OutputModule.tsx)
- [`src/types/ocr.ts`](src/types/ocr.ts)
- [`vite.config.ts`](vite.config.ts)
- [`src/test/ocr-ui.test.tsx`](src/test/ocr-ui.test.tsx)

## 4. How The Frontend Is Organized

The app is mainly a two-panel screen:

- left side: image input and visual inspection
- right side: OCR results and diagnostics

This design is useful because the user can compare:

- what the system saw
- what the system predicted

## 5. Main Entry Points

### `src/App.tsx`

This file sets up:

- React router
- query client provider
- tooltip provider
- toaster components

It is the top-level application wrapper.

### `src/pages/Index.tsx`

This is the main OCR page.

It holds the shared application state:

- `isProcessing`
- `isComplete`
- `ocrResult`
- `errorMessage`
- `activeLineIndex`

This file connects the left and right panels together.

## 6. Shared Frontend State

The page-level state is important because both panels need access to the same OCR result.

For example:

- the left panel lets the user click a line overlay
- the right panel then shows diagnostics for that selected line

That means the frontend is not made of two unrelated sections. Both sides are synchronized through the shared `activeLineIndex` and OCR response state.

## 7. Left Panel: Input Module

The left panel is implemented in:

- [`src/components/InputModule.tsx`](src/components/InputModule.tsx)

This component handles:

- file upload
- drag-and-drop upload
- image preview
- processed/original image switching
- line overlay drawing
- overlay toggling
- re-upload
- clearing the current image

### What the input module shows

It shows:

- the uploaded image
- the processed image returned by the backend
- line count
- category hint
- line selection overlays

### Important implementation details

#### Local image preview

The component uses `URL.createObjectURL(file)` so the user can immediately see the uploaded file.

#### Render-frame calculation

The component measures how the image is displayed in the browser so bounding boxes from the backend can be scaled correctly onto the visible image.

This is important because:

- the image is shown using `object-contain`
- the displayed size is not always the original size
- line boxes must still align correctly

#### Processed image overlay logic

Line overlays are only accurate on the processed image view because the backend returns boxes for the processed OCR image.

So the component lets the user switch between:

- original image
- processed image

That makes the OCR pipeline easier to understand visually.

## 8. Right Panel: Output Module

The right panel is implemented in:

- [`src/components/OutputModule.tsx`](src/components/OutputModule.tsx)

This component renders the actual OCR outputs and debugging information.

It shows:

- transliteration section
- translation section
- OCR breakdown badges
- line-by-line OCR output
- selected-line diagnostics
- token trace
- character trace
- technical analysis panel

## 9. Transliteration And Translation Display

The output panel explicitly separates three script forms:

- Brahmi
- Latin
- Devanagari

And then it shows:

- English translation
- Hindi translation

This is useful because the user can see the transformation stage by stage rather than only seeing the final translation.

## 10. Line Breakdown View

The frontend shows all detected lines returned by the backend.

For each line it can show:

- line index
- bounding box coordinates
- decoded text
- text breakdown

When a line is selected:

- the corresponding line overlay is highlighted on the left
- detailed diagnostics are shown on the right

This makes the frontend a real OCR inspection tool, not just a results page.

## 11. Token Trace And Character Trace

One of the strongest parts of this frontend is that it displays OCR debug traces.

### Token trace

Shows:

- token string
- token id
- whether token is special
- approximate confidence

### Character trace

Shows:

- decoded character
- Unicode codepoint
- Unicode name
- whether the character is a space

This helps explain:

- what the model decoded
- how that text is represented internally
- where possible OCR issues may be happening

## 12. The OCR Response Type System

Frontend OCR response types are defined in:

- [`src/types/ocr.ts`](src/types/ocr.ts)

Important types include:

- `OCRResponse`
- `OCRLineResult`
- `OCRBoundingBox`
- `OCRTextBreakdown`
- `OCRTokenTraceEntry`
- `OCRCharacterTraceEntry`

Why this matters:

- the backend returns a complex JSON payload
- TypeScript helps keep the frontend predictable
- components know exactly which fields are expected

## 13. How The Frontend Talks To The Backend

The upload call is made with:

- `fetch("/api/upload", { method: "POST", body: formData })`

That means:

- the frontend does not hardcode a backend hostname in the request code
- Vite development proxy handles routing during local development

Proxy setup is defined in:

- [`vite.config.ts`](vite.config.ts)

Current proxy behavior:

- `/api/*` is proxied to `http://127.0.0.1:8000`

Frontend dev server:

- `http://localhost:8080`

## 14. User Flow In The Frontend

The user flow is:

1. User uploads an image.
2. Frontend switches to processing state.
3. File is sent to backend.
4. OCR response returns.
5. Frontend stores OCR result in state.
6. First detected line becomes active by default.
7. Left panel shows image and overlays.
8. Right panel shows OCR output and diagnostics.

If the request fails:

- an error state is shown

## 15. Visual Design Behavior

The UI includes several deliberate behaviors:

- processing animation while OCR runs
- typewriter animation for English output
- badge-based text statistics
- overlay highlighting for selected lines
- original/processed image mode switching

These are not only cosmetic. They help users understand where the OCR result came from.

## 16. Testing

Frontend tests are in:

- [`src/test/ocr-ui.test.tsx`](src/test/ocr-ui.test.tsx)

The tests cover:

- successful OCR rendering
- line selection and diagnostics
- processed/original image switching
- backend failure states

This is important because the UI depends on a structured OCR response and interactive line selection behavior.

## 17. How To Run The Frontend

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

Build production bundle:

```bash
npm run build
```

## 18. Good Short Description For Reports

You can describe the frontend like this:

> The frontend is a React and TypeScript web application designed as both a user interface and an OCR diagnostics tool. It allows users to upload inscription images, inspect the backend-preprocessed image, view detected line overlays, and read the OCR output, transliteration, translation, and token-level debugging information in a synchronized two-panel layout.

## 19. Current Limitations

- overlays currently work at line level, not character-box level
- no history or batch session view exists yet
- the frontend assumes backend traces and boxes are already correct
- OCR quality still depends on the backend model output
