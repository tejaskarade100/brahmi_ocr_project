# Brahmi OCR Improvement Plan

## Current implementation audit

Some of the original plan is already partially implemented in code, but it is not complete enough yet to expect strong accuracy.

Already present:

- `dataset/build_targets.py` now supports category-style quotas for fixed classes, words, phrases, sentences, and multiline.
- `dataset/generate_synthetic.py` now generates explicit `word`, `phrase`, `sentence`, and `multiline` mixed samples and writes metadata into `labels.json`.
- `training/dataset_loader.py` now has runtime caps and a weighted sampler.
- `training/train.py` now has gradient accumulation and category-wise evaluation metrics.
- `inference/predict.py` already exposes useful debug output such as `text_breakdown`, `token_trace`, `character_trace`, and preprocessing metadata.

Still missing or not safe enough yet:

- There is no proof yet that the current changes are correct end-to-end.
- The training path still needs a correctness pass before relying on it for accuracy.
- Multiline handling is still only synthetic sequence generation, not real layout-aware inference.
- The backend does not yet return preprocessed images, line divisions, or character-level spatial boxes for a frontend overlay.
- A 95%+ accuracy target cannot be promised until there is a proper benchmark split covering chars, words, phrases, multiline, and real photos.

## Current diagnosis

The current pipeline was structurally biased toward short outputs, which caused sequence collapse toward single-character predictions.

The codebase has started fixing this, but the remaining work is now about:

- validating correctness
- hardening the balanced training path
- improving real-image robustness
- adding backend outputs for later frontend visualization

## Newly recommended implementation plan

### Phase 0: Verify and harden the current implementation first

Goal: make sure the newly added logic is actually correct before generating more data or running long training jobs.

Must verify and fix:

1. `training/train.py` scheduler step count must match optimizer steps after gradient accumulation.
2. `training/train.py` weighted sampling must work correctly even if multiple dataset roots are combined.
3. `training/dataset_loader.py` summaries must stay accurate after sample caps or debug sample truncation.
4. Dry-run generation must be validated for all `sequence_type` values.
5. Training must be smoke-tested on a small subset to ensure:
   - batch loading works
   - labels decode correctly
   - category IDs are correct
   - evaluation metrics do not crash

Success criteria for Phase 0:

- one dry-run manifest builds cleanly
- one dry synthetic generation run writes valid `labels.json`
- one short training run completes validation without crashing
- category breakdown printed by training looks sensible

### Phase 1: Rebalance the dataset for accuracy and efficiency

Goal: stop overwhelming the model with single characters and fit training inside free GPU limits.

Recommended target mix:

- characters + ngrams: 25% to 35%
- single words: 20% to 25%
- phrases: 25% to 30%
- sentences + multiline: 20% to 25%

Recommended practical counts:

- fixed classes: cap each folder at 60 to 120 samples
- words: 8,000 to 12,000 total
- phrases: 8,000 to 12,000 total
- sentences + multiline: 6,000 to 10,000 total

Recommended runtime controls:

- `--max_fixed_per_class 80`
- `--max_words 10000`
- `--max_phrases 10000`
- `--max_long_sentences 8000`
- `--balanced_sampling`

### Phase 2: Improve mixed-text generation quality

Goal: make the model learn real reading behavior, not just isolated glyphs.

Implement and verify:

1. explicit generation modes:
   - `word`
   - `phrase`
   - `sentence`
   - `multiline`
2. richer spacing and line-break variation
3. dynamic font scaling for long text
4. better sequence metadata in `labels.json`

Important constraint:

- do not rely only on random syllable strings
- add a curated phrase bank and sentence bank
- use random strings as augmentation, not as the only language source

### Phase 3: Improve image robustness

Goal: support various shapes, aspect ratios, resolutions, surfaces, and difficult real images.

Keep and expand:

1. aspect-ratio-safe letterboxing
2. multiple image sizes for training and inference
3. clean / manuscript / stone style variation

Add next:

1. perspective warp
2. low-resolution rendering
3. JPEG artifacts
4. stronger crop and padding variation
5. rough background compositing for stone / paper / palm-leaf style surfaces

This phase is important for your goal of handling chars, words, phrases, sentences, multiline text, and images of many shapes and resolutions.

### Phase 4: Train in stages instead of one giant run

Goal: improve sequence learning while staying within Colab/Kaggle limits.

Recommended schedule:

1. Stage A:
   - balanced chars + words
   - 1 to 2 epochs
   - image size `320` or `384`
2. Stage B:
   - full balanced mix
   - words + phrases + long text
   - 3 to 5 epochs
3. Stage C:
   - hardest real-like and long samples
   - 1 to 2 epochs

Do not start from a huge unbalanced dataset. That wastes free GPU time and reinforces the wrong output prior.

### Phase 5: Tighten training metrics

Goal: measure collapse and actual reading performance properly.

Track:

- CER
- WER
- exact match
- first-character accuracy
- prediction-to-label length ratio
- separate metrics for:
  - chars
  - words
  - phrases
  - long sequences

This is necessary because a single overall metric can hide the exact failure mode you described.

### Phase 6: Add true multiline and line-aware inference

Goal: handle real multiline inscriptions instead of only synthetic multiline training strings.

The current inference path is still single-image OCR.

Next addition:

1. line segmentation before OCR
2. crop each detected line
3. OCR each line independently
4. reassemble lines in reading order

This is the correct way to support multiline reliably.

### Phase 7: Make the backend frontend-ready

Goal: prepare the API so your later frontend can visualize preprocessing and OCR structure.

For the frontend, you said you want to show:

- how the image is preprocessed
- the preprocessed image itself
- whether the input is a character, word, phrase, or multiline text
- line divisions on the preprocessed image
- which character corresponds to what
- final output

The backend should eventually return:

- original image metadata
- preprocessed image path or base64 image
- preprocess step metadata
- predicted text
- text breakdown
- token trace
- character trace
- detected line boxes
- per-line OCR output
- optional character boxes later

Important note:

- textual `character_trace` already exists
- preprocessed debug metadata already exists
- line boxes do not exist yet
- character spatial boxes do not exist yet

### Phase 8: Build a real benchmark before claiming 95%+

Goal: make the 95%+ target measurable instead of guessed.

Create a fixed benchmark split with:

1. single characters
2. consonant + matra combinations
3. words
4. phrases
5. sentences
6. multiline
7. clean synthetic
8. noisy synthetic
9. real photos

Track accuracy separately for each bucket.

## Recommended implementation order

1. Phase 0 correctness audit and smoke test.
2. Regenerate the manifest with the new quota logic.
3. Run a dry generation pass and inspect all sequence types.
4. Run a short capped training run and verify metrics.
5. Generate the full balanced dataset only after the short run looks healthy.
6. Run staged training on the balanced dataset.
7. Add line-aware inference.
8. Add frontend-ready API outputs.

## Practical training target

For the next iteration, aim for:

- total trainable samples per run: 25k to 40k
- balanced sequence mix
- capped fixed classes
- strong phrase and long-text coverage
- staged training
- per-category evaluation

This is much more likely to improve accuracy than simply making the dataset larger.

## Accuracy target reality

You want:

- good character recognition
- good word identification
- support for various aspect ratios and resolutions
- support for chars, words, phrases, and multiline inputs
- 95%+ accuracy

That is a valid target, but it should be treated as a benchmark goal, not a guarantee from the current implementation.

The correct path is:

1. verify the new pipeline
2. train on a balanced capped dataset
3. evaluate by category
4. improve real multiline and frontend outputs

Do not trust a single overall accuracy number. You need separate targets for chars, words, phrases, multiline, synthetic, and real images.
