import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AddedToken

print("Loading processor...")
model_dir = "model/brahmi_trocr"
processor = TrOCRProcessor.from_pretrained(model_dir, use_fast=False)

# Let's say we have this string
text = "𑀓𑀸 𑀔𑀺 𑀕𑀼"

print(f"Original text: {text}")

# Tokenize
tokens = processor.tokenizer(text)
print(f"Token IDs: {tokens.input_ids}")

# Decode back
decoded = processor.tokenizer.decode(tokens.input_ids)
print(f"Decoded back: {decoded}")

# Let's check vocab size
print(f"Vocab size: {len(processor.tokenizer)}")
print(f"Does it have Brahmi chars? '𑀓' in tokenizer: {'𑀓' in processor.tokenizer.get_vocab()}")
