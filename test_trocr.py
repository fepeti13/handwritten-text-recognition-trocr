from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import sys

# Load image
image_path = "data/Screenshot from 2025-10-07 18-27-23.png"
image = Image.open(image_path).convert("RGB")

# Load base model
print("Loading model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Run OCR
print("Processing image...")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"\nRecognized text: {text}")

# Load large model
print("Loading model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

# Run OCR
print("Processing image...")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"\nRecognized text: {text}")