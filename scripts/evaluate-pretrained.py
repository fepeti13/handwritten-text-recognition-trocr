# scripts/evaluate-pretrained.py
import pandas as pd
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import os
from tqdm import tqdm

# Character Error Rate calculation
def compute_cer(prediction, reference):
    """Simple CER calculation"""
    if len(reference) == 0:
        return 0 if len(prediction) == 0 else 1
    
    # Levenshtein distance
    d = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(prediction) + 1):
        d[0][j] = j
    
    for i in range(1, len(reference) + 1):
        for j in range(1, len(prediction) + 1):
            if reference[i-1] == prediction[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(reference)][len(prediction)] / len(reference)


# Load model
print("Loading pretrained model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Load test data (use small sample for speed)
df = pd.read_csv('data/processed/test.csv')
print(f"Test set size: {len(df)}")

# Sample for quick test (or use all)
SAMPLE_SIZE = 100  # Change to len(df) for full evaluation
df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

images_dir = 'data/processed/images'

predictions = []
references = []

print(f"\nEvaluating on {len(df_sample)} samples...")

for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    # Load image
    image_path = os.path.join(images_dir, row['image_path'])
    image = Image.open(image_path).convert("RGB")
    
    # Predict
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=64)
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    predictions.append(pred_text)
    references.append(row['text'])

# Compute metrics
cer_scores = [compute_cer(pred, ref) for pred, ref in zip(predictions, references)]
avg_cer = sum(cer_scores) / len(cer_scores)

print("\n" + "="*60)
print("BASELINE RESULTS (Pretrained Model - No Fine-tuning)")
print("="*60)
print(f"Samples evaluated: {len(predictions)}")
print(f"Average CER: {avg_cer*100:.2f}%")
print(f"Accuracy: {(1-avg_cer)*100:.2f}%")
print("="*60)

# Show examples
print("\nExample Predictions:")
print("-"*60)
for i in range(min(20, len(predictions))):
    print(f"\nGround Truth: {references[i]}")
    print(f"Prediction:   {predictions[i]}")
    print(f"CER: {cer_scores[i]*100:.1f}%")
    print("-"*60)

# Save results
results_df = pd.DataFrame({
    'image_path': df_sample['image_path'].tolist(),
    'ground_truth': references,
    'prediction': predictions,
    'cer': [s*100 for s in cer_scores]
})
results_df.to_csv('data/processed/baseline_predictions.csv', index=False)

print(f"\n✓ Full results saved to: data/processed/baseline_predictions.csv")

# Hungarian character analysis
print("\n" + "="*60)
print("Hungarian Characters Analysis")
print("="*60)
hungarian_chars = 'áéíóöőúüű'
for char in hungarian_chars:
    gt_count = sum(ref.count(char) for ref in references)
    pred_count = sum(pred.count(char) for pred in predictions)
    print(f"'{char}': Ground truth={gt_count}, Predicted={pred_count}")