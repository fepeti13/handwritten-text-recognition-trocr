# scripts/evaluate-test-set.py
import pandas as pd
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import os
from tqdm import tqdm

def compute_cer(prediction, reference):
    if len(reference) == 0:
        return 0 if len(prediction) == 0 else 1
    
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


print("Loading fine-tuned model...")
processor = TrOCRProcessor.from_pretrained('models/base-hungarian')
model = VisionEncoderDecoderModel.from_pretrained('models/base-hungarian')

df = pd.read_csv('data/processed/test.csv')
print(f"Test set: {len(df)} samples")

images_dir = 'data/processed/images'

predictions = []
references = []

print("Evaluating...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join(images_dir, row['image_path'])
    image = Image.open(image_path).convert("RGB")
    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=64)
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    predictions.append(pred_text)
    references.append(row['text'])

cer_scores = [compute_cer(pred, ref) for pred, ref in zip(predictions, references)]
avg_cer = sum(cer_scores) / len(cer_scores)

print("\n" + "="*60)
print("TEST SET RESULTS (Fine-tuned Model)")
print("="*60)
print(f"Test samples: {len(predictions)}")
print(f"CER: {avg_cer*100:.2f}%")
print(f"Accuracy: {(1-avg_cer)*100:.2f}%")
print("="*60)

print("\n20 Example Predictions:")
print("-"*60)
for i in range(min(20, len(predictions))):
    print(f"\nGround Truth: {references[i]}")
    print(f"Prediction:   {predictions[i]}")
    print(f"CER: {cer_scores[i]*100:.1f}%")
    print("-"*60)

results_df = pd.DataFrame({
    'image_path': df['image_path'].tolist(),
    'ground_truth': references,
    'prediction': predictions,
    'cer': [s*100 for s in cer_scores]
})
results_df.to_csv('data/processed/test_predictions_finetuned.csv', index=False)
print(f"\n✓ Results saved to: data/processed/test_predictions_finetuned.csv")