import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class TrOCRDataset(Dataset):
    def __init__(self, csv_path, images_dir, processor, max_length=64):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.processor = processor
        self.max_length = max_length
        print(f"Loaded {len(self.df)} samples from {csv_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, row['image_path'])
        text = row['text']
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        ).input_ids
        
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 
                  for label in labels]
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels)
        }