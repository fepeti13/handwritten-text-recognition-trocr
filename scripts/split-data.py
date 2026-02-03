# scripts/split-data.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/processed/metadata.csv')
print(f"Total: {len(df)} lines")

# Split 80/10/10
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print(f"Train: {len(train_df)}")
print(f"Val: {len(val_df)}")
print(f"Test: {len(test_df)}")