# scripts/analyze-text-lengths.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/metadata.csv')

# Text length stats
df['text_length'] = df['text'].str.len()

print("=== Text Length Statistics ===")
print(f"Total lines: {len(df)}")
print(f"Mean length: {df['text_length'].mean():.1f} chars")
print(f"Median length: {df['text_length'].median():.1f} chars")
print(f"Min: {df['text_length'].min()}")
print(f"Max: {df['text_length'].max()}")

print("\n=== Distribution ===")
print(f"1 char: {len(df[df['text_length'] == 1])}")
print(f"2-3 chars: {len(df[df['text_length'].between(2, 3)])}")
print(f"4-10 chars: {len(df[df['text_length'].between(4, 10)])}")
print(f"11-20 chars: {len(df[df['text_length'].between(11, 20)])}")
print(f"20+ chars: {len(df[df['text_length'] > 20])}")

# Show some examples
print("\n=== Examples ===")
print("Very short:")
print(df[df['text_length'] <= 3]['text'].head(10).tolist())

print("\nMedium:")
print(df[df['text_length'].between(10, 20)]['text'].head(5).tolist())

print("\nLong:")
print(df[df['text_length'] > 30]['text'].head(5).tolist())

# Plot
plt.figure(figsize=(10, 5))
plt.hist(df['text_length'], bins=50, edgecolor='black')
plt.xlabel('Text Length (characters)')
plt.ylabel('Count')
plt.title('Text Length Distribution')
plt.savefig('data/processed/text_length_distribution.png')
print("\n✓ Plot saved to data/processed/text_length_distribution.png")