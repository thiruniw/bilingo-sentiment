import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load Sinhala dataset
sinhala_ds = load_dataset("theekshana/sinhala-news-sentiment-classification")
sinhala_df = pd.DataFrame(sinhala_ds['train'])

# Keep only necessary columns and rename
sinhala_df = sinhala_df[['comment', 'simplified_label']].rename(columns={
    'comment': 'text',
    'simplified_label': 'label'
})

# Create train/test split
sinhala_train, sinhala_test = train_test_split(
    sinhala_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=sinhala_df['label']
)

# Clean text
sinhala_train['text'] = sinhala_train['text'].str.replace('\n', ' ').str.strip()
sinhala_test['text'] = sinhala_test['text'].str.replace('\n', ' ').str.strip()

print("Sinhala train/test ready with 'text' and 'label' columns!")
print(sinhala_train.head())


# Load English dataset
english_ds = load_dataset("carant-ai/english_sentiment_dataset")
english_df = pd.DataFrame(english_ds['train'])

# Keep only necessary columns
english_df = english_df[['text', 'label_text']].rename(columns={'label_text': 'label'})

english_df = english_df.sample(n=1000, random_state=42).reset_index(drop=True)

# Clean text
english_df['text'] = english_df['text'].str.replace('\n', ' ').str.strip()

# capitalize labels for consistency
english_df['label'] = english_df['label'].str.capitalize()

# create train/test split 
english_train, english_test = train_test_split(
    english_df,
    test_size=0.2,
    random_state=42,
    stratify=english_df['label']
)

print(" English train/test ready!")
