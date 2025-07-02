import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['query'] = df['query'].apply(clean_text)
    df['image descriptions'] = df['image descriptions'].apply(clean_text)
    df = df.dropna(subset=['query', 'image descriptions', 'Toxic Category'])
    df = df.drop_duplicates()
    le = LabelEncoder()
    df['Toxic Category Encoded'] = le.fit_transform(df['Toxic Category'])
    df.to_csv(output_csv, index=False)
    return df, le

if __name__ == "__main__":
    preprocess(r'\data\cellula-toxic.csv', r'\data\cleaned.csv')