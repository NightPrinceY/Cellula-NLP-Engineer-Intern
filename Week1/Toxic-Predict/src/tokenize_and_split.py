import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def tokenize_and_split(cleaned_csv, tokenizer_path, max_words=10000, max_len=150, test_size=0.2):
    df = pd.read_csv(cleaned_csv)
    df['text'] = df['query'] + ' ' + df['image descriptions']
    X = df['text'].values
    y = df['Toxic Category Encoded'].values

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post', truncating='post')

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=test_size, random_state=42, stratify=y
    )

    # Ensure the directory exists
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    tokenize_and_split(
        r'\data\cleaned.csv',
        r'\data\tokenizer.pkl'
    )