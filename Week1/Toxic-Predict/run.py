from src.preprocess import preprocess
from src.tokenize_and_split import tokenize_and_split
from models.model import train_model
import pandas as pd

def main():
    
    preprocess(
        input_csv='data/cellula-toxic.csv',
        output_csv='data/cleaned.csv'
    )

    X_train, X_test, y_train, y_test = tokenize_and_split(
        cleaned_csv='data/cleaned.csv',
        tokenizer_path='data/tokenizer.pkl',
        max_words=10000,
        max_len=150,
        test_size=0.2
    )
    train_df = pd.read_csv('data/cleaned.csv').iloc[:len(X_train)]
    test_df = pd.read_csv('data/cleaned.csv').iloc[len(X_train):]
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    train_model(
        train_csv='data/train.csv',
        eval_csv='data/test.csv',
        tokenizer_path='data/tokenizer.pkl',
        model_path='models/toxic_classifier.keras'
    )

if __name__ == "__main__":
    main()