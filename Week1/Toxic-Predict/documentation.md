# Toxic Comment Classification Project Documentation

## Week 1 Tasks

- [x] **Data Preprocessing**
    - Clean and normalize text data  
    - Tokenize input text  
    - Encode multi-class labels  
    - Split Data  
- [x] **PEFT-LoRA on DistilBERT**
    - Set up parameter-efficient fine-tuning (LoRA)  
    - Initialize DistilBERT with pre-trained weights  
    - Configure training parameters for multi-class classification  
    - Begin initial fine-tuning experiments  
- [x] **Deep Learning Baseline**
    - Design a deep learning classifier architecture (e.g., CNN/LSTM with embeddings)  
    - Implement training pipeline  
    - Train on the same dataset for direct comparison  

*Both DistilBERT (with LoRA) and a custom deep learning model were developed and trained for comparison.*

## 1. Data Preprocessing Steps

### A. Loading Raw Data
- The raw dataset is stored in `data/cellula-toxic.csv`. This CSV file contains the original comments and their associated labels.
- The data is loaded using pandas (`pd.read_csv`) in the preprocessing scripts (see `src/preprocess.py`).

### B. Data Cleaning
- The cleaning process involves:
  - Lowercasing all text to ensure uniformity.
  - Removing special characters, punctuation, and extra whitespace.
  - Optionally removing stopwords and performing stemming/lemmatization (as implemented in `src/preprocess.py`).
- Cleaned data is saved to `data/cleaned.csv` for further processing.

### C. Tokenization
- A custom tokenizer is built and/or loaded (see `src/tokenize_and_split.py`).
- The tokenizer is either trained on the cleaned text or loaded from `data/tokenizer.json` or `data/tokenizer.pkl`.
- Each comment is converted into a sequence of integer tokens, suitable for model input.

### D. Label Encoding
- Multi-class labels are mapped to integer values using the mapping defined in `data/label_map.json`.
- This ensures consistency in label representation for both training and evaluation.

### E. Data Splitting
- The cleaned and tokenized data is split into training, validation, and test sets.
  - Training set: `data/train.csv`
  - Validation set: `data/eval.csv`
  - Test set: `data/test.csv`
- The split is typically stratified to maintain label distribution across sets (see logic in `src/tokenize_and_split.py`).

### F. Saving Artifacts
- The tokenizer object is saved as `data/tokenizer.json` and `data/tokenizer.pkl` for reproducibility.
- The label mapping is saved as `data/label_map.json`.
- The processed datasets are saved as CSV files in the `data/` directory.

---

#### Example Code References

- **Data Cleaning:** See `src/preprocess.py` for functions that handle text normalization and cleaning.
- **Tokenization & Splitting:** See `src/tokenize_and_split.py` for tokenization logic and dataset splitting.
- **Label Mapping:** The label mapping logic is also handled in `src/tokenize_and_split.py` and saved to `data/label_map.json`.

## 2. Hyperparameters & Training Details

### A. DistilBERT + PEFT-LoRA (see `FineTuned-DB-ToxicClassifier/`)
- Model: Fine-tuned DistilBERT with LoRA adapters
- Training script: `FineTuned-DB-ToxicClassifier/train.py`
- Key hyperparameters (see `training_args.bin` and `trainer_state.json`):
    - Learning rate: 5e-05 (linear scheduler)
    - Train batch size: 16
    - Eval batch size: 16
    - Epochs: 3
    - Optimizer: AdamW (betas=(0.9, 0.999), epsilon=1e-08, no additional args)
    - Seed: 42
    - Eval steps: 500
    - Max steps: 306
- Checkpoints saved at steps 102, 204, 306.
- Model artifacts: `adapter_model.safetensors`, `training_args.bin`, etc.

### B. Deep Learning Baseline (see `models/model.py`)
- Model: Custom deep learning classifier (e.g., CNN/LSTM with embeddings)
- Training script: `models/model.py`
- Model artifacts: `toxic_classifier.keras`, `toxic_classifier_v3.keras`
- Trained on the same dataset for direct comparison with DistilBERT.

## 3. Preliminary Observations

### A. Custom Deep Learning Model (Built from Scratch)

- The custom model architecture was designed as follows:

  ```python
  tf.keras.layers.Embedding(vocab_size, 128, input_length=max_len),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
  tf.keras.layers.GlobalMaxPooling1D(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(num_classes, activation='softmax')
  ```

  - The first layer is an embedding layer for input sequences.
  - The second layer is a bidirectional LSTM to capture sequence dependencies in both directions.
  - GlobalMaxPooling1D is applied to reduce the sequence output to a 1D vector for the dense layers.
  - Dropout layers are used to reduce overfitting and model complexity.
  - The final dense layer has 9 neurons (one for each class) with softmax activation for multi-class classification.

- **Performance:**
  - The model achieved strong results, with accuracy close to the fine-tuned DistilBERT base uncased model.
  - Example evaluation output:

    ```
    Classification Report:
                  precision    recall  f1-score   support

               2       1.00      0.91      0.95        11
               3       0.94      0.98      0.96        45
               6       0.33      0.25      0.29         4
               7       0.97      1.00      0.99        35
               8       1.00      0.86      0.92         7

        accuracy                           0.94       102
       macro avg       0.85      0.80      0.82       102
    weighted avg       0.94      0.94      0.94       102

    Confusion Matrix:
    [[10  0  0  1  0]
     [ 0 44  1  0  0]
     [ 0  3  1  0  0]
     [ 0  0  0 35  0]
     [ 0  0  1  0  6]]
    ```

- These results indicate the custom model is highly competitive, especially for the major classes, and can serve as a strong baseline for toxic comment classification.

---

### B. DistilBERT (with LoRA)

- For the transformer-based approach, I used the `distilbert-base-uncased` model as the base for fine-tuning, following the mentor's recommendation. I have previously fine-tuned this model for personal projects and found its performance and efficiency impressive.

- **Why DistilBERT?**
  - DistilBERT is a distilled version of BERT, offering a smaller model size and faster inference while retaining most of the performance of the original BERT. This makes it ideal for production and resource-constrained environments.

- **Fine-Tuning Approach:**
  - I leveraged the PEFT (Parameter-Efficient Fine-Tuning) library and LoRA (Low-Rank Adaptation) technique to fine-tune DistilBERT for the toxic comment classification task.
  - PEFT and LoRA are state-of-the-art methods for efficient and effective adaptation of large language models, enabling high performance with reduced computational cost and memory usage.
  - While I am still learning the nuances of all training hyperparameters, I focused on best practices and iteratively improved the configuration.

- **Training Results:**
  - The model was fine-tuned for 3 epochs. Key evaluation metrics from the final checkpoint:

    ```json
    {
        "epoch": 3.0,
        "eval_loss": 0.4127,
        "eval_runtime": 1.82,
        "eval_samples_per_second": 167.0,
        "eval_steps_per_second": 10.44
    }
    ```

  - These results demonstrate strong performance and efficient training, validating the effectiveness of the PEFT + LoRA approach on top of DistilBERT.

- **Artifacts and Evaluation:**
  - Model checkpoints and evaluation results are available in `FineTuned-DB-ToxicClassifier/model/` (`all_results.json`, `eval_results.json`).
  - The deep learning baseline models are saved in `models/` for direct comparison.

- **Next Steps:**
  - Review and compare evaluation metrics between the custom model and DistilBERT.
  - Tune hyperparameters further for potential improvements.
  - Prepare the best-performing model for deployment.

---


