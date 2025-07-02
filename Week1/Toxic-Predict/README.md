# Toxic-Predict

**Internship Project – Week 1: Text Moderation Foundation**

Toxic-Predict is a machine learning project developed as part of the Cellula AI Internship, focused on building a safe and responsible multi-modal toxic content moderation system. This repository documents the **first phase** of the project: a robust text moderation pipeline, which will be extended to a dual-stage, multi-modal system in subsequent weeks.

---

## 🚩 Project Context

This project is part of the **Cellula AI Internship** proposal:  
**"Safe and Responsible Multi-Modal Toxic Content Moderation"**

The overall goal is to build a dual-stage, multi-modal moderation system for both text and images, combining state-of-the-art NLP and vision models. The final system will include:

- **Stage 1:** Hard moderation filter (Llama Guard) to block legally or ethically disallowed content.
- **Stage 2:** Soft classifier (DistilBERT fine-tuned with PEFT-LoRA or Deep Learning Classifier) for nuanced, multi-class toxic content classification.
- **Image Extension:** Captioning images with BLIP and applying the same moderation pipeline to captions.
- **Deployment:** Streamlit web app for demonstration and real-world use.

This README documents the **Week 1 deliverables**: the text data pipeline, deep learning and transformer-based models, and benchmarking results.

---

## Project Structure

```
.
├── app.py
├── run.py
├── test.py
├── requirements.txt
├── README.md
├── data/
│   ├── cellula-toxic.csv
│   ├── cleaned.csv
│   ├── eval.csv
│   ├── test.csv
│   ├── tokenizer.pkl
│   └── train.csv
├── models/
│   ├── model.py
│   ├── toxic_classifier.h5
│   └── toxic_classifier.keras
├── notebooks/
│   ├── Preprocessing.ipynb
│   └── tokenization.ipynb
└── src/
    ├── preprocess.py
    └── tokenize_and_split.py
```

---

## ✨ Features

- Modular data pipeline: cleaning, tokenization, label encoding, stratified splitting
- Deep learning baseline (Keras/TensorFlow, Bidirectional LSTM)
- Transformer-based classifier (DistilBERT + PEFT-LoRA)
- Benchmarking and comparative evaluation
- All artifacts and code versioned for reproducibility
- Ready for extension to dual-stage and multi-modal moderation

---

---

## 📊 Data Pipeline Overview

**Dataset:** `data/cellula-toxic.csv` (multi-class labeled user comments)

**Pipeline Steps:**
- **Loading:** Used pandas to load and inspect the dataset.
- **Cleaning:** Lowercased text, removed emojis/special characters/HTML entities, whitespace normalization, optional stopword removal and lemmatization (NLTK, spaCy).
- **Tokenization:** Custom tokenizer (HuggingFace or Keras) fitted on cleaned corpus, saved as `tokenizer.pkl` and `tokenizer.json`.
- **Label Encoding:** Used `data/label_map.json` for consistent multi-class mapping.
- **Splitting:** Stratified train/val/test split (60/20/20) using sklearn, maintaining class balance.
- **Artifacts:** All processed data and tokenizer artifacts saved for reproducibility.

---

## 📂 Data

- CSV files with columns: `query`, `image descriptions`, `Toxic Category`, and `Toxic Category Encoded`.
- Data splits: `train.csv`, `eval.csv`, `test.csv`, and `cleaned.csv` for processed data.
- 9 categories: Safe, Violent Crimes, Elections, Sex-Related Crimes, Unsafe, Non-Violent Crimes, Child Sexual Exploitation, Unknown S-Type, Suicide & Self-Harm.

---

## 🤖 Modeling Approaches

### 1. Deep Learning Baseline (Bidirectional LSTM)

**Why LSTM?**

- LSTMs (Long Short-Term Memory networks) are a type of RNN designed to capture long-range dependencies in sequential data. They are effective for text classification because they can model the order and context of words, which is crucial for detecting nuanced toxicity.
- Bidirectional LSTM allows the model to consider both past and future context in a sentence.
- LSTMs are less computationally intensive than transformers and are easy to interpret and debug.

**Architecture:**
```python
Embedding(vocab_size, 128, input_length=max_len),
Bidirectional(LSTM(64, return_sequences=True)),
GlobalMaxPooling1D(),
Dropout(0.3),
Dense(64, activation='relu'),
Dropout(0.3),
Dense(num_classes, activation='softmax')
```

**Performance:**
- Accuracy: 94%
- Macro F1: 82%, Weighted F1: 94%

**Sample Classification Report:**
```
      precision    recall  f1-score   support
2       1.00        0.91      0.95       11
3       0.94        0.98      0.96       45
6       0.33        0.25      0.29        4
7       0.97        1.00      0.99       35
8       1.00        0.86      0.92        7
Accuracy: 0.94 (n=102)
```

### 2. Transformer-Based Model: DistilBERT + PEFT (LoRA)

**What is DistilBERT?**

- [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) is a distilled (compressed) version of BERT, offering nearly the same performance as BERT but with fewer parameters and faster inference.

**What is PEFT?**

- **Parameter-Efficient Fine-Tuning (PEFT)** is a family of techniques that allow large pre-trained models to be adapted to new tasks by training only a small subset of parameters, rather than the entire model. This is especially important for deploying transformer models in production, where memory and compute resources may be limited.

**What is LoRA?**

- **LoRA (Low-Rank Adaptation)** is a PEFT method that injects small, trainable low-rank matrices into each layer of a transformer model. Instead of updating all the weights in the model, LoRA only updates these additional matrices, drastically reducing the number of trainable parameters.
- **Benefits:**
    - Efficiency: Less memory and compute required
    - Speed: Faster training and inference
    - Performance: Comparable to full fine-tuning
    - Modularity: Adapters can be swapped in/out for experimentation

**Why use PEFT/LoRA?**

- The toxic comment classification task benefits from the language understanding of large models like DistilBERT, but full fine-tuning is resource-intensive. LoRA enables efficient adaptation of DistilBERT to our specific dataset, making it practical to deploy high-performing models even with limited resources.

**Training Details:**
- Base Model: [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- Fine-Tuning Method: LoRA adapters via PEFT ([Hugging Face PEFT Docs](https://huggingface.co/docs/peft/index))
- Epochs: 3
- Learning Rate: 5e-5
- Optimizer: AdamW

**Results:**
```
"epoch": 3.0,
"eval_loss": 0.4127,
"eval_runtime": 1.82,
"eval_samples_per_second": 167.0,
"eval_steps_per_second": 10.44
```

**Artifacts:**
- [peft-distilbert-toxic-classifier (Hugging Face)](https://huggingface.co/NightPrince/peft-distilbert-toxic-classifier)
- [peft-distilbert-toxic-classifier (GitHub)](https://github.com/NightPrinceY/peft-distilbert-toxic-classifier)

---

## 📈 Comparative Summary

| Aspect           | LSTM Baseline      | DistilBERT + LoRA         |
|------------------|-------------------|---------------------------|
| Performance      | 94% Accuracy      | Eval Loss: 0.41           |
| Training Cost    | Low               | Medium                    |
| Inference Speed  | High              | Medium                    |
| Flexibility      | Good for Edge     | Better for NLP Stack      |
| Next Steps       | Hyperparam Tuning | RoBERTa / DeBERTa PEFT    |

---

## 📦 Artifacts & Resources

- All model weights, tokenizer files, and evaluation results are versioned in the repo.
- **Baseline model (from scratch):**
    - [Hugging Face: NightPrince/Toxic_Classification](https://huggingface.co/NightPrince/Toxic_Classification)
    - [GitHub: Toxic-classificatrion](https://github.com/NightPrinceY/Toxic-classificatrion/tree/main)
- **DistilBERT + LoRA model:**
    - [Hugging Face: NightPrince/peft-distilbert-toxic-classifier](https://huggingface.co/NightPrince/peft-distilbert-toxic-classifier)
    - [GitHub: peft-distilbert-toxic-classifier](https://github.com/NightPrinceY/peft-distilbert-toxic-classifier)

---

## 🤗 Hugging Face Inference

This model is available on the Hugging Face Hub:

- [NightPrince/Toxic_Classification (baseline LSTM)](https://huggingface.co/NightPrince/Toxic_Classification)
- [NightPrince/peft-distilbert-toxic-classifier (DistilBERT+LoRA)](https://huggingface.co/NightPrince/peft-distilbert-toxic-classifier)

### Inference API Usage

You can use the Hugging Face Inference API or widget with two fields:

- `text`: The main query or post text
- `image_desc`: The image description (if any)

**Example (Python):**
```python
from huggingface_hub import InferenceClient
client = InferenceClient("NightPrince/Toxic_Classification")
result = client.text_classification({
    "text": "This is a dangerous post",
    "image_desc": "Knife shown in the image"
})
print(result)  # {'label': 'toxic', 'score': 0.98}
```

### Custom Pipeline Details

- The model uses a custom `pipeline.py` for multi-input inference.
- The output is a dictionary with the predicted `label` (class name) and `score` (confidence).
- Class names are mapped using `label_map.json`.

**Files in the repo:**
- `pipeline.py` (custom inference logic)
- `tokenizer.json` (Keras tokenizer)
- `label_map.json` (class code to name mapping)
- TensorFlow SavedModel files (`saved_model.pb`, `variables/`)

**Requirements:**
```
tensorflow
keras
numpy
```

---

## 📚 Resources & References

- [Cellula Internship Project Proposal](#)
- [BLIP: Bootstrapped Language-Image Pre-training](https://github.com/salesforce/BLIP)
- [Llama Guard](https://llama.meta.com/llama-guard/)
- [DistilBERT (Hugging Face)](https://huggingface.co/distilbert/distilbert-base-uncased)
- [DistilBERT + LoRA (Hugging Face)](https://huggingface.co/NightPrince/peft-distilbert-toxic-classifier)
- [DistilBERT + LoRA (GitHub)](https://github.com/NightPrinceY/peft-distilbert-toxic-classifier)
- [Baseline LSTM Model (Hugging Face)](https://huggingface.co/NightPrince/Toxic_Classification)
- [Baseline LSTM Model (GitHub)](https://github.com/NightPrinceY/Toxic-classificatrion/tree/main)
- [Streamlit](https://streamlit.io/)

---

**Author:** Yahya Muhammad Alnwsany  
**Contact:** yahyaalnwsany39@gmail.com  
**Portfolio:** https://nightprincey.github.io/Portfolio/
