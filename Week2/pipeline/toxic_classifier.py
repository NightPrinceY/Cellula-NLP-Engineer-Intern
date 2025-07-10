import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel
import os

id2label = {
    0: "Child Sexual Exploitation",
    1: "Elections",
    2: "Non-Violent Crimes",
    3: "Safe",
    4: "Sex-Related Crimes",
    5: "Suicide & Self-Harm",
    6: "Unknown S-Type",
    7: "Violent Crimes",
    8: "unsafe"
}

MODEL_DIR = r"C:\\Users\\NightPrince\\OneDrive\\Desktop\\Cellula-Internship\\Week1\\peft-distilbert-toxic-classifier\\last-checkpoint"


def load_toxic_classifier():
    base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=9)
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    try:
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        return pipe
    except Exception:
        def manual_pipe(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.logits.softmax(dim=-1).squeeze().tolist()
            return [[{"label": str(i), "score": float(score)} for i, score in enumerate(scores)]]
        return manual_pipe

def classify_toxicity(text_input, caption):
    pipe = load_toxic_classifier()
    full_input = text_input + " [SEP] " + caption
    preds = pipe(full_input)
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
        preds = preds[0]
    if not isinstance(preds, list) or len(preds) == 0 or not isinstance(preds[0], dict):
        return "Unknown", 0.0, "No prediction"
    preds_sorted = sorted(preds, key=lambda x: x.get('score', 0), reverse=True)
    top_label = preds_sorted[0].get('label', '0')
    top_score = preds_sorted[0].get('score', 0.0)
    label_id = int(top_label.split("_")[-1]) if "_" in top_label else int(top_label)
    final_label = id2label.get(label_id, "Unknown")
    scores_table = "\n".join(
        [f"{id2label.get(int(item.get('label', '0').split('_')[-1]), 'Unknown')}: {round(float(item.get('score', 0))*100, 2)}%" for item in preds]
    )
    return final_label, top_score, scores_table
