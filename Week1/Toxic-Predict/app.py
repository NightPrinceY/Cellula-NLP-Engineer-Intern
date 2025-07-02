import gradio as gr
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load tokenizer and model
tokenizer_path = r"data/tokenizer.pkl"
model_path = r"models/toxic_classifier.keras"

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
model = tf.keras.models.load_model(model_path)

# Label map
label_map = {
    0: "Child Sexual Exploitation",
    1: "Elections",
    2: "Non-Violent Crimes",
    3: "Safe",
    4: "Sex-Related Crimes",
    5: "Suicide & Self-Harm",
    6: "Unknown S-Type",
    7: "Violent Crimes",
    8: "Unsafe"
}

def classify_toxic(query, image_desc):
    max_len = 150
    text = query + " " + image_desc
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(pad)
    pred_label = np.argmax(pred, axis=1)[0]
    return label_map.get(pred_label, "Unknown")

iface = gr.Interface(
    fn=classify_toxic,
    inputs=[
        gr.Textbox(label="Query"),
        gr.Textbox(label="Image Description")
    ],
    outputs=gr.Textbox(label="Predicted Toxic Category"),
    title="Toxic Category Classifier",
    description="Enter a query and image description to classify the prompt into one of the toxic categories"
)

if __name__ == "__main__":
    iface.launch()