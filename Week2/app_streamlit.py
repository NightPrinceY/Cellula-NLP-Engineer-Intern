
import streamlit as st
from PIL import Image
from pipeline.blip_caption import caption_image
from pipeline.llama_guard import llama_guard_filter
from pipeline.toxic_classifier import classify_toxicity

# 🌐 Streamlit UI
st.set_page_config(page_title="Toxic Moderation System", layout="centered")
st.title("🛡️ Dual-Stage Toxic Moderation")
st.markdown("Moderate text and images using **Llama Guard** + **DistilBERT-LoRA**.\n\n- Stage 1: Hard Safety Filter (Llama Guard)\n- Stage 2: Fine Toxic Classifier (LoRA DistilBERT)")


text_input = st.text_area("✏️ Enter a text message", height=150)
uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

image_caption = ""
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("🔍 Generating caption with BLIP..."):
        image_caption = caption_image(image)
    st.success(f"📝 Caption: `{image_caption}`")

if st.button("🚀 Run Moderation"):
    full_text = text_input + " [SEP] " + image_caption
    with st.spinner("🛡️ Stage 1: Llama Guard..."):
        safety = llama_guard_filter(full_text)

    if safety == "unsafe":
        st.error("❌ Llama Guard flagged this content as **UNSAFE**.\nModeration stopped.")
    elif safety == "safe":
        st.success("✅ Safe by Llama Guard. Proceeding to classifier...")
        with st.spinner("🧠 Stage 2: DistilBERT Toxic Classifier..."):
            label, score, scores = classify_toxicity(text_input, image_caption)
        st.markdown(f"### 🔍 Prediction: `{label}` ({round(score*100, 2)}%)")
        st.text("📊 Class Probabilities:\n" + scores)
    else:
        st.warning(f"Llama Guard API returned: {safety}. Proceed with caution.")

#All Thanks to Cellula for the opportunity