


# 🛡️ Week 2: Dual-Stage Toxic Moderation App

Welcome to the **production-ready, research-driven Streamlit app** for multi-modal toxic content moderation, developed during the Cellula AI Internship. This system combines state-of-the-art models for both text and image moderation, with a robust, modular pipeline and professional documentation.

---

## 🚀 Quick Links

- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/NightPrince/Dual-Stage-Toxic-Moderation) — Try the app instantly in your browser.
- **Source Code:** [GitHub Repo](https://github.com/NightPrinceY/https-huggingface.co-spaces-NightPrince-Dual-Stage-Toxic-Moderation)
- **Week 1 Foundation:** [Week1/README.md](../Week1/Toxic-Predict/README.md) — Data pipeline, model training, and benchmarking.

---

## 📦 Project Overview

This app implements a **dual-stage moderation pipeline** for both text and images, designed for real-world safety, transparency, and extensibility:

### 1. Stage 1: Hard Filter (Llama Guard)
- **Llama Guard API (OpenRouter):** Instantly blocks content that is legally or ethically unsafe (e.g., violence, hate, sexual exploitation).
- **Strict Prompting:** Ensures only 'safe' or 'unsafe' is returned for maximum reliability.
- **Immediate Feedback:** If unsafe, the user is notified and moderation stops.

### 2. Stage 2: Fine-Grained Classifier (DistilBERT+LoRA)
- **DistilBERT (PEFT-LoRA):** Fine-tuned for nuanced, multi-class toxic content classification (9 categories).
- **Transparent Output:** Shows predicted category and class probabilities for user trust.
- **Class Imbalance Addressed:** SMOTE, class weights, and oversampling used for robust results.

### 3. Image Support (BLIP)
- **BLIP (Salesforce):** Generates captions for uploaded images, enabling moderation of visual content via the same pipeline.
- **Unified Moderation:** Caption is appended to text and passed through both moderation stages.

---

## 🏗️ How to Run Locally

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **API Key:**
   - Copy `.env.example` to `.env` and add your OpenRouter API key (unless hardcoded).
3. **Model Files:**
   - Place the fine-tuned model in:
     `C:/Users/NightPrince/OneDrive/Desktop/Cellula-Internship/Week1/peft-distilbert-toxic-classifier/last-checkpoint/`
4. **Run the app:**
   ```sh
   streamlit run app_streamlit.py
   ```

---

## 🧠 Week 2 Highlights & Achievements

- **Modular Pipeline:** All moderation logic is split into clear modules (`pipeline/`), making the codebase maintainable and extensible.
- **Production-Ready UI:** Streamlit app with robust error handling, clear feedback, and support for both text and images.
- **Advanced Model Selection:** Benchmarked PEFT-LoRA DistilBERT vs. CNN/LSTM; addressed class imbalance for real-world reliability.
- **Comprehensive Documentation:** Professional README, detailed HTML report, and in-code docstrings for full reproducibility.

---

## 📝 File Structure

```
Week2/
├── app_streamlit.py         # Main Streamlit app (uses modular pipeline)
├── pipeline/                # Modular pipeline: blip_caption.py, llama_guard.py, toxic_classifier.py
├── requirements.txt         # Python dependencies (transformers, streamlit, peft, torch, etc.)
├── .env.example             # Example environment file for API keys
├── README.md                # This file
├── internship_week2_report.html # Detailed, professional report
```

---

## 🔬 Model Details

### Stage 1: Llama Guard (OpenRouter)
- [Llama Guard](https://llama.meta.com/llama-guard/) — Meta's safety classifier, accessed via OpenRouter API.
- Used as a hard filter to block unsafe content before further analysis.

### Stage 2: DistilBERT + LoRA (Hugging Face)
- [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) — Lightweight transformer model.
- [LoRA (Low-Rank Adaptation)](https://huggingface.co/docs/peft/index) — Efficient fine-tuning via PEFT.
- Fine-tuned on the internship dataset for 9-class toxic content classification.
- [Model weights and config](https://huggingface.co/NightPrince/peft-distilbert-toxic-classifier) available on Hugging Face.

### Image Captioning: BLIP (Salesforce)
- [BLIP](https://github.com/salesforce/BLIP) — Vision-language model for generating captions from images.
- Used to convert uploaded images into text for moderation.

---

## 📊 Results & Reporting

- See [Week1/README.md](../Week1/Toxic-Predict/README.md) for full benchmarking, class imbalance analysis, and model selection rationale.
- Week 2 focused on robust deployment, dual-stage logic, and user experience improvements.

---

## 🙏 Acknowledgements

- Thanks to **Cellula AI** and my mentor for guidance, support, and the project proposal.
- Built by Yahya Muhammad Alnwsany ([Portfolio](https://nightprincey.github.io/Portfolio/))

---

**For more details, see the [Week 1 README](../Week1/Toxic-Predict/README.md), the [Hugging Face Space](https://huggingface.co/spaces/NightPrince/Dual-Stage-Toxic-Moderation), and the detailed [internship_week2_report.html](./internship_week2_report.html).**
