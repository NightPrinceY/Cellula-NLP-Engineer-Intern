from transformers import BlipProcessor, BlipForConditionalGeneration

import streamlit as st

def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Only cache the model/processor, not the image input
@st.cache_resource
def get_blip_model():
    return load_caption_model()

def caption_image(_img):
    # _img: leading underscore disables Streamlit's hashing for this argument
    processor, model = get_blip_model()
    inputs = processor(images=_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    out = model.generate(pixel_values=pixel_values)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
