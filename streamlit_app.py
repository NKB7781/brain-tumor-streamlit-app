import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# ------------------ Page Config ------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# ------------------ Load CNN Model ------------------
@st.cache_resource
def load_cnn_model():
    model_path = "custom_cnn_model.h5"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        return None
    
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {str(e)}")
        return None

model = load_cnn_model()

# ------------------ UI ------------------
st.title("🧠 Brain Tumor Classification - Custom CNN")
st.write("Upload a brain MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # ------------------ Preprocess ------------------
    if model:
        img = img.resize((224, 224))
        img_array = image._

