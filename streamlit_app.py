import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

model_path = "best_resnet_model.h5"
if not os.path.exists(model_path):
    url = "https://drive.google.com/file/d/1EoL148o3_WQYt-eL2DxC7kopbZA6ZGGD/view?usp=drive_link"
    gdown.download(url, model_path, quiet=False)

# ⬇️ Load Models
@st.cache_resource
def load_models():
    cnn_model = load_model("custom_cnn_model.h5")
    resnet_model = load_model("best_resnet_model.h5")
    return cnn_model, resnet_model

cnn_model, resnet_model = load_models()

# Class labels (based on your dataset)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# ⬆️ Image Preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# 🎯 Prediction Function
def predict(img, model):
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# 📱 UI Layout
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload an MRI image and select a model to classify the tumor.")

# 📤 Upload Image
uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])

# 🧠 Select Model
model_choice = st.selectbox("Choose a Model", ["Custom CNN", "ResNet50 Fine-tuned"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        model = cnn_model if model_choice == "Custom CNN" else resnet_model
        label, confidence = predict(image_pil, model)
        st.success(f"Prediction: **{label.upper()}** with **{confidence:.2f}%** confidence")

