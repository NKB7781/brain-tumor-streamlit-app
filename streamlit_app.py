import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

file_id = "1EoL148o3_WQYt-eL2DxC7kopbZA6ZGGD"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "best_resnet_model.h5", quiet=False)

#Load Models
@st.cache_resource(allow_output_mutation=True)
def load_models():
    custom_model = load_model("custom_cnn_model.h5")
    resnet_model = load_model("best_resnet_model.h5")
    return custom_model, resnet_model

custom_model, resnet_model = load_models()
 
#Define Prediction Function
def predict_tumor(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    return class_labels[class_index], prediction[0][class_index]

#Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
model_choice = st.radio("Select Model to Use", ("Custom CNN", "ResNet50 Fine-Tuned"))

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        if model_choice == "Custom CNN":
            label, confidence = predict_tumor(img, custom_model)
        else:
            label, confidence = predict_tumor(img, resnet_model)

        st.success(f"Prediction: **{label.upper()}** with confidence {confidence:.2f}")
