import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load Keras Model
@st.cache_resource
def load_food_model():
    model_path = "custom_cnn_image_classifier.keras"
    url = "https://drive.google.com/file/d/1ka-TzZ2ss4jZLZWToPcRVpBxtVltwmDO/view?usp=drive_linkk"
    gdown.download(url, model_path, quiet = False)
    
    return load_model(model_path)


model = load_food_model()

# Define class names and descriptions (modify as per your dataset)
class_info = {
    0: {'name': 'Jollof Rice', 'description': 'Jollof rice is a flavorful West African dish made by cooking rice in a rich tomato-based sauce with onions, peppers, and a blend of spices. Often prepared with added ingredients like vegetables, chicken, beef, or fish, it is a vibrant one-pot meal known for its smoky aroma and bold taste. Jollof rice is both nutritious and satisfying, offering more vitamins and minerals than plain white rice.'},
    1: {'name': 'White Rice', 'description': 'White rice is a refined grain with the husk, bran, and germ removed during processing, giving it a soft texture and longer shelf life. It is a versatile staple food with a mild flavor, often served as a side dish or base for various meals. While low in fiber and nutrients compared to whole grains, it provides a quick energy source due to its high carbohydrate content.'},
}

st.title("Food Image Classifier 🍽️")
st.write("Upload an image of food, and the model will predict its category (White or Jellof Rice Detection)")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    img = img.resize((150, 150))  # Adjust size based on model input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Make Prediction
    predictions = model.predict(img_array)

    # Check if binary or multi-class classification
    if predictions.shape[1] == 1:  # Binary classification
        predicted_classes = (predictions > 0.5).astype('int32').flatten()
    else:  # Multi-class classification
        predicted_classes = np.argmax(predictions, axis=1)

    predicted_class = predicted_classes[0]

    # Retrieve food name and description
    if predicted_class in class_info:
        food_name = class_info[predicted_class]['name']
        food_description = class_info[predicted_class]['description']
        st.write(f"### 🍲 Prediction: {food_name}")
        st.write(f"📖 {food_description}")
    else:
        st.write("⚠️ Unknown food detected. The model is trained to detect only White/Jello Rice")
