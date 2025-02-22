%%writefile app.py

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# Define the same model architecture (must match the saved model)
class FoodDetectionCNN(nn.Module):
    def __init__(self):
        super(FoodDetectionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 512),  # Adjust shape based on input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),  # Binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x




# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),  # Resize to model's input size
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to match training data
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension
def predict(image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.squeeze().tolist()
import streamlit as st
from PIL import Image




# Load PyTorch Model
@st.cache_resource
def load_food_model():
# Load model
  model = FoodDetectionCNN()
  model.load_state_dict(torch.load("/content/modelnew.pth", map_location=torch.device("cpu")))
  model.eval()
  return model

model = load_food_model()

# Define class names and descriptions (modify as per your dataset)
class_info = {
    0: {'name': 'Jollof Rice', 'description': 'Jollof rice is a flavorful West African dish made by cooking rice in a rich tomato-based sauce with onions, peppers, and a blend of spices. Often prepared with added ingredients like vegetables, chicken, beef, or fish, it is a vibrant one-pot meal known for its smoky aroma and bold taste. Jollof rice is both nutritious and satisfying, offering more vitamins and minerals than plain white rice.'},
    1: {'name': 'White Rice', 'description': 'White rice is a refined grain with the husk, bran, and germ removed during processing, giving it a soft texture and longer shelf life. It is a versatile staple food with a mild flavor, often served as a side dish or base for various meals. While low in fiber and nutrients compared to whole grains, it provides a quick energy source due to its high carbohydrate content.'},
}

st.title("Food Image Classifier 🍽️")
st.write("Upload an image of food, and the model will predict its category (White or Jollof Rice Detection)")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_tensor = preprocess_image(img)

    # Make Prediction
    with torch.no_grad():
        predictions = model(img_tensor)
        
    # Check if binary or multi-class classification
    if predictions.shape[1] == 1:  # Binary classification
        predicted_classes = (predictions > 0.5).int().flatten()
    else:  # Multi-class classification
        predicted_classes = torch.argmax(predictions, dim=1)

    predicted_class = predicted_classes.item()

    # Retrieve food name and description
    if predicted_class in class_info:
        food_name = class_info[predicted_class]['name']
        food_description = class_info[predicted_class]['description']
        st.write(f"### 🍲 Prediction: {food_name}")
        st.write(f"📖 {food_description}")
    else:
        st.write("⚠️ Unknown food detected. The model is trained to detect only White/Jollof Rice")
