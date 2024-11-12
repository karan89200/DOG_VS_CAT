import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load the pre-trained model
model = load_model('cat_dog_classifier.h5')

# Title of the Streamlit App
st.title("Cat vs Dog Image Classifier")
st.write("Upload an image of a cat or dog, and the model will classify it.")

# Image uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process and display the uploaded image
if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert image to RGB format if it's in grayscale or other format
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize the image to 256x256 pixels and normalize it
    img_array = np.array(image.resize((256, 256))) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # Create batch dimension

    # Make prediction
    prediction = model.predict(img_batch)[0][0]
    predicted_label = "Dog" if prediction > 0.5 else "Cat"

    # Display the result
    st.write(f"Prediction: {predicted_label}")
    st.write(f"Confidence Score: {prediction:.2f}")
