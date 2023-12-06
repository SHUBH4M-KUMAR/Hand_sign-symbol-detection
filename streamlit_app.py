import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your pre-trained model (replace this with your actual model loading code)
def load_model():
    model = tf.keras.models.load_model("path/to/your/model")
    return model

# Preprocess input image for the model
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Preprocess input video frames for the model
def preprocess_video_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = tf.keras.preprocessing.image.img_to_array(frame)
    frame = tf.expand_dims(frame, 0)
    return frame

# Perform hand sign detection on the input
def predict_hand_sign(input_data, model):
    # Add code to perform predictions using your model on the input data
    # This is a placeholder; replace it with your actual prediction logic
    prediction = model.predict(input_data)
    return prediction

# Main Streamlit app
def main():
    st.title("Hand Sign Detection with Streamlit")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "mp4"])

    # Load the pre-trained model
    model = load_model()

    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            # If the uploaded file is an image
            input_data = preprocess_image(uploaded_file)
            prediction = predict_hand_sign(input_data, model)
            st.image(uploaded_file, caption=f"Prediction: {prediction}", use_column_width=True)
        elif uploaded_file.type.startswith('video'):
            # If the uploaded file is a video
            cap = cv2.VideoCapture(uploaded_file)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                input_data = preprocess_video_frame(frame)
                prediction = predict_hand_sign(input_data, model)
                st.image(frame, caption=f"Prediction: {prediction}", use_column_width=True)
            cap.release()
        else:
            st.warning("Unsupported file format. Please upload an image or a video.")

if __name__ == "__main__":
    main()
