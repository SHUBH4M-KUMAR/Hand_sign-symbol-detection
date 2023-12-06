import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from tensorflow.keras.models import load_model as keras_load_model

def load_custom_model():
    # Get the absolute path to the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to the model within the repository
    model_relative_path = "after_5000_steps/ckpt-2"

    # Construct the full path to the model file in your GitHub repository
    model_path = os.path.join(script_dir, model_relative_path)

    # Load the model
    model = keras_load_model(model_path)


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
    model = load_custom_model()

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
