# app.py
import streamlit as st
import cv2
from PIL import Image
import numpy as np

def main():
    st.title("Hand Sign Detection with Streamlit")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "mp4"])

    if uploaded_file is not None:
        # Display the uploaded image or video
        st.image(uploaded_file, caption="Uploaded Image/Video.", use_column_width=True)

        # Add code for hand sign detection on the uploaded file
        detect_hand_signs(uploaded_file)

def detect_hand_signs(uploaded_file):
    # Add code for hand sign detection on the uploaded file
    pass

if __name__ == "__main__":
    main()

