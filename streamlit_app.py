import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Function to load the checkpoint model
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

# Function to perform object detection using the loaded model
def detect_objects(model, image):
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image_tensor = tf.convert_to_tensor(image)

    # Perform inference
    detections = model(image_tensor)

    return detections

# Main function
def main():
    st.title("Object Detection with Checkpoint Model")

    # Load the checkpoint model
    model_path = "after_5000_steps"
    model = load_model(model_path)

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the uploaded image
        image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Perform object detection
        detections = detect_objects(model, image)

        # Visualize the detected objects on the image
        visualize_detection(image, detections)

# Function to visualize the detected objects on the image
def visualize_detection(image, detections):
    # Process detection results and draw bounding boxes
    # (Replace this with your actual post-processing logic)
    # For demonstration purposes, let's just draw a random bounding box
    h, w, _ = image.shape
    xmin = np.random.randint(0, w)
    ymin = np.random.randint(0, h)
    xmax = np.random.randint(xmin, w)
    ymax = np.random.randint(ymin, h)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display the image with detected objects
    st.image(image, channels="BGR")

if __name__ == "__main__":
    main()
