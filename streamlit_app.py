import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import io  # Added for video upload support

st.set_page_config(layout="wide")  # Set layout to wide

# Add the path to your pipeline.config and checkpoint folder
CONFIG_PATH = r'C:\Users\shubh\OneDrive\Desktop\ML Models\Tensorflow\hand_sign_detection\pipeline.config'
CHECKPOINT_PATH = r'C:\Users\shubh\OneDrive\Desktop\ML Models\Tensorflow\hand_sign_detection\output'

# Add the path to your label map file
LABEL_MAP_PATH = r'C:\Users\shubh\OneDrive\Desktop\ML Models\Tensorflow\hand_sign_detection\Train\hand-sign_label_map.pbtxt'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
try:
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)

def detect_objects(image_np, threshold):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=threshold,
        agnostic_mode=False)

    # Convert color space to RGB
    image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(image_np_with_detections)

    return pil_image

def generate_data():
    # Simulate an idealized performance trend for x
    x = np.linspace(0, 1, 100)  # Values from 0 to 1 representing time
    # Simulate a more realistic performance trend
    y = np.sin(x) + np.random.normal(0, 0.1, size=100)

    return x, y

def generate_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=category_index.keys())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=category_index.keys(),
                yticklabels=category_index.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot()

def main():
    st.title("Object Detection with Streamlit")

    # Sidebar with menu items
    st.sidebar.subheader("Menu")
    menu_items = ["Home", "Object Detection", "Confusion Matrix", "About"]
    choice = st.sidebar.selectbox("", menu_items)

    if choice == "Home":
        st.subheader("Welcome to the Object Detection App")
        st.write(
            "This app allows you to perform real-time object detection using TensorFlow."
        )
        st.markdown("### Features:")
        st.write("- Real-time object detection.")
        st.write("- Supports webcam, images, and videos.")
        st.write("- Configurable options for threshold and more.")
    elif choice == "Object Detection":
        st.subheader("Object Detection Configuration")

        input_types = ["Select", "Webcam", "Image", "Video"]
        input_type = st.selectbox("Select Input Type", input_types, index=0)

        if input_type == "Select":
            st.warning("Please select an input type from the dropdown.")
        elif input_type == "Webcam":
            st.warning("To stop the webcam, press 'q' key.")
            threshold = st.slider("Score Threshold", 0.0, 1.0, 0.6, 0.05)
            start_detection = st.button("Start Object Detection")

            if start_detection:
                cap = cv2.VideoCapture(0)
                progress_bar = st.progress(0)
                image_placeholder = st.empty()
                x, y = generate_data()
                line_chart = st.line_chart(data={'x': x, 'y': y})
                with st.spinner("Detecting objects..."):
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Error reading from webcam.")
                            break
                        image = detect_objects(frame, threshold)
                        image_placeholder.image(image, channels="RGB", use_column_width=True, caption="Model Performance")
                        # Update progress bar
                        progress_bar.progress(1.0)
                        # Update chart data
                        x, y = generate_data()
                        line_chart.line_chart(data={'x': x, 'y': y})
                        # Sleep for a short duration to simulate processing time
                        time.sleep(0.1)
        elif input_type == "Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                threshold = st.slider("Score Threshold", 0.0, 1.0, 0.6, 0.05)
                start_detection = st.button("Start Object Detection")

                if start_detection:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    progress_bar = st.progress(0)
                    image_placeholder = st.empty()
                    x, y = generate_data()
                    line_chart = st.line_chart(data={'x': x, 'y': y})
                    with st.spinner("Detecting objects..."):
                        image = detect_objects(img, threshold)
                        image_placeholder.image(image, channels="RGB", use_column_width=True, caption="Model Performance")
                        # Update progress bar
                        progress_bar.progress(1.0)
                        # Update chart data
                        x, y = generate_data()
                        line_chart.line_chart(data={'x': x, 'y': y})
                        # Sleep for a short duration to simulate processing time
                        time.sleep(0.1)
        elif input_type == "Video":
            uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
            if uploaded_file is not None:
                threshold = st.slider("Score Threshold", 0.0, 1.0, 0.6, 0.05)
                start_detection = st.button("Start Object Detection")

                if start_detection:
                    # Use BytesIO to read video file as bytes
                    file_bytes = uploaded_file.read()

                    # Create a temporary file to store the video
                    temp_video_path = "temp_video.mp4"
                    with open(temp_video_path, "wb") as temp_video:
                        temp_video.write(file_bytes)

                    # Open the temporary video file
                    video = cv2.VideoCapture(temp_video_path)

                    progress_bar = st.progress(0)
                    image_placeholder = st.empty()
                    x, y = generate_data()
                    line_chart = st.line_chart(data={'x': x, 'y': y})
                    with st.spinner("Detecting objects..."):
                        while True:
                            ret, frame = video.read()
                            if not ret:
                                st.error("Error reading from video.")
                                break
                            image = detect_objects(frame, threshold)
                            image_placeholder.image(image, channels="RGB", use_column_width=True,
                                                    caption="Model Performance")
                            # Update progress bar
                            progress_bar.progress(1.0)
                            # Update chart data
                            x, y = generate_data()
                            line_chart.line_chart(data={'x': x, 'y': y})
                            # Sleep for a short duration to simulate processing time
                            time.sleep(0.1)

                    # Release the video file and remove the temporary file
                    video.release()
                    os.remove(temp_video_path)

        else:
            st.warning("Please select a valid input type from the dropdown.")
    elif choice == "Confusion Matrix":
        st.subheader("Confusion Matrix")
        st.warning("To generate the confusion matrix, upload a CSV file containing true and predicted labels.")
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
        generate_matrix = st.button("Generate Confusion Matrix")

        if generate_matrix and uploaded_csv is not None:
            # Assume CSV columns: 'True Labels', 'Predicted Labels'
            df = pd.read_csv(uploaded_csv)
            true_labels = df['True Labels']
            predicted_labels = df['Predicted Labels']
            generate_confusion_matrix(true_labels, predicted_labels)
    elif choice == "About":
        st.subheader("About this App")
        st.write(
            "This app is developed using Streamlit and TensorFlow for real-time object detection."
        )
        st.markdown(
            "#### Tools and Libraries Used:"
            "\n- Streamlit"
            "\n- TensorFlow"
            "\n- OpenCV"
            "\n- scikit-learn"
            "\n- Matplotlib"
            "\n- Seaborn"
        )
        st.markdown("#### Developer:")
        st.write("Your Name")
        st.write("[GitHub Profile](https://github.com/yourusername)")
    else:
        st.warning("Please select a valid option from the menu.")

if __name__ == "__main__":
    main()
