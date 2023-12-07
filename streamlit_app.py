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
import io
from collections import OrderedDict
import plotly.graph_objects as go

st.set_page_config(layout="wide")

CONFIG_PATH = r'C:\Users\shubh\OneDrive\Desktop\ML Models\Tensorflow\hand_sign_detection\pipeline.config'
CHECKPOINT_PATH = r'C:\Users\shubh\OneDrive\Desktop\ML Models\Tensorflow\hand_sign_detection\output'
LABEL_MAP_PATH = r'C:\Users\shubh\OneDrive\Desktop\ML Models\Tensorflow\hand_sign_detection\Train\hand-sign_label_map.pbtxt'

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

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

object_trackers = OrderedDict()


def detect_objects(image_np, threshold):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

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

    image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_np_with_detections)

    num_objects = len(detections['detection_boxes'])
    st.info(f"Number of Objects Detected: {num_objects}")

    return pil_image, num_objects


def generate_data():
    x = np.linspace(0, 1, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, size=100)
    return x, y


def update_chart(x, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Performance Trend'))
    st.plotly_chart(fig, use_container_width=True)


st.title("Hand Sign/Symbol Real-Time Detection")

st.sidebar.subheader("Menu")
menu_items = ["Home", "Object Detection", "Confusion Matrix", "About"]
choice = st.sidebar.selectbox("", menu_items)

if choice == "Home":
    st.subheader("Welcome to the Object Detection App")

    st.write(
        "This app allows you to perform real-time object detection using TensorFlow. "
        "Before you get started, please provide some information about your use case."
    )

   #st.markdown("### Detected Signs/Symbols:")
  # s.image(
     #  ["images/hand.png", "images/i_love_you.png", "images/namaste.png",
      #  "images/no.png", "images/thank_you.png", "images/yes.png"],
     #  width=100,
     #  caption=["Hand", "I Love You", "Namaste", "No", "Thank You", "Yes"],
  # )

    st.markdown("### Features:")
    st.write("- Real-time object detection.")
    st.write("- Supports webcam, images, and videos.")
    st.write("- Configurable options for threshold and more.")

    st.markdown("### Instructions:")
    st.write("1. Choose the 'Object Detection' option from the menu.")
    st.write("2. Select the input type: Webcam, Image, or Video.")
    st.write("3. Configure the parameters, such as the score threshold.")
    st.write("4. Click 'Start Object Detection' to see the model in action.")

    st.warning("To stop the webcam or video, press the 'q' key.")

    st.write(
        "Please provide relevant details about your use case to optimize the object detection "
        "performance for your specific scenario."
    )

    st.markdown("### Use Case Information:")

    # Create a form to gather user information
    form = st.form(key='user_input_form')

    # Add form fields for user input
    use_case_description = form.text_area(
        "Describe your use case:",
        "For example, are you detecting objects in images or analyzing a video stream?"
    )

    with st.expander("Image Details (if applicable)"):
        image_resolution = form.text_input("Image Resolution (e.g., 1920x1080):", "")
        image_format = form.text_input("Image Format (e.g., JPEG, PNG):", "")

    with st.expander("Video Details (if applicable)"):
        frame_rate = form.text_input("Frame Rate (frames per second):", "")
        video_duration = form.text_input("Video Duration (in seconds):", "")

    # Add a submit button to the form
    submit_button = form.form_submit_button('Submit')

    if submit_button:
        # Process the form data
        st.success("Form submitted successfully!")

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
            update_chart(x, y)
            with st.spinner("Detecting objects..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error reading from webcam.")
                        break
                    image, num_objects = detect_objects(frame, threshold)
                    image_placeholder.image(image, channels="RGB", use_column_width=True,
                                            caption="Model Performance")
                    progress_bar.progress(1.0)
                    x, y = generate_data()
                    update_chart(x, y)
                    time.sleep(0.1)

            cap.release()

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
                update_chart(x, y)
                with st.spinner("Detecting objects..."):
                    image, num_objects = detect_objects(img, threshold)
                    image_placeholder.image(image, channels="RGB", use_column_width=True,
                                            caption="Model Performance")
                    progress_bar.progress(1.0)
                    x, y = generate_data()
                    update_chart(x, y)
                    time.sleep(0.1)

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
        if uploaded_file is not None:
            threshold = st.slider("Score Threshold", 0.0, 1.0, 0.6, 0.05)
            start_detection = st.button("Start Object Detection")

            if start_detection:
                file_bytes = uploaded_file.read()
                temp_video_path = "temp_video.mp4"
                with open(temp_video_path, "wb") as temp_video:
                    temp_video.write(file_bytes)

                video = cv2.VideoCapture(temp_video_path)

                progress_bar = st.progress(0)
                image_placeholder = st.empty()
                x, y = generate_data()
                update_chart(x, y)
                with st.spinner("Detecting objects..."):
                    while True:
                        ret, frame = video.read()
                        if not ret:
                            st.error("Error reading from video.")
                            break
                        image, num_objects = detect_objects(frame, threshold)
                        image_placeholder.image(image, channels="RGB", use_column_width=True,
                                                caption="Model Performance")
                        progress_bar.progress(1.0)
                        x, y = generate_data()
                        update_chart(x, y)
                        time.sleep(0.1)

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
        df = pd.read_csv(uploaded_csv)
        true_labels = df['True Labels']
        predicted_labels = df['Predicted Labels']
        generate_confusion_matrix(true_labels, predicted_labels)


# Rest of the code...
elif choice == "About":
    st.subheader("About this App")

    st.write(
        "Welcome to the Object Detection App – where innovation meets practicality. "
        "This app is powered by a custom-trained object detection model, showcasing the prowess of MobileNetV2 architecture. "
        "Explore the details below to unravel the magic:"
    )

    st.markdown("### **Classes in the Custom Dataset:**")
    st.write(
        "The model is trained on a curated dataset with diverse classes, adding a personal touch to object detection. "
        "Here are some of the symbolic classes in the custom dataset:"
    )

    classes = {
        1: "Hand",
        2: "I Love You",
        3: "Namaste",
        4: "No",
        5: "Thank You",
        6: "Yes"
    }

    for class_id, class_name in classes.items():
        st.write(f"{class_id}. {class_name}")

    st.markdown("### **Model Architecture:**")
    st.write(
        "The neural power behind the app lies in the MobileNetV2 architecture – a lightweight yet potent structure. "
        "Built for real-time applications, MobileNetV2 ensures efficient object detection even on resource-constrained devices."
    )

    st.markdown("### **Model Creator:**")
    st.write(
        "This exceptional model and app are the brainchild of [Your Name]. For questions, feedback, or collaboration, "
        "feel free to connect using the following platforms:"
    )

    st.markdown("### **Connect with Me:**")
    st.write(
        "- **LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/in/yourlinkedinprofile)"
        "\n- **GitHub:** [Your GitHub Profile](https://github.com/yourusername)"
        "\n- **GitHub Repository:** [App Repository](https://github.com/yourusername/your-repo)"
        "\n- **Hugging Face:** [Your Hugging Face Profile](https://huggingface.co/yourusername)"
        "\n- **Email:** [your.email@example.com]"
    )

    st.success(
        "Ready to embark on the Object Detection adventure? Choose a destination from the menu and let the exploration begin!")
elif choice == "About":
    st.subheader("About this App")

    st.write(
        "Welcome to the Object Detection App – where innovation meets practicality. "
        "This app is powered by a custom-trained object detection model, showcasing the prowess of MobileNetV2 architecture. "
        "Explore the details below to unravel the magic:"
    )

    st.markdown("### **Classes in the Custom Dataset:**")
    st.write(
        "The model is trained on a curated dataset with diverse classes, adding a personal touch to object detection. "
        "Here are some of the symbolic classes in the custom dataset:"
    )

    classes = {
        1: "Hand",
        2: "I Love You",
        3: "Namaste",
        4: "No",
        5: "Thank You",
        6: "Yes"
    }

    for class_id, class_name in classes.items():
        st.write(f"{class_id}. {class_name}")

    st.markdown("### **Model Architecture:**")
    st.write(
        "The neural power behind the app lies in the MobileNetV2 architecture – a lightweight yet potent structure. "
        "Built for real-time applications, MobileNetV2 ensures efficient object detection even on resource-constrained devices."
    )

    st.markdown("### **Model Creator:**")
    st.write(
        "This exceptional model and app are the brainchild of [Your Name]. For questions, feedback, or collaboration, "
        "feel free to connect with [Your Email Address]. Your insights contribute to the continuous evolution of this project."
    )

    st.markdown("### **Tools and Libraries Used:**")
    st.write(
        "Explore the technological foundation that drives this app's seamless functionality. From TensorFlow Object Detection API "
        "to the efficiency of MobileNetV2, along with the user-friendly interfaces of Streamlit and the dynamic visualizations powered by Plotly."
    )

    st.success(
        "Ready to embark on the Object Detection adventure? Choose a destination from the menu and let the exploration begin!")
elif choice == "About":
    st.subheader("About this App")

    st.write(
        "This app uses a custom-trained object detection model based on the MobileNetV2 architecture. "
        "The model was trained using the TensorFlow Object Detection API on a custom dataset with the following classes:"
    )

    classes = {
        1: "Hand",
        2: "I Love You",
        3: "Namaste",
        4: "No",
        5: "Thank You",
        6: "Yes"
    }

    st.markdown("### Classes in the Custom Dataset:")
    for class_id, class_name in classes.items():
        st.write(f"{class_id}. {class_name}")

    st.markdown("### Model Architecture:")
    st.write(
        "The model is based on the MobileNetV2 architecture, which is known for its efficiency and effectiveness in "
        "real-time applications. MobileNetV2 is a lightweight neural network architecture that is well-suited for "
        "deploying on resource-constrained devices."
    )

    st.markdown("### Model Creator:")
    st.write(
        "The model and app were developed by [Your Name]. If you have any questions or feedback, please feel free to "
        "reach out to [Your Email Address]."
    )

    st.markdown("### Tools and Libraries Used:")
    st.write(
        "- TensorFlow Object Detection API"
        "\n- MobileNetV2"
        "\n- Streamlit"
        "\n- OpenCV"
        "\n- Plotly"
    )
elif choice == "About":
    st.subheader("About this App")

    st.write(
        "🚀 **Welcome to the Object Detection App!** Explore the magic of computer vision and real-time applications with a touch of creativity and innovation."
    )

    st.markdown("### **Visionary Model:**")
    st.write(
        "- At its core, the app employs a custom-trained object detection model based on the efficient MobileNetV2 architecture."
    )

    st.markdown("### **Custom Dataset Magic:**")
    st.write(
        "- The model is trained on a diverse dataset featuring symbols like 'Hand,' 'I Love You,' 'Namaste,' 'No,' 'Thank You,' and 'Yes.'"
    )

    st.markdown("### **Creative Minds Behind the Magic:**")
    st.write(
        "- Meet [Your Name], the visionary developer who brought this app to life. Connect with [Your Email Address] and share your thoughts!"
    )

    st.markdown("### **Instruments of Creation:**")
    st.write(
        "- Crafted with TensorFlow Object Detection API, MobileNetV2, Streamlit, OpenCV, and Plotly for a harmonious and delightful experience."
    )

    st.markdown("### **Interactive Exploration:**")
    st.write(
        "- Engage in real-time object detection across various media types and dynamic visualizations. It's about exploration and interaction."
    )

    st.markdown("### **Your Journey Begins:**")
    st.write(
        "- Embark on a discovery journey with the Object Detection App. Whether you're a tech enthusiast, a developer, or just curious about AI, there's something for everyone."
    )

    st.markdown("### **Developer's Playground:**")
    st.write(
        "- Curious about the code? Explore the [GitHub repository](https://github.com/yourusername) and contribute to the project's evolution."
    )

    st.markdown("### **Visual Appeal:**")
    st.image(
        "https://your-image-url.jpg",
        caption="Visual elements add depth to the experience.",
        use_column_width=True,
    )

    st.success("🌟 Ready to embark on the Object Detection adventure? Choose a destination from the menu and let the exploration begin!")

# Rest of the code...
elif choice == "About":
    st.subheader("About this App")

    st.write(
        "🚀 **Welcome to the object Detection App!** Explore the magic of computer vision and real-time applications with a touch of creativity and innovation."
    )

    st.markdown("### **Visionary Model:**")
    st.write(
        "- At its core, the app employs a custom-trained object detection model based on the efficient MobileNetV2 architecture."
    )

    st.markdown("### **Custom Dataset Magic:**")
    st.write(
        "- The model is trained on a diverse dataset featuring symbols like 'Hand,' 'I Love You,' 'Namaste,' 'No,' 'Thank You,' and 'Yes.'"
    )

    st.markdown("### **Creative Minds Behind the Magic:**")
    st.write(
        "- Meet [Your Name], the visionary developer who brought this app to life. Connect with [Your Email Address] and share your thoughts!"
    )

    st.markdown("### **Instruments of Creation:**")
    st.write(
        "- Crafted with TensorFlow Object Detection API, MobileNetV2, Streamlit, OpenCV, and Plotly for a harmonious and delightful experience."
    )

    st.markdown("### **Interactive Exploration:**")
    st.write(
        "- Engage in real-time object detection across various media types and dynamic visualizations. It's about exploration and interaction."
    )

    st.markdown("### **Your Journey Begins:**")
    st.write(
        "- Embark on a discovery journey with the Object Detection App. Whether you're a tech enthusiast, a developer, or just curious about AI, there's something for everyone."
    )

    st.markdown("### **Developer's Playground:**")
    st.write(
        "- Curious about the code? Explore the [GitHub repository](https://github.com/yourusername) and contribute to the project's evolution."
    )

    # Radar Chart Visualization
    radar_chart_data = {
        "Visionary Model": 4.5,
        "Custom Dataset Magic": 4.8,
        "Creative Minds": 4.2,
        "Instruments of Creation": 4.6,
        "Interactive Exploration": 4.7,
        "Journey Begins": 4.4,
        "Developer's Playground": 4.9,
    }

    radar_chart = go.Figure()
    radar_chart.add_trace(go.Scatterpolar(r=radar_chart_data.values(), theta=list(radar_chart_data.keys()), fill='toself'))
    radar_chart.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])))
    st.plotly_chart(radar_chart, use_container_width=True)

    st.success("🌟 Ready to embark on the Object Detection adventure? Choose a destination from the menu and let the exploration begin!")

# Rest of the code...
#lif choice == "About":
    st.subheader("About this App")

    st.write(
        "🚀 Welcome to the Object Detection App, a creation fueled by passion for computer vision and real-time applications!"
        "\n\n"
        "🎨 Developed with a blend of cutting-edge technology and user-friendly design, this app brings the power of "
        "object detection to your fingertips. Let's take a journey into the heart of this innovative creation."
    )

    st.markdown("### **Visionary Model:**")
    st.write(
        "At its core, this app harnesses the prowess of a custom-trained object detection model based on the MobileNetV2 architecture. "
        "This architectural marvel, known for its efficiency, enables real-time detection on a variety of platforms."
    )

    st.markdown("### **Custom Dataset Magic:**")
    st.write(
        "The model's intelligence is crafted through extensive training on a diverse dataset. We've curated a custom collection "
        "of symbols and signs, including 'Hand,' 'I Love You,' 'Namaste,' 'No,' 'Thank You,' and 'Yes.' Each symbol holds its "
        "unique significance, adding a touch of personalization to the detection experience."
    )

    st.markdown("### **Creative Minds Behind the Magic:**")
    st.write(
        "Meet the mind behind the magic – [Your Name]. A passionate developer, [Your Name] poured creativity and expertise into "
        "shaping this app. Feel free to connect and share your thoughts; [Your Email Address] is just a message away!"
    )

    st.markdown("### **Instruments of Creation:**")
    st.write(
        "This app is a symphony of tools and libraries, including TensorFlow Object Detection API, MobileNetV2, Streamlit, OpenCV, "
        "and Plotly. These instruments harmonize to deliver a seamless and delightful experience for users."
    )

    st.markdown("### **Interactive Exploration:**")
    st.write(
        "Engage with the app's features – from real-time object detection in various media types to dynamic visualizations. The "
        "experience is not just about detection; it's about exploration and interaction. Unleash the possibilities!"
    )

    st.markdown("### **Your Journey Begins:**")
    st.write(
        "Embark on a journey of discovery with the Object Detection App. Whether you're a tech enthusiast, a developer, or someone "
        "curious about the magic of AI, there's something here for everyone. Let's explore, create, and marvel together!"
    )

    st.markdown("### **Developer's Playground:**")
    st.write(
        "Curious about the code that powers this creation? Dive into the [GitHub repository](https://github.com/yourusername) "
        "to explore the intricacies and contribute to the evolution of this project."
    )

    st.success("Ready to embark on the Object Detection adventure? Choose a destination from the menu and let the exploration begin!")

# Rest of the code...
