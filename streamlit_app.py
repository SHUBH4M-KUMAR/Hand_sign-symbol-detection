import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Function to load the object detection model
def load_detection_model(config_path, checkpoint_path):
    configs = config_util.get_configs_from_pipeline_file(config_path)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    return detection_model

# Function to perform object detection on an image
@st.cache(allow_output_mutation=True)
def detect_objects(image, model):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections

# Streamlit app
def main():
    st.title("Object Detection with Streamlit")

    config_path = st.sidebar.text_input("Enter config file path:")
    checkpoint_path = st.sidebar.text_input("Enter checkpoint path:")
    
    detection_model = load_detection_model(config_path, checkpoint_path)

    cap = cv2.VideoCapture(0)

    while st.checkbox("Run Object Detection"):
        ret, frame = cap.read()
        image_np = np.array(frame)

        detections = detect_objects(image_np, detection_model)

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
            min_score_thresh=0.6,
            agnostic_mode=False
        )

        st.image(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB), channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
