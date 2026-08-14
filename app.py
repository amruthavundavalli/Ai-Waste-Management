import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Waste Management",
    page_icon="♻️",
    layout="centered"
)

# -----------------------------
# Load YOLO model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("♻️ AI Waste Management")
st.write("Upload a waste image and the AI model will detect objects.")

st.divider()

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Select Waste Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Detection
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Waste", use_container_width=True):

        with st.spinner("Detecting..."):

            # Convert PIL image to numpy
            image_array = np.array(image)

            # Run YOLO
            results = model(image_array)

            # Get annotated image
            result_image = results[0].plot()

            # Display result
            st.subheader("Detection Result")

            st.image(
                result_image,
                channels="BGR",
                use_container_width=True
            )

            # -----------------------------
            # Detected objects
            # -----------------------------
            detected = []

            for result in results:
                if result.boxes is not None:
                    for cls in result.boxes.cls:
                        class_id = int(cls)
                        class_name = model.names[class_id]
                        detected.append(class_name)

            if detected:

                st.success("Waste/Object detected!")

                st.write("### Detected Objects")

                for item in detected:
                    st.write(f"• {item}")

            else:
                st.info("No object detected.")

else:
    st.info("Please upload a waste image to start detection.")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption("AI Waste Management System")
