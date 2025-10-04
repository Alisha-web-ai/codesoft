import streamlit as st
from deepface import DeepFace
from PIL import Image
import cv2
import os
import numpy as np

st.title("🖼 Face Detection & Verification App")

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose Task", ["Face Detection", "Face Verification"])

# -------------------------
# FACE DETECTION
# -------------------------
if choice == "Face Detection":
    st.header("👤 Face Detection")
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if img_file:
        # Save temporary image
        img_path = "temp_detect.jpg"
        with open(img_path, "wb") as f:
            f.write(img_file.read())

        # Show original image
        st.image(img_path, caption="Uploaded Image", use_container_width=True)

        if st.button("Run Face Detection"):
            with st.spinner("⏳ Detecting faces..."):
                try:
                    detections = DeepFace.extract_faces(
                        img_path=img_path,
                        detector_backend="opencv",
                        enforce_detection=False
                    )

                    st.success(f"✅ Detected {len(detections)} face(s).")

                    # Draw rectangles around detected faces
                    img = cv2.imread(img_path)
                    for face in detections:
                        x, y, w, h = face["facial_area"].values()
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                             caption="Detected Faces", use_container_width=True)

                    # Show each cropped face
                    st.subheader("Cropped Faces:")
                    cols = st.columns(len(detections))
                    for i, face in enumerate(detections):
                        face_array = np.array(face["face"])
                        face_img = Image.fromarray((face_array * 255).astype("uint8"))
                        cols[i].image(face_img, caption=f"Face {i+1}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        if os.path.exists(img_path):
            os.remove(img_path)

# -------------------------
# FACE VERIFICATION
# -------------------------
elif choice == "Face Verification":
    st.header("🔍 Face Verification")
    st.write("Upload two images to check if they belong to the same person.")

    img1_file = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
    img2_file = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

    if img1_file and img2_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1_file, caption="First Image", use_container_width=True)
        with col2:
            st.image(img2_file, caption="Second Image", use_container_width=True)

        # Save temporarily
        img1_path = "temp_img1.jpg"
        img2_path = "temp_img2.jpg"

        with open(img1_path, "wb") as f:
            f.write(img1_file.read())
        with open(img2_path, "wb") as f:
            f.write(img2_file.read())

        if st.button("Run Verification"):
            with st.spinner("⏳ Running verification... Please wait."):
                try:
                    result = DeepFace.verify(
                        img1_path,
                        img2_path,
                        model_name="Facenet",        # Faster model
                        detector_backend="opencv",
                        enforce_detection=False
                    )

                    st.success("✅ Verification Completed!")
                    st.json(result)

                    if result["verified"]:
                        st.subheader("✅ The faces MATCH!")
                    else:
                        st.subheader("❌ The faces DO NOT match.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)