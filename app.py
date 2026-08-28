import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_model.keras")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🐱 Cat vs Dog Image Classifier")
st.write("Upload an image and the trained CNN will predict whether it is a cat or a dog.")

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(img_array, verbose=0)[0][0]

    # < 0.5 = Cat
    # >= 0.5 = Dog

    if prediction < 0.5:
        predicted_class = "CAT 🐱"
        confidence = (1 - prediction) * 100
    else:
        predicted_class = "DOG 🐶"
        confidence = prediction * 100

    # -----------------------------
    # Display result
    # -----------------------------
    st.subheader("Prediction")

    st.success(f"**{predicted_class}**")

    st.write(f"Confidence: **{confidence:.2f}%**")

    st.progress(float(confidence / 100))