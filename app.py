import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mango_model.h5")

model = load_model()

class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

st.title("🌿 Mango Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a Mango Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((128, 128))   # change if your model uses different size
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
