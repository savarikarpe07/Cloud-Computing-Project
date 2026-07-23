import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon="🍃",
    layout="centered"
)

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mango_model.h5")
    return model

model = load_model()

# -----------------------------------
# Class Names
# -----------------------------------
classes = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

# -----------------------------------
# Header
# -----------------------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#2E8B57;'>
        🍃 Mango Leaf Disease Detection System
    </h1>

    <h4 style='text-align:center;color:gray;'>
        Deep Learning Based Disease Prediction using MobileNetV2
    </h4>

    <hr>
    """,
    unsafe_allow_html=True
)

st.write(
    "Upload a mango leaf image to identify the disease."
)

# -----------------------------------
# Upload Image
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Mango Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# Prediction
# -----------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    # -----------------------------------
    # PREPROCESSING
    # -----------------------------------
    img = image.resize((128, 128))

    img_array = np.array(img, dtype=np.float32)

    # IMPORTANT:
    # DO NOT divide by 255
    # Model already contains Rescaling layer

    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------------
    # Prediction
    # -----------------------------------
    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_class_index = np.argmax(prediction)

    predicted_class = classes[predicted_class_index]

    confidence = float(
        np.max(prediction) * 100
    )

    st.markdown("---")

    st.markdown("## 🔍 Prediction Result")

    st.success(
        f"🍃 Predicted Disease: {predicted_class}"
    )

    st.metric(
        "Confidence Score",
        f"{confidence:.2f}%"
    )

    st.progress(confidence / 100)

    # -----------------------------------
    # Show All Probabilities
    # -----------------------------------
    st.markdown("### Disease Probabilities")

    for i, disease in enumerate(classes):

        prob = float(
            prediction[0][i] * 100
        )

        st.write(
            f"**{disease}** : {prob:.2f}%"
        )

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;color:gray'>
        Developed using TensorFlow, MobileNetV2 and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
