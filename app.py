import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
model = tf.keras.models.load_model("digit_model.h5")
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

st.set_page_config(page_title="AI Digit Recognition", layout="centered")

st.title("✍️ AI Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below")

# Create drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict Digit"):

    if canvas_result.image_data is not None:

        # Get image data
        img = canvas_result.image_data

        # Convert to grayscale
        img = Image.fromarray((img[:, :, 0]).astype('uint8'))

        # Resize to MNIST size
        img = img.resize((28, 28))

        # Convert to numpy array
        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        # Reshape for model
        img_array = img_array.reshape(1, 28, 28, 1)

        # Prediction
        prediction = model.predict(img_array)

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"✅ Predicted Digit: {predicted_digit}")
        st.info(f"📊 Confidence: {confidence:.2f}%")