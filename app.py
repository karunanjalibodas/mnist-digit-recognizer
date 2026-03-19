import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import cv2

# Load model
model = load_model("mnist_model.keras")

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

st.title("🧠 MNIST Digit Recognizer")
st.write("Upload or draw a handwritten digit (0–9)")

# =====================================
# 🔥 PREPROCESS FUNCTION (FINAL)
# =====================================
def preprocess_image(img):
    img = np.array(img)

    # Resize larger first
    img = cv2.resize(img, (64, 64))

    # Smooth noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Auto invert
    if np.mean(img) > 127:
        img = 255 - img

    # Thicken strokes
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(contour)
        digit = img[y:y+h, x:x+w]

        # Resize to 20x20
        digit = cv2.resize(digit, (20, 20))

        # Create blank 28x28
        new_img = np.zeros((28, 28))

        # Center digit
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        new_img[y_offset:y_offset+20, x_offset:x_offset+20] = digit

        img = new_img

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1, 28, 28, 1)

    return img


# =========================
# 📤 IMAGE UPLOAD
# =========================
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')

    st.image(img, caption="Uploaded Image", width=150)

    processed_img = preprocess_image(img)

    prediction = model.predict(processed_img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {digit}")
    st.info(f"Confidence: {confidence:.2f}")

    if confidence < 0.7:
        st.warning("⚠️ Low confidence. Try clearer or centered digit.")


# =========================
# ✍️ DRAW CANVAS
# =========================
st.subheader("✍️ Draw Digit")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0]

    if np.sum(img) > 0:
        img = Image.fromarray(img.astype('uint8'))

        st.image(img, caption="Drawn Image", width=150)

        processed_img = preprocess_image(img)

        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Prediction: {digit}")
        st.info(f"Confidence: {confidence:.2f}")

        if confidence < 0.7:
            st.warning("⚠️ Draw thicker and centered digit.")


# =========================
# 📌 FOOTER
# =========================
st.caption("Best results: white thick digit on black background (MNIST style)")