# 🧠 MNIST Digit Recognizer Web App

---

## 📌 Project Overview

* The MNIST Digit Recognizer is a deep learning-based web application that identifies handwritten digits (0–9).
* Users can either upload an image or draw a digit on a canvas.
* The system predicts the digit in real-time using a Convolutional Neural Network (CNN).

---

## 🚀 Features

* ✍️ Draw digits using an interactive canvas
* 📤 Upload handwritten digit images
* 🔍 Real-time prediction
* 📊 Confidence score display
* 📈 Probability distribution chart
* ⚡ Fast and user-friendly interface

---

## 🧠 Tech Stack

* Programming Language: Python
* Frontend & Deployment: Streamlit
* Machine Learning: TensorFlow / Keras

* Libraries Used:
  * NumPy
  * OpenCV
  * Pillow (PIL)
  * streamlit-drawable-canvas

---

## 📊 Dataset

* Dataset: MNIST Dataset
* 70,000 grayscale images of handwritten digits
* Image size: 28 × 28 pixels

* Split:
  * 60,000 training images
  * 10,000 testing images

---

## ⚙️ How It Works

* User inputs digit (upload or draw)
* Image is preprocessed (resize, normalize, invert)
* Processed image is passed to CNN model
* Model predicts digit (0–9)
* Result is displayed with confidence score

---

## 🏗️ Project Structure

* app.py → Main Streamlit application
* mnist_model.h5 → Trained model
* requirements.txt → Dependencies
* runtime.txt → Python version (3.10)

---

## 🖥️ Installation & Setup

### 1. Clone Repository
* git clone https://github.com/karunanjalibodas/mnist-digit-recognizer.git
* cd mnist-digit-recognizer

---

### 2. Install Dependencies
* pip install -r requirements.txt

---

### 3. Run Application
* streamlit run app.py

---

## 📈 Results

* Achieved ~98% accuracy on MNIST dataset
* Works for both uploaded and drawn digits
* Provides real-time predictions

---

## ⚠️ Challenges Faced

* TensorFlow compatibility issues
* Model loading errors due to Keras version mismatch
* Deployment issues on Streamlit Cloud
* Image preprocessing inconsistencies

---

## ✅ Solutions

* Used Python 3.10 via runtime.txt
* Converted model to .h5 format
* Simplified preprocessing pipeline
* Fixed deployment configuration

---

## 🚀 Future Enhancements

* Explainable AI (heatmaps)
* Improved UI/UX
* User authentication
* Prediction history storage
* Cloud deployment with custom domain

---

## 📚 Learning Outcomes

* Understanding CNN architecture
* Hands-on with TensorFlow & Keras
* Streamlit deployment
* Debugging real-world ML issues
* End-to-end ML project development

---

## 👩‍💻 Author

* Karunanjali
* INT-2026-1462

---

## ⭐ If you like this project

* Give it a ⭐ on GitHub
* Share with others
