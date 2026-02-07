## Face Mask Detection System 😷
This is a real-time Computer Vision project built using Python, OpenCV, and TensorFlow/Keras. The system detects whether a person is wearing a mask or not through a live webcam feed.

## 🚀 Features
Real-time Detection: Uses webcam to detect faces and classify them instantly.

Deep Learning Model: Powered by a model trained on Google's Teachable Machine.

Visual Alerts: Displays a Green box for "Mask On" and a Red box for "No Mask".

## 🛠️ Tech Stack
Language: Python

Libraries: OpenCV, TensorFlow, Keras, NumPy

Algorithm: Haar Cascades (for face detection)

## 📋 Prerequisites
Before running the project, make sure you have the following installed:

Python 3.10 or higher

A working webcam

## ⚙️ Installation & Setup
Clone the repository:

## Bash
git clone https://github.com/your-username/face-mask-detector.git
cd face-mask-detector
Create a Virtual Environment (Recommended):

Bash
python -m venv .venv
source .venv/bin/activate  # For Mac/Linux
.venv\Scripts\activate     # For Windows
Install Dependencies:

Bash
pip install tensorflow==2.15.0 opencv-python numpy
Required Files: Ensure you have these files in your project folder:

main.py (The code)

keras_model.h5 (Trained model)

haarcascade_frontalface_default.xml (Face detection file)

🏃 How to Run
Simply run the following command in your terminal:

Bash
python main.py
Press 'q' to exit the camera window.
