import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import cv2
import numpy as np
from keras.models import load_model

# 1. Load the Model
model = load_model('keras_model.h5', compile=False)

# This loads the built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Start Camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # IMPROVED DETECTION: 
    # Lowered the 'minNeighbors' to 5 (from 10) so it detects faces with masks more easily.
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
        face_img = (face_img / 127.5) - 1 

        prediction = model.predict(face_img, verbose=0)
        index = np.argmax(prediction)
        
        # Based on your video:
        # If WITHOUT mask was Class 0 and WITH mask was Class 1
        if index == 1: 
            label = "MASK ON"
            color = (0, 255, 0) # Green
        else:
            label = "NO MASK!"
            color = (0, 0, 255) # Red

        # Draw box and text
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Face Mask Detector', img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()