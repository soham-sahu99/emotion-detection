import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

#  Step 1: Check if model file exists before loading
model_path = r"facial_emotion_detection_model.h5"

if not os.path.exists(model_path):
    print(f" Error: Model file not found at {model_path}")
    exit()  # Exit if the model file is missing

#  Step 2: Load the trained model
model = load_model(model_path)
print(" Model loaded successfully!")

#  Step 3: Define emotion labels (adjust based on your model's classes)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

#  Step 4: Load OpenCV face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#  Step 5: Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Error: Unable to access webcam!")
        break

    # Convert frame to grayscale (model expects grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]

        # Resize face to match model input size (48x48)
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Normalize pixel values (if required by your model)
        face_roi = face_roi / 255.0  # Normalize to range [0,1]

        # Expand dimensions to match model input shape (1, 48, 48, 1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Predict emotion
        prediction = model.predict(face_roi)
        emotion_index = np.argmax(prediction)
        emotion_text = emotion_labels[emotion_index]

        # Draw rectangle around the face & label it with emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the output video with emotion labels
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Press 'q' to exit the webcam window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  Step 6: Release resources and close windows
cap.release()
cv2.destroyAllWindows()
print(" Webcam closed successfully!")
