import cv2
import numpy as np
import pygame

# Initialize Pygame mixer
pygame.mixer.init()

# Load sound file
sound = pygame.mixer.Sound(r"C:\Users\adity\OneDrive\Pictures\DDD\music.wav")

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to check if eyes are closed
def are_eyes_closed(eyes):
    # If no eyes are detected, return False
    if len(eyes) == 0:
        return False
    # Check the height of the first detected eye
    (ex, ey, ew, eh) = eyes[0]
    # If the height is smaller than a threshold, consider the eye closed
    if eh < 40:
        return True
    return False

# Load video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Equalize histogram to improve contrast
    gray = cv2.equalizeHist(gray)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) which is the face area
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Apply Gaussian blur to reduce noise
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        # Check if eyes are closed
        if are_eyes_closed(eyes):
            # Play sound
            sound.play()

        # Draw rectangles around the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
