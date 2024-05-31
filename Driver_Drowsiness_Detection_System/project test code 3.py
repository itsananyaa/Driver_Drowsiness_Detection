import cv2
import numpy as np
import pygame
import time

# Initialize Pygame mixer
pygame.mixer.init()

# Load sound file
sound = pygame.mixer.Sound(r"C:\Users\adity\OneDrive\Desktop\DDD\music.wav")

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to check if eyes are closed
def are_eyes_closed(eyes):
    # If no eyes are detected, return True
    if len(eyes) == 0:
        return True
    # Check the height of the eyes
    heights = [h for (_, _, _, h) in eyes]
    avg_height = sum(heights) / len(heights)
    # If the average height is smaller than a threshold, consider the eyes closed
    if avg_height < 45:  # Adjust threshold as needed
        return True
    return False

# Load video capture
cap = cv2.VideoCapture(0)

# Initialize variables
start_time = None
sound_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) which is the face area
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        # Print the height of detected eyes
        for (_, _, _, h) in eyes:
            print("Eye Height:", h)

        # Check if eyes are closed
        if are_eyes_closed(eyes):
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > 2 and not sound_playing:
                # Play sound
                sound.play()
                sound_playing = True
        else:
            start_time = None
            sound_playing = False

        # Draw rectangles around the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
