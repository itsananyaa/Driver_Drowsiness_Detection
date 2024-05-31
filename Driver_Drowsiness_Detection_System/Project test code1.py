import cv2
import winsound

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the sound file
sound_file = r'C:\Users\adity\OneDrive\Desktop\DDD\music.wav'

# Function to play sound
def play_sound():
    winsound.PlaySound(sound_file, winsound.SND_FILENAME)

# Function to detect faces and play sound
def detect_and_play_sound():
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Quit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        return

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        play_sound()

    # Display the frame
    cv2.imshow('Face Detection', frame)

# Create a video capture object
cap = cv2.VideoCapture(0)

# Loop over each frame in the video
while True:
    detect_and_play_sound()