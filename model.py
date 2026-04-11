import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = vision.HandLandmarker.create_from_options(options)

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press Q to quit.")

frame_timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB as MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image to a MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Increase timestamp because VIDEO mode requires time for each frame
    frame_timestamp_ms = int(time.time() * 1000)

    # Perform detection
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # If a hand is detected, draw landmarks
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Convert landmarks to a format suitable for drawing
            hand_landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
            for landmark in hand_landmarks:
                hand_landmarks_proto.landmark.add(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z
                )

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS
            )

        # Print only the first 3 landmarks for testing
        first_hand = result.hand_landmarks[0]
        print("Detected hand landmarks:")
        for i, lm in enumerate(first_hand[:3]):
            print(f"Landmark {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

    # Display the image
    cv2.imshow("Hand Landmarks Test", frame)

    # للخروج
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()