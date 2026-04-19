import argparse
import csv
import glob
import os
import tempfile
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPLCONFIG_DIR = os.path.join(tempfile.gettempdir(), "hand_landmarks_matplotlib")
os.makedirs(MPLCONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG_DIR)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hand landmark samples.")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List Linux /dev/video* devices and exit.",
    )
    parser.add_argument(
        "--camera",
        help="Camera index or device path, for example 0, 2, or /dev/video2.",
    )
    return parser.parse_args()


def get_video_device_name(device_path):
    device_name = os.path.basename(device_path)
    name_path = os.path.join("/sys/class/video4linux", device_name, "name")
    try:
        with open(name_path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def list_cameras():
    video_devices = sorted(glob.glob("/dev/video*"))
    if not video_devices:
        print("No /dev/video* devices found.")
        return

    for device_path in video_devices:
        print(f"{device_path}: {get_video_device_name(device_path)}")


def camera_source_name(source):
    return source if isinstance(source, str) else f"camera index {source}"


def parse_camera_source(value):
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def get_camera_candidates(selected_camera):
    if selected_camera is not None:
        return [selected_camera]

    video_devices = sorted(glob.glob("/dev/video*"))
    if video_devices:
        return video_devices

    return list(range(5))


def open_working_camera(selected_camera):
    for source in get_camera_candidates(selected_camera):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(10):
            success, frame = cap.read()
            if success and frame is not None:
                print(f"Using {camera_source_name(source)}")
                return cap
            time.sleep(0.1)

        cap.release()

    return None


args = parse_args()

if args.list_cameras:
    list_cameras()
    exit()

# =========================
# إعداد MediaPipe
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_landmarks_on_image(image, hand_landmarks):
    h, w, _ = image.shape

    for landmarks in hand_landmarks:
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_point = (
                int(landmarks[start_idx].x * w),
                int(landmarks[start_idx].y * h),
            )
            end_point = (
                int(landmarks[end_idx].x * w),
                int(landmarks[end_idx].y * h),
            )
            cv2.line(image, start_point, end_point, (255, 255, 255), 2)

        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image

# =========================
# إدخال اسم الحرف / الكلاس
# =========================
label = input("Enter label (example: alif, ba, ta): ").strip()

if not label:
    print("Label cannot be empty.")
    exit()

# =========================
# اسم ملف البيانات
# =========================
CSV_FILE = os.path.join(BASE_DIR, "dataset.csv")

# =========================
# إنشاء header إذا الملف غير موجود
# =========================
if not os.path.exists(CSV_FILE):
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# =========================
# فتح الكاميرا
# =========================
cap = open_working_camera(parse_camera_source(args.camera))

if cap is None:
    print("Error: Could not open webcam.")
    print("Make sure the camera is connected and no other app is using it.")
    print("Try a specific camera, for example: venv/bin/python landsmark-collector.py --camera /dev/video2")
    exit()

print("\n==============================")
print(f"Current label: {label}")
print("Press 's' to save sample")
print("Press 'q' to quit")
print("==============================\n")

sample_count = 0
latest_row = None

frame_timestamp_ms = 0
empty_frame_count = 0

# =========================
# MediaPipe Hand Landmarker
# =========================
with vision.HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, image = cap.read()
        if not success:
            empty_frame_count += 1
            if empty_frame_count == 1:
                print("Camera opened, but it is not returning frames yet.")
            if empty_frame_count >= 30:
                print("Error: Camera kept returning empty frames.")
                print("Try closing other camera apps or reconnecting the webcam.")
                break
            time.sleep(0.1)
            continue
        empty_frame_count = 0

        # نعكس الصورة مثل المرآة
        image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # معالجة الصورة
        current_timestamp_ms = int(time.time() * 1000)
        if current_timestamp_ms <= frame_timestamp_ms:
            current_timestamp_ms = frame_timestamp_ms + 1
        frame_timestamp_ms = current_timestamp_ms
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        latest_row = None

        # إذا اكتشف اليد
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # رسم اليد
                image = draw_landmarks_on_image(image, [hand_landmarks])

                # تحويل landmarks إلى صف CSV
                row = []
                for lm in hand_landmarks:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(label)
                latest_row = row

        # عرض معلومات على الشاشة
        cv2.putText(
            image,
            f"Label: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Saved: {sample_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            image,
            "Press S to save | Q to quit",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # عرض الصورة
        cv2.imshow("Collect Hand Landmarks", image)

        key = cv2.waitKey(1) & 0xFF

        # حفظ عينة
        if key == ord('s'):
            if latest_row is not None:
                with open(CSV_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(latest_row)

                sample_count += 1
                print(f"Saved sample #{sample_count} for label '{label}'")
            else:
                print("No hand detected. Sample not saved.")

        # خروج
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinished. Total samples saved for '{label}': {sample_count}")
