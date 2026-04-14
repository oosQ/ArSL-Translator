import cv2
import csv
import os
import mediapipe as mp

# =========================
# إعداد MediaPipe
# =========================
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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
CSV_FILE = "dataset.csv"

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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n==============================")
print(f"Current label: {label}")
print("Press 's' to save sample")
print("Press 'q' to quit")
print("==============================\n")

sample_count = 0
latest_row = None

# =========================
# MediaPipe Hands
# =========================
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # نعكس الصورة مثل المرآة
        image = cv2.flip(image, 1)

        # تحسين الأداء
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # معالجة الصورة
        results = hands.process(image_rgb)

        image.flags.writeable = True

        latest_row = None

        # إذا اكتشف اليد
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # رسم اليد
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # تحويل landmarks إلى صف CSV
                row = []
                for lm in hand_landmarks.landmark:
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