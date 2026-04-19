# Phone Landmark Collector

This is a standalone Flutter app. It runs the phone camera, detects hand
landmarks on the phone, and saves rows locally as `dataset.csv`.

No Python server is required.

## Run

From this folder:

```bash
cd /home/ali/Desktop/hand_landmarks_project/phone_collector
flutter pub get
flutter run
```

If this folder does not have Android/iOS platform files yet, run this once:

```bash
flutter create .
```

Then run:

```bash
flutter pub get
flutter run
```

## Android Permissions

After `flutter create .`, open:

```text
android/app/src/main/AndroidManifest.xml
```

Make sure this line exists above the `<application>` tag:

```xml
<uses-permission android:name="android.permission.CAMERA" />
```

## Collect

1. Enter the label, for example `alif`.
2. Show your hand to the camera.
3. Wait until the app says `Hand detected`.
4. Tap **Save Sample**.

The saved CSV row uses the same format as your Python collector:

```text
x0,y0,z0,...,x20,y20,z20,label
```

Use the share button in the app bar to send/export the phone's `dataset.csv`.
