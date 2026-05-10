# Hand Landmark Sign-To-Text Model Implementation Notes

## Recommended Screenshots

Take screenshots from the Flutter app in these places:

1. **Data Collector tab**
   - Show the camera preview with the hand landmark skeleton drawn on the hand.
   - Show the label field with one label such as `alif`, `ba`, or `ta`.
   - Show the `Saved samples` counter after saving a sample.

2. **Table View tab**
   - Show `dataset.csv` and the number of collected rows.
   - Show the table columns with landmark values such as `x0`, `y0`, `z0`, `x8`, `y8`, `z8`, and `Label`.
   - Open one row detail if you need a screenshot of all 63 landmark coordinates plus the label.

3. **Detector Model tab**
   - Show the camera preview with landmarks.
   - Show the `Live Prediction` card with the predicted Arabic sign name and confidence.
   - Show the `Model Metrics` card with test accuracy, train accuracy, sample count, unknown threshold, and label counts.

4. **Model result examples**
   - Capture one confident correct prediction, for example `حرف الألف`.
   - Capture an `Unknown sign` result by showing a gesture that is not in the trained classes.
   - Capture the model metrics card as evidence of the final trained model result.

## Current Model Result

The exported model is saved at:

`phone_collector/assets/models/gesture_model.json`

Current metrics from the exported model:

| Metric | Value |
| --- | ---: |
| Total samples | 3,298 |
| Train samples | 2,308 |
| Validation samples | 495 |
| Test samples | 495 |
| Train accuracy | 99.96% |
| Validation accuracy | 100.00% |
| Test accuracy | 99.80% |
| Unknown confidence threshold | 45% |
| Unknown margin threshold | 8% |

The model supports these labels:

`alif`, `ba`, `dal`, `ha`, `jeem`, `kha`, `raa`, `ta`, `tha`, `thal`, `unknown`

Test-set result summary:

- 494 out of 495 test samples were classified correctly.
- The only test error was one `unknown` sample predicted as `alif`.
- Most classes have 100% precision and recall on the test set.
- `alif` precision is 97.83% because one unknown sign was classified as `alif`.
- `unknown` recall is 97.78% because one unknown sample was missed.

## How The System Works

The app uses two models in sequence:

1. **MediaPipe Hand Landmarker**
   - Detects one hand from the camera image.
   - Returns 21 landmarks for the hand.
   - Each landmark has `x`, `y`, and `z` coordinates.

2. **Custom gesture classifier**
   - Uses the 21 landmarks to build a feature vector.
   - Runs a small neural network classifier from `gesture_model.json`.
   - Outputs the predicted sign label and confidence.

The full prediction flow is:

```text
Camera frame
  -> MediaPipe hand detection
  -> 21 hand landmarks
  -> Normalize landmarks around the wrist
  -> Add distance and joint-angle features
  -> Standardize features using training mean and standard deviation
  -> MLP neural network
  -> Softmax probabilities
  -> Unknown-threshold check
  -> Final sign text
```

## Feature Extraction

Each sample starts with 63 raw values:

```text
21 landmarks * 3 coordinates = 63 values
```

The training and app code transform the raw landmarks into 86 model features:

| Feature type | Count | Purpose |
| --- | ---: | --- |
| Normalized landmark coordinates | 63 | Shape of the full hand |
| Distance features | 13 | Distances between key fingertips, wrist, and palm points |
| Angle features | 10 | Finger bending and joint pose information |
| Total | 86 | Final classifier input |

Normalization uses the wrist landmark as the origin. Then all points are divided by the maximum distance from the wrist. This helps the model work even when the hand is closer to or farther from the camera.

## Classifier Architecture

The classifier type is:

```text
mlp_landmark_classifier
```

It is a small multilayer perceptron:

```text
86 input features
  -> Dense hidden layer with 64 neurons
  -> ReLU activation
  -> Dense output layer with 11 class scores
  -> Softmax confidence probabilities
```

Training uses:

- Stratified dataset split: 70% train, 15% validation, 15% test.
- Class weighting to reduce class imbalance effects.
- L2 regularization to reduce overfitting.
- Early stopping using validation loss.
- A validation search for the unknown-sign confidence and margin thresholds.

## Unknown Sign Handling

The model returns `Unknown sign` when either condition is true:

```text
best class confidence < 45%
```

or:

```text
best confidence - second best confidence < 8%
```

This means the app does not only choose the highest class. It also checks whether the prediction is confident enough and clearly separated from the second-best class.

## Evaluation Formulas

The confusion matrix is read by comparing actual labels against predicted labels. Correct predictions are on the diagonal. Mistakes are outside the diagonal.

```text
Accuracy = correct predictions / total predictions
```

For each class:

```text
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-score = 2 * (Precision * Recall) / (Precision + Recall)
Support = number of real samples for the class
```

Where:

- `TP` means the model predicted the class and the real label was also that class.
- `FP` means the model predicted the class but the real label was different.
- `FN` means the real label was the class but the model predicted another class.
- `TN` means the sample was not this class and the model did not predict this class.

For the current test set:

```text
Accuracy = 494 / 495 = 99.80%
```

For `alif`:

```text
Precision = 45 / (45 + 1) = 97.83%
Recall = 45 / (45 + 0) = 100.00%
```

For `unknown`:

```text
Precision = 44 / (44 + 0) = 100.00%
Recall = 44 / (44 + 1) = 97.78%
```

The generated technical result files are in:

`docs/model_results/`

## Challenges Faced

The main implementation challenges were:

- **Hand position variation:** the same sign changes when the hand moves closer, farther, left, right, or rotates. Normalizing around the wrist and scaling by hand size reduces this issue.
- **Lighting and camera noise:** weak lighting and motion blur can make MediaPipe landmarks less stable.
- **Similar signs:** some signs have very close finger shapes, so extra distance and angle features were added to improve separation.
- **Unknown signs:** a classifier normally forces every input into one trained class, so confidence and margin thresholds were added to reject uncertain gestures.
- **Overfitting:** the model could memorize collected samples, so the project uses train/validation/test splits, L2 regularization, class weighting, and early stopping.
- **Mobile real-time performance:** the app must run camera capture, landmark detection, and classification continuously, so the classifier is kept small and exported as JSON.
- **Camera orientation:** front-camera mirroring and sensor rotation must be handled so the landmark overlay matches the real hand position.

## Files To Mention

| File | Role |
| --- | --- |
| `dataset.csv` | Full landmark dataset with labels |
| `prepare_dataset_splits.py` | Creates train, validation, and test CSV splits |
| `train_gesture_model.py` | Trains the MLP and exports the JSON model |
| `phone_collector/assets/models/gesture_model.json` | Exported model weights, labels, thresholds, and metrics |
| `phone_collector/lib/main.dart` | Flutter app, camera processing, feature extraction, prediction, UI |
| `hand_landmarker.task` | MediaPipe hand landmark detector model |
| `model.py` | Python webcam test for drawing hand landmarks |

## Suggested Report Text

This project recognizes Arabic hand signs by combining MediaPipe hand landmark detection with a custom trained neural network classifier. The camera frame is first processed by MediaPipe, which extracts 21 hand landmarks. These landmarks are normalized around the wrist and converted into a feature vector containing landmark coordinates, key point distances, and finger joint angles. The final 86-feature vector is passed to a small multilayer perceptron trained on the collected dataset.

The dataset contains 3,298 samples across 11 labels. It was split into 2,308 training samples, 495 validation samples, and 495 test samples. The exported model achieved 99.80% test accuracy. The application also includes unknown-sign handling using confidence and margin thresholds, allowing the system to reject gestures that are not clearly recognized.
