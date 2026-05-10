# Model Result Screenshots And Confusion Matrix

## Final Model Metrics

| Metric | Value |
| --- | ---: |
| Total samples | 3,298 |
| Train samples | 2,308 |
| Validation samples | 495 |
| Test samples | 495 |
| Train accuracy | 99.96% |
| Validation accuracy | 100.00% |
| Test accuracy | 99.80% |
| Thresholded test accuracy | 99.80% |
| Correct test predictions | 494/495 |
| Test errors | 1 |
| Unknown confidence threshold | 45.00% |
| Unknown margin threshold | 8.00% |

## Test Confusion Matrix

Rows are actual labels. Columns are predicted labels.

| Actual \ Predicted | alif | ba | dal | ha | jeem | kha | raa | ta | tha | thal | unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alif | 45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ba | 0 | 45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dal | 0 | 0 | 45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ha | 0 | 0 | 0 | 45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| jeem | 0 | 0 | 0 | 0 | 45 | 0 | 0 | 0 | 0 | 0 | 0 |
| kha | 0 | 0 | 0 | 0 | 0 | 45 | 0 | 0 | 0 | 0 | 0 |
| raa | 0 | 0 | 0 | 0 | 0 | 0 | 45 | 0 | 0 | 0 | 0 |
| ta | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45 | 0 | 0 | 0 |
| tha | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45 | 0 | 0 |
| thal | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45 | 0 |
| unknown | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 44 |

## Per-Class Test Metrics

| Label | Precision | Recall | Support |
| --- | ---: | ---: | ---: |
| alif | 97.83% | 100.00% | 45 |
| ba | 100.00% | 100.00% | 45 |
| dal | 100.00% | 100.00% | 45 |
| ha | 100.00% | 100.00% | 45 |
| jeem | 100.00% | 100.00% | 45 |
| kha | 100.00% | 100.00% | 45 |
| raa | 100.00% | 100.00% | 45 |
| ta | 100.00% | 100.00% | 45 |
| tha | 100.00% | 100.00% | 45 |
| thal | 100.00% | 100.00% | 45 |
| unknown | 100.00% | 97.78% | 45 |

## How The Statistics Are Calculated

The confusion matrix compares the real label with the predicted label. Rows are the real classes and columns are the predicted classes. The diagonal cells are correct predictions. Values outside the diagonal are mistakes.

For one class, the statistics are calculated like this:

```text
TP = True Positive
FP = False Positive
FN = False Negative
TN = True Negative
```

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

```text
Precision = TP / (TP + FP)
```

Precision answers: when the model predicts this class, how often is it correct?

```text
Recall = TP / (TP + FN)
```

Recall answers: from all real samples of this class, how many did the model find?

```text
F1-score = 2 * (Precision * Recall) / (Precision + Recall)
```

F1-score combines precision and recall into one balanced value.

```text
Support = number of real test samples for that class
```

For this model, each class has `45` test samples. The total test set has:

```text
11 classes * 45 samples = 495 test samples
```

Overall test accuracy:

```text
Accuracy = correct predictions / total test samples
Accuracy = 494 / 495
Accuracy = 0.99798 = 99.80%
```

Example for class `alif`:

```text
TP = 45
FP = 1
FN = 0

Precision = 45 / (45 + 1) = 0.9783 = 97.83%
Recall = 45 / (45 + 0) = 1.0000 = 100.00%
```

The `alif` class has one false positive because one real `unknown` sample was predicted as `alif`.

Example for class `unknown`:

```text
TP = 44
FP = 0
FN = 1

Precision = 44 / (44 + 0) = 1.0000 = 100.00%
Recall = 44 / (44 + 1) = 0.9778 = 97.78%
```

The `unknown` class has one false negative because one real unknown sign was not detected as unknown.

## Technical Challenges Faced

1. **Hand position changes**
   - The same sign can look different when the hand is closer, farther, rotated, or shifted in the camera frame.
   - To reduce this problem, landmarks are normalized using the wrist as the origin and scaled by the maximum distance from the wrist.

2. **Lighting and camera quality**
   - Bad lighting, shadows, motion blur, or low camera quality can reduce MediaPipe landmark accuracy.
   - This affects the classifier because the neural network depends on the landmark coordinates.

3. **Similar hand signs**
   - Some Arabic signs have similar finger positions, so the distance between classes can be small.
   - The model uses extra distance and joint-angle features to help separate similar signs.

4. **Unknown sign detection**
   - A normal classifier always chooses one of the trained labels, even when the gesture is not part of the dataset.
   - This project adds confidence and margin thresholds so uncertain predictions can become `Unknown sign`.

5. **Overfitting risk**
   - A model can memorize the training data and perform worse on new camera input.
   - The training script uses validation data, early stopping, L2 regularization, and a separate test set to reduce this risk.

6. **Real-time mobile inference**
   - The app must detect landmarks and run classification continuously from the camera stream.
   - A small MLP model exported as JSON keeps inference lightweight enough for the Flutter app.

7. **Front camera orientation**
   - The front camera can mirror or rotate the preview.
   - The app handles sensor orientation and front-camera drawing so the landmark overlay matches the visible hand.

## More Technical Screenshot Ideas

Use these screenshots if you need a stronger implementation/result section:

1. **Landmark overlay screenshot**
   - Shows that MediaPipe detects the hand and extracts 21 points.
   - Use this to explain the input to the classifier.

2. **Live prediction screenshot**
   - Shows the final predicted label and confidence.
   - Use this as the main model output screenshot.

3. **Unknown sign screenshot**
   - Shows that the model can reject unclear or untrained signs.
   - Use this to explain the threshold technique.

4. **Metrics screenshot**
   - Shows test accuracy, train accuracy, total samples, thresholds, and label counts.
   - Use this as evidence of evaluation.

5. **Dataset table screenshot**
   - Shows the saved landmark coordinates and labels.
   - Use this to explain how the dataset was collected.

6. **Confusion matrix screenshot**
   - Shows class-by-class performance.
   - Use the generated `confusion_matrix_test.png` in your report.

## Result Screenshot Technique

Use the phone app screenshots for the implementation result section. Capture the images in this order:

1. **Detector Model - correct result**: open the `Detector` tab, show one trained sign, and capture the camera preview, landmark skeleton, `Live Prediction`, confidence, and `Model Metrics` card.
2. **Detector Model - unknown result**: show a hand pose that is not part of the trained classes and capture the `Unknown sign` output.
3. **Model Metrics card**: scroll so the metrics are visible, especially test accuracy, train accuracy, sample count, threshold, and label counts.
4. **Data Collector tab**: capture the camera preview with landmarks and the label input before or after saving a sample.
5. **Table View tab**: capture the dataset table to show the stored landmark coordinates and labels.
6. **Confusion matrix PNG**: use `docs/model_results/confusion_matrix_test.png` as the technical evaluation screenshot.

For a clean screenshot, keep the phone vertical, use good lighting, place the hand fully inside the camera frame, and wait until the confidence value is stable for a moment before capturing.

## Short Explanation For Report

The confusion matrix shows that the model classified 494 of 495 test samples correctly. All trained sign classes were classified correctly on the test set. The only error occurred in the `unknown` class, where one unknown sample was predicted as `alif`. This gives a final test accuracy of 99.80%.
