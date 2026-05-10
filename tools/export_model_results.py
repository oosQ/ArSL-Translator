import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "phone_collector" / "assets" / "models" / "gesture_model.json"
OUTPUT_DIR = ROOT / "docs" / "model_results"


def percent(value):
    return f"{value * 100:.2f}%"


def load_model():
    with MODEL_PATH.open() as f:
        return json.load(f)


def confusion_array(confusion_rows):
    return np.array([row["counts"] for row in confusion_rows], dtype=int)


def save_confusion_csv(labels, matrix, path):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual \\ predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def save_per_class_csv(per_class, path):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "support"])
        for label, metrics in per_class.items():
            writer.writerow(
                [
                    label,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    metrics["support"],
                ]
            )


def save_confusion_png(labels, matrix, path):
    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title("Test Confusion Matrix", pad=16, fontsize=16)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("Actual label", fontsize=12)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    max_value = matrix.max()
    threshold = max_value / 2
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            color = "white" if value > threshold else "#1f2937"
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold" if value else "normal",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_markdown(model, labels, matrix, path):
    metrics = model["metrics"]
    test = metrics["test_thresholded"]
    per_class = test["per_class"]

    correct = int(np.trace(matrix))
    total = int(matrix.sum())
    errors = total - correct

    rows = [
        "# Model Result Screenshots And Confusion Matrix",
        "",
        "## Final Model Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total samples | {metrics['sample_count']:,} |",
        f"| Train samples | {metrics['train_count']:,} |",
        f"| Validation samples | {metrics['val_count']:,} |",
        f"| Test samples | {metrics['test_count']:,} |",
        f"| Train accuracy | {percent(metrics['train_accuracy'])} |",
        f"| Validation accuracy | {percent(metrics['val_accuracy'])} |",
        f"| Test accuracy | {percent(metrics['test_accuracy'])} |",
        f"| Thresholded test accuracy | {percent(test['thresholded_accuracy'])} |",
        f"| Correct test predictions | {correct}/{total} |",
        f"| Test errors | {errors} |",
        f"| Unknown confidence threshold | {percent(model['thresholds']['unknown_confidence'])} |",
        f"| Unknown margin threshold | {percent(model['thresholds']['unknown_margin'])} |",
        "",
        "## Test Confusion Matrix",
        "",
        "Rows are actual labels. Columns are predicted labels.",
        "",
        "| Actual \\ Predicted | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join(["---:"] * len(labels)) + " |",
    ]

    for label, row in zip(labels, matrix):
        rows.append("| " + label + " | " + " | ".join(str(v) for v in row) + " |")

    rows.extend(
        [
            "",
            "## Per-Class Test Metrics",
            "",
            "| Label | Precision | Recall | Support |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label in labels:
        item = per_class[label]
        rows.append(
            f"| {label} | {percent(item['precision'])} | "
            f"{percent(item['recall'])} | {item['support']} |"
        )

    rows.extend(
        [
            "",
            "## How The Statistics Are Calculated",
            "",
            "The confusion matrix compares the real label with the predicted label. Rows are the real classes and columns are the predicted classes. The diagonal cells are correct predictions. Values outside the diagonal are mistakes.",
            "",
            "For one class, the statistics are calculated like this:",
            "",
            "```text",
            "TP = True Positive",
            "FP = False Positive",
            "FN = False Negative",
            "TN = True Negative",
            "```",
            "",
            "```text",
            "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
            "```",
            "",
            "```text",
            "Precision = TP / (TP + FP)",
            "```",
            "",
            "Precision answers: when the model predicts this class, how often is it correct?",
            "",
            "```text",
            "Recall = TP / (TP + FN)",
            "```",
            "",
            "Recall answers: from all real samples of this class, how many did the model find?",
            "",
            "```text",
            "F1-score = 2 * (Precision * Recall) / (Precision + Recall)",
            "```",
            "",
            "F1-score combines precision and recall into one balanced value.",
            "",
            "```text",
            "Support = number of real test samples for that class",
            "```",
            "",
            "For this model, each class has `45` test samples. The total test set has:",
            "",
            "```text",
            "11 classes * 45 samples = 495 test samples",
            "```",
            "",
            "Overall test accuracy:",
            "",
            "```text",
            "Accuracy = correct predictions / total test samples",
            f"Accuracy = {correct} / {total}",
            f"Accuracy = {correct / total:.5f} = {percent(correct / total)}",
            "```",
            "",
            "Example for class `alif`:",
            "",
            "```text",
            "TP = 45",
            "FP = 1",
            "FN = 0",
            "",
            "Precision = 45 / (45 + 1) = 0.9783 = 97.83%",
            "Recall = 45 / (45 + 0) = 1.0000 = 100.00%",
            "```",
            "",
            "The `alif` class has one false positive because one real `unknown` sample was predicted as `alif`.",
            "",
            "Example for class `unknown`:",
            "",
            "```text",
            "TP = 44",
            "FP = 0",
            "FN = 1",
            "",
            "Precision = 44 / (44 + 0) = 1.0000 = 100.00%",
            "Recall = 44 / (44 + 1) = 0.9778 = 97.78%",
            "```",
            "",
            "The `unknown` class has one false negative because one real unknown sign was not detected as unknown.",
            "",
            "## Technical Challenges Faced",
            "",
            "1. **Hand position changes**",
            "   - The same sign can look different when the hand is closer, farther, rotated, or shifted in the camera frame.",
            "   - To reduce this problem, landmarks are normalized using the wrist as the origin and scaled by the maximum distance from the wrist.",
            "",
            "2. **Lighting and camera quality**",
            "   - Bad lighting, shadows, motion blur, or low camera quality can reduce MediaPipe landmark accuracy.",
            "   - This affects the classifier because the neural network depends on the landmark coordinates.",
            "",
            "3. **Similar hand signs**",
            "   - Some Arabic signs have similar finger positions, so the distance between classes can be small.",
            "   - The model uses extra distance and joint-angle features to help separate similar signs.",
            "",
            "4. **Unknown sign detection**",
            "   - A normal classifier always chooses one of the trained labels, even when the gesture is not part of the dataset.",
            "   - This project adds confidence and margin thresholds so uncertain predictions can become `Unknown sign`.",
            "",
            "5. **Overfitting risk**",
            "   - A model can memorize the training data and perform worse on new camera input.",
            "   - The training script uses validation data, early stopping, L2 regularization, and a separate test set to reduce this risk.",
            "",
            "6. **Real-time mobile inference**",
            "   - The app must detect landmarks and run classification continuously from the camera stream.",
            "   - A small MLP model exported as JSON keeps inference lightweight enough for the Flutter app.",
            "",
            "7. **Front camera orientation**",
            "   - The front camera can mirror or rotate the preview.",
            "   - The app handles sensor orientation and front-camera drawing so the landmark overlay matches the visible hand.",
            "",
            "## More Technical Screenshot Ideas",
            "",
            "Use these screenshots if you need a stronger implementation/result section:",
            "",
            "1. **Landmark overlay screenshot**",
            "   - Shows that MediaPipe detects the hand and extracts 21 points.",
            "   - Use this to explain the input to the classifier.",
            "",
            "2. **Live prediction screenshot**",
            "   - Shows the final predicted label and confidence.",
            "   - Use this as the main model output screenshot.",
            "",
            "3. **Unknown sign screenshot**",
            "   - Shows that the model can reject unclear or untrained signs.",
            "   - Use this to explain the threshold technique.",
            "",
            "4. **Metrics screenshot**",
            "   - Shows test accuracy, train accuracy, total samples, thresholds, and label counts.",
            "   - Use this as evidence of evaluation.",
            "",
            "5. **Dataset table screenshot**",
            "   - Shows the saved landmark coordinates and labels.",
            "   - Use this to explain how the dataset was collected.",
            "",
            "6. **Confusion matrix screenshot**",
            "   - Shows class-by-class performance.",
            "   - Use the generated `confusion_matrix_test.png` in your report.",
            "",
            "## Result Screenshot Technique",
            "",
            "Use the phone app screenshots for the implementation result section. Capture the images in this order:",
            "",
            "1. **Detector Model - correct result**: open the `Detector` tab, show one trained sign, and capture the camera preview, landmark skeleton, `Live Prediction`, confidence, and `Model Metrics` card.",
            "2. **Detector Model - unknown result**: show a hand pose that is not part of the trained classes and capture the `Unknown sign` output.",
            "3. **Model Metrics card**: scroll so the metrics are visible, especially test accuracy, train accuracy, sample count, threshold, and label counts.",
            "4. **Data Collector tab**: capture the camera preview with landmarks and the label input before or after saving a sample.",
            "5. **Table View tab**: capture the dataset table to show the stored landmark coordinates and labels.",
            "6. **Confusion matrix PNG**: use `docs/model_results/confusion_matrix_test.png` as the technical evaluation screenshot.",
            "",
            "For a clean screenshot, keep the phone vertical, use good lighting, place the hand fully inside the camera frame, and wait until the confidence value is stable for a moment before capturing.",
            "",
            "## Short Explanation For Report",
            "",
            "The confusion matrix shows that the model classified 494 of 495 test samples correctly. All trained sign classes were classified correctly on the test set. The only error occurred in the `unknown` class, where one unknown sample was predicted as `alif`. This gives a final test accuracy of 99.80%.",
        ]
    )

    path.write_text("\n".join(rows) + "\n")


def main():
    model = load_model()
    labels = model["labels"]
    test = model["metrics"]["test_thresholded"]
    matrix = confusion_array(test["confusion_matrix"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_csv(labels, matrix, OUTPUT_DIR / "confusion_matrix_test.csv")
    save_per_class_csv(test["per_class"], OUTPUT_DIR / "per_class_test_metrics.csv")
    save_confusion_png(labels, matrix, OUTPUT_DIR / "confusion_matrix_test.png")
    save_markdown(model, labels, matrix, OUTPUT_DIR / "result_screenshot_guide.md")

    print(f"Wrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
