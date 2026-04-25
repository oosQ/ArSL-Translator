import csv
import json
import math
import os
from collections import Counter

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(BASE_DIR, "data_splits")
TRAIN_PATH = os.path.join(SPLIT_DIR, "train.csv")
VAL_PATH = os.path.join(SPLIT_DIR, "val.csv")
TEST_PATH = os.path.join(SPLIT_DIR, "test.csv")
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "phone_collector",
    "assets",
    "models",
    "gesture_model.json",
)

SEED = 7
HIDDEN_SIZE = 64
EPOCHS = 350
LEARNING_RATE = 0.015
L2 = 1e-4
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 25

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

ANGLE_TRIPLETS = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
]

DISTANCE_PAIRS = [
    (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    (4, 8), (8, 12), (12, 16), (16, 20),
    (5, 9), (9, 13), (13, 17), (5, 17),
]


def load_split(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        raise ValueError(f"{os.path.basename(path)} does not contain enough rows")

    features = []
    labels = []
    for row in rows[1:]:
        if len(row) < 64:
            continue
        try:
            raw_values = [float(row[i]) for i in range(63)]
        except ValueError:
            continue
        label = row[63].strip()
        if not label:
            continue

        features.append(build_feature_vector(raw_values))
        labels.append(label)

    if not features:
        raise ValueError(f"No valid samples found in {path}")

    return np.array(features, dtype=np.float64), labels


def build_feature_vector(raw_values):
    points = normalize_points(raw_values)
    flat_points = [coord for point in points for coord in point]
    distance_features = [point_distance(points[a], points[b]) for a, b in DISTANCE_PAIRS]
    angle_features = [joint_angle(points[a], points[b], points[c]) for a, b, c in ANGLE_TRIPLETS]
    return flat_points + distance_features + angle_features


def normalize_points(values):
    wrist = values[0:3]
    points = []
    max_distance = 0.0

    for index in range(21):
        point = (
            values[index * 3] - wrist[0],
            values[index * 3 + 1] - wrist[1],
            values[index * 3 + 2] - wrist[2],
        )
        points.append(point)
        max_distance = max(max_distance, point_distance((0.0, 0.0, 0.0), point))

    scale = max_distance if max_distance > 0 else 1.0
    return [(x / scale, y / scale, z / scale) for x, y, z in points]


def point_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def joint_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    cb = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    ab_norm = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    cb_norm = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)
    if ab_norm == 0 or cb_norm == 0:
        return 0.0

    dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
    cosine = max(-1.0, min(1.0, dot / (ab_norm * cb_norm)))
    return math.acos(cosine) / math.pi


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (test_x - mean) / std,
        mean,
        std,
    )


def encode_labels(labels):
    classes = sorted(set(labels))
    index_by_label = {label: idx for idx, label in enumerate(classes)}
    encoded = np.array([index_by_label[label] for label in labels], dtype=np.int64)
    return encoded, classes


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def relu(x):
    return np.maximum(x, 0.0)


def forward(features, params):
    z1 = features @ params["w1"] + params["b1"]
    a1 = relu(z1)
    logits = a1 @ params["w2"] + params["b2"]
    probs = softmax(logits)
    return z1, a1, logits, probs


def class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return len(labels) / (num_classes * counts)


def loss_and_metrics(features, labels, params, weights_by_class=None):
    _, _, logits, probs = forward(features, params)
    eps = 1e-9
    sample_weights = (
        np.ones(len(labels), dtype=np.float64)
        if weights_by_class is None
        else weights_by_class[labels]
    )
    sample_losses = -np.log(probs[np.arange(len(labels)), labels] + eps)
    loss = float(np.sum(sample_losses * sample_weights) / np.sum(sample_weights))
    loss += 0.5 * L2 * (
        np.sum(params["w1"] ** 2) + np.sum(params["w2"] ** 2)
    )

    predictions = np.argmax(probs, axis=1)
    accuracy = float((predictions == labels).mean())
    return loss, accuracy, probs


def train_mlp(train_x, train_y, val_x, val_y, num_classes):
    rng = np.random.default_rng(SEED)
    params = {
        "w1": rng.normal(0.0, 0.05, size=(train_x.shape[1], HIDDEN_SIZE)),
        "b1": np.zeros(HIDDEN_SIZE, dtype=np.float64),
        "w2": rng.normal(0.0, 0.05, size=(HIDDEN_SIZE, num_classes)),
        "b2": np.zeros(num_classes, dtype=np.float64),
    }

    one_hot_lookup = np.eye(num_classes)
    weights_by_class = class_weights(train_y, num_classes)
    best_params = None
    best_val_loss = float("inf")
    patience = 0

    for _ in range(EPOCHS):
        order = rng.permutation(len(train_x))
        shuffled_x = train_x[order]
        shuffled_y = train_y[order]

        for start in range(0, len(train_x), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_x = shuffled_x[start:end]
            batch_y = shuffled_y[start:end]
            batch_y_one_hot = one_hot_lookup[batch_y]
            batch_weights = weights_by_class[batch_y]

            z1, a1, _, probs = forward(batch_x, params)
            error = (probs - batch_y_one_hot) * batch_weights[:, np.newaxis]
            error /= np.sum(batch_weights)

            grad_w2 = a1.T @ error + (L2 * params["w2"])
            grad_b2 = error.sum(axis=0)

            hidden_error = (error @ params["w2"].T) * (z1 > 0)
            grad_w1 = batch_x.T @ hidden_error + (L2 * params["w1"])
            grad_b1 = hidden_error.sum(axis=0)

            params["w2"] -= LEARNING_RATE * grad_w2
            params["b2"] -= LEARNING_RATE * grad_b2
            params["w1"] -= LEARNING_RATE * grad_w1
            params["b1"] -= LEARNING_RATE * grad_b1

        val_loss, _, _ = loss_and_metrics(val_x, val_y, params, weights_by_class)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {
                "w1": params["w1"].copy(),
                "b1": params["b1"].copy(),
                "w2": params["w2"].copy(),
                "b2": params["b2"].copy(),
            }
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break

    return best_params if best_params is not None else params


def compute_confusion(labels, predictions, classes):
    matrix = []
    for actual_index, actual_label in enumerate(classes):
        row = []
        for predicted_index, _ in enumerate(classes):
            row.append(int(np.sum((labels == actual_index) & (predictions == predicted_index))))
        matrix.append({"label": actual_label, "counts": row})
    return matrix


def per_class_metrics(labels, predictions, classes):
    metrics = {}
    for index, label in enumerate(classes):
        true_positive = int(np.sum((labels == index) & (predictions == index)))
        false_positive = int(np.sum((labels != index) & (predictions == index)))
        false_negative = int(np.sum((labels == index) & (predictions != index)))
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "support": int(np.sum(labels == index)),
        }
    return metrics


def best_unknown_threshold(val_probs, val_true, classes):
    unknown_index = classes.index("unknown") if "unknown" in classes else None
    best = (0.50, 0.12, -1.0)

    for confidence in np.arange(0.45, 0.91, 0.05):
        for margin in np.arange(0.08, 0.31, 0.02):
            predictions = predict_with_thresholds(
                val_probs,
                classes,
                confidence,
                margin,
            )
            score = macro_f1(val_true, predictions, classes)
            if score > best[2]:
                best = (float(confidence), float(margin), float(score))

    # If unknown class does not exist, thresholds still help abstain.
    return best[0], best[1], unknown_index


def macro_f1(true_labels, predicted_labels, classes):
    scores = []
    for index, _ in enumerate(classes):
        tp = np.sum((true_labels == index) & (predicted_labels == index))
        fp = np.sum((true_labels != index) & (predicted_labels == index))
        fn = np.sum((true_labels == index) & (predicted_labels != index))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return float(sum(scores) / len(scores))


def predict_with_thresholds(probabilities, classes, confidence_threshold, margin_threshold):
    predictions = []
    unknown_index = classes.index("unknown") if "unknown" in classes else None

    for row in probabilities:
        best_index = int(np.argmax(row))
        sorted_row = np.sort(row)
        best_probability = float(sorted_row[-1])
        second_probability = float(sorted_row[-2]) if len(sorted_row) > 1 else 0.0
        is_unknown = (
            best_probability < confidence_threshold or
            (best_probability - second_probability) < margin_threshold
        )
        if is_unknown and unknown_index is not None:
            predictions.append(unknown_index)
        else:
            predictions.append(best_index)
    return np.array(predictions, dtype=np.int64)


def evaluate(
    features,
    labels,
    params,
    classes,
    confidence_threshold,
    margin_threshold,
    weights_by_class,
):
    _, accuracy, probabilities = loss_and_metrics(
        features,
        labels,
        params,
        weights_by_class,
    )
    predictions = predict_with_thresholds(
        probabilities,
        classes,
        confidence_threshold,
        margin_threshold,
    )
    thresholded_accuracy = float((predictions == labels).mean())
    return {
        "raw_accuracy": accuracy,
        "thresholded_accuracy": thresholded_accuracy,
        "confusion_matrix": compute_confusion(labels, predictions, classes),
        "per_class": per_class_metrics(labels, predictions, classes),
    }


def main():
    train_x, train_labels = load_split(TRAIN_PATH)
    val_x, val_labels = load_split(VAL_PATH)
    test_x, test_labels = load_split(TEST_PATH)

    train_x, val_x, test_x, mean, std = standardize(train_x, val_x, test_x)
    train_y, classes = encode_labels(train_labels)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    val_y = np.array([class_to_index[label] for label in val_labels], dtype=np.int64)
    test_y = np.array([class_to_index[label] for label in test_labels], dtype=np.int64)
    weights_by_class = class_weights(train_y, len(classes))

    params = train_mlp(train_x, train_y, val_x, val_y, len(classes))
    _, train_accuracy, train_probs = loss_and_metrics(
        train_x,
        train_y,
        params,
        weights_by_class,
    )
    _, val_accuracy, val_probs = loss_and_metrics(
        val_x,
        val_y,
        params,
        weights_by_class,
    )
    _, test_accuracy, _ = loss_and_metrics(
        test_x,
        test_y,
        params,
        weights_by_class,
    )

    confidence_threshold, margin_threshold, _ = best_unknown_threshold(
        val_probs,
        val_y,
        classes,
    )

    payload = {
        "type": "mlp_landmark_classifier",
        "version": 2,
        "labels": classes,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "layers": [
            {
                "activation": "relu",
                "weights": params["w1"].tolist(),
                "bias": params["b1"].tolist(),
            },
            {
                "activation": "linear",
                "weights": params["w2"].tolist(),
                "bias": params["b2"].tolist(),
            },
        ],
        "feature_config": {
            "normalized_landmarks": 63,
            "distance_pairs": DISTANCE_PAIRS,
            "angle_triplets": ANGLE_TRIPLETS,
        },
        "thresholds": {
            "unknown_confidence": confidence_threshold,
            "unknown_margin": margin_threshold,
        },
        "metrics": {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "sample_count": len(train_labels) + len(val_labels) + len(test_labels),
            "train_count": len(train_labels),
            "val_count": len(val_labels),
            "test_count": len(test_labels),
            "label_counts": Counter(train_labels + val_labels + test_labels),
            "validation_thresholded": evaluate(
                val_x,
                val_y,
                params,
                classes,
                confidence_threshold,
                margin_threshold,
                weights_by_class,
            ),
            "test_thresholded": evaluate(
                test_x,
                test_y,
                params,
                classes,
                confidence_threshold,
                margin_threshold,
                weights_by_class,
            ),
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "unknown_confidence_threshold": confidence_threshold,
        "unknown_margin_threshold": margin_threshold,
        "class_counts": dict(sorted(Counter(train_labels + val_labels + test_labels).items())),
    }
    print(f"Model written to {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
