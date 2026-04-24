import csv
import json
import math
import os
import random
from collections import Counter

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "phone_collector",
    "assets",
    "models",
    "gesture_model.json",
)

SEED = 7
EPOCHS = 900
LEARNING_RATE = 0.08
L2 = 1e-4
TRAIN_RATIO = 0.8


def normalize_landmarks(values):
    wrist_x, wrist_y, wrist_z = values[0], values[1], values[2]
    normalized = []
    max_distance = 0.0

    for i in range(21):
        dx = values[(i * 3)] - wrist_x
        dy = values[(i * 3) + 1] - wrist_y
        dz = values[(i * 3) + 2] - wrist_z
        normalized.extend([dx, dy, dz])
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        max_distance = max(max_distance, distance)

    scale = max_distance if max_distance > 0 else 1.0
    return [value / scale for value in normalized]


def load_dataset(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        raise ValueError("dataset.csv does not contain any training rows")

    features = []
    labels = []
    for row in rows[1:]:
        if len(row) < 64:
            continue
        try:
            values = [float(row[i]) for i in range(63)]
        except ValueError:
            continue
        label = row[63].strip()
        if not label:
            continue
        features.append(normalize_landmarks(values))
        labels.append(label)

    if not features:
        raise ValueError("No valid rows found in dataset.csv")

    return np.array(features, dtype=np.float64), labels


def stratified_split(features, labels):
    grouped = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)

    train_indices = []
    test_indices = []
    rng = random.Random(SEED)
    for indices in grouped.values():
        rng.shuffle(indices)
        split_at = max(1, int(len(indices) * TRAIN_RATIO))
        if split_at >= len(indices):
            split_at = len(indices) - 1
        train_indices.extend(indices[:split_at])
        test_indices.extend(indices[split_at:])

    if not test_indices:
        test_indices = train_indices[-len(grouped):]
        train_indices = train_indices[:-len(grouped)]

    return (
        features[train_indices],
        [labels[i] for i in train_indices],
        features[test_indices],
        [labels[i] for i in test_indices],
    )


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def encode_labels(labels):
    classes = sorted(set(labels))
    index_by_label = {label: i for i, label in enumerate(classes)}
    encoded = np.array([index_by_label[label] for label in labels], dtype=np.int64)
    return encoded, classes


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def train_softmax_regression(train_x, train_y, num_classes):
    rng = np.random.default_rng(SEED)
    weights = rng.normal(0.0, 0.01, size=(train_x.shape[1], num_classes))
    bias = np.zeros(num_classes, dtype=np.float64)

    one_hot = np.eye(num_classes)[train_y]
    batch_size = min(32, len(train_x))

    for _ in range(EPOCHS):
        order = rng.permutation(len(train_x))
        shuffled_x = train_x[order]
        shuffled_y = one_hot[order]

        for start in range(0, len(train_x), batch_size):
            end = start + batch_size
            batch_x = shuffled_x[start:end]
            batch_y = shuffled_y[start:end]

            logits = batch_x @ weights + bias
            probs = softmax(logits)
            error = probs - batch_y

            grad_w = (batch_x.T @ error) / len(batch_x) + (L2 * weights)
            grad_b = error.mean(axis=0)

            weights -= LEARNING_RATE * grad_w
            bias -= LEARNING_RATE * grad_b

    return weights, bias


def accuracy(features, labels, weights, bias):
    logits = features @ weights + bias
    predictions = np.argmax(logits, axis=1)
    return float((predictions == labels).mean())


def main():
    features, labels = load_dataset(DATASET_PATH)
    train_x, train_labels, test_x, test_labels = stratified_split(features, labels)
    train_x, test_x, mean, std = standardize(train_x, test_x)
    train_y, classes = encode_labels(train_labels)
    test_y = np.array([classes.index(label) for label in test_labels], dtype=np.int64)

    weights, bias = train_softmax_regression(train_x, train_y, len(classes))

    payload = {
        "type": "softmax_regression",
        "input_size": int(train_x.shape[1]),
        "labels": classes,
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "metrics": {
            "train_accuracy": accuracy(train_x, train_y, weights, bias),
            "test_accuracy": accuracy(test_x, test_y, weights, bias),
            "sample_count": len(labels),
            "train_count": len(train_labels),
            "test_count": len(test_labels),
            "label_counts": Counter(labels),
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Model written to {OUTPUT_PATH}")
    print(json.dumps(payload["metrics"], indent=2, default=dict))


if __name__ == "__main__":
    main()
