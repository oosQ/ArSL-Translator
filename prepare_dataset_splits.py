import csv
import json
import os
import random
from collections import Counter, defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_splits")
SEED = 7
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        raise ValueError("dataset.csv does not contain enough rows")

    header = rows[0]
    data_rows = [row for row in rows[1:] if len(row) >= 64 and row[63].strip()]
    return header, data_rows


def stratified_split(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[63].strip()].append(row)

    rng = random.Random(SEED)
    train_rows = []
    val_rows = []
    test_rows = []

    for label, label_rows in grouped.items():
        rng.shuffle(label_rows)
        count = len(label_rows)
        train_count = max(1, int(round(count * TRAIN_RATIO)))
        val_count = max(1, int(round(count * VAL_RATIO)))
        remaining = count - train_count - val_count

        if remaining <= 0:
            remaining = 1
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1

        test_count = remaining

        train_rows.extend(label_rows[:train_count])
        val_rows.extend(label_rows[train_count:train_count + val_count])
        test_rows.extend(label_rows[train_count + val_count:train_count + val_count + test_count])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_split(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def summarize(rows):
    return dict(sorted(Counter(row[63].strip() for row in rows).items()))


def main():
    header, rows = load_rows(DATASET_PATH)
    train_rows, val_rows, test_rows = stratified_split(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_split(os.path.join(OUTPUT_DIR, "train.csv"), header, train_rows)
    write_split(os.path.join(OUTPUT_DIR, "val.csv"), header, val_rows)
    write_split(os.path.join(OUTPUT_DIR, "test.csv"), header, test_rows)

    summary = {
        "seed": SEED,
        "source_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_label_counts": summarize(train_rows),
        "val_label_counts": summarize(val_rows),
        "test_label_counts": summarize(test_rows),
    }

    with open(os.path.join(OUTPUT_DIR, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
