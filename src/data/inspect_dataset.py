import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "nasa_smap_msl"

TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
LABEL_FILE = DATA_ROOT / "labeled_anomalies.csv"


def load_array(path):
    """Load a NumPy telemetry file without allowing pickled objects."""
    return np.load(path, allow_pickle=False)


def main():
    print("=" * 65)
    print("NASA SMAP / MSL DATASET INSPECTION")
    print("=" * 65)

    for path in (TRAIN_DIR, TEST_DIR, LABEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path not found: {path}")

    train_files = {p.stem: p for p in TRAIN_DIR.glob("*.npy")}
    test_files = {p.stem: p for p in TEST_DIR.glob("*.npy")}

    labels = pd.read_csv(LABEL_FILE)

    required_columns = {
        "chan_id",
        "spacecraft",
        "anomaly_sequences",
        "class",
        "num_values",
    }

    missing_columns = required_columns - set(labels.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    label_ids = set(labels["chan_id"].astype(str))
    train_ids = set(train_files)
    test_ids = set(test_files)

    print("\n1. FILE AND LABEL COUNTS")
    print(f"Training files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    print(f"Label rows: {len(labels)}")
    print(f"Unique label channel IDs: {len(label_ids)}")

    print("\n2. CHANNEL ID CONSISTENCY")
    print(f"Train channels missing test files: {sorted(train_ids - test_ids)}")
    print(f"Test channels missing train files: {sorted(test_ids - train_ids)}")
    print(f"Channels missing labels: {sorted((train_ids | test_ids) - label_ids)}")
    print(f"Labels without matching telemetry: {sorted(label_ids - (train_ids & test_ids))}")

    duplicate_labels = labels[labels["chan_id"].duplicated(keep=False)]
    print(f"Duplicate label rows: {len(duplicate_labels)}")

    issues = []
    records = []

    for _, row in labels.iterrows():
        channel = str(row["chan_id"])

        if channel not in train_files or channel not in test_files:
            issues.append(f"{channel}: missing train or test file")
            continue

        try:
            train = load_array(train_files[channel])
            test = load_array(test_files[channel])

            if train.size == 0 or test.size == 0:
                issues.append(f"{channel}: empty telemetry array")

            if not np.issubdtype(train.dtype, np.number):
                issues.append(f"{channel}: training array is not numeric")

            if not np.issubdtype(test.dtype, np.number):
                issues.append(f"{channel}: test array is not numeric")

            train_nonfinite = (
                int(np.size(train) - np.isfinite(train).sum())
                if np.issubdtype(train.dtype, np.number)
                else -1
            )
            test_nonfinite = (
                int(np.size(test) - np.isfinite(test).sum())
                if np.issubdtype(test.dtype, np.number)
                else -1
            )

            if train_nonfinite > 0:
                issues.append(f"{channel}: {train_nonfinite} non-finite training values")

            if test_nonfinite > 0:
                issues.append(f"{channel}: {test_nonfinite} non-finite test values")

            try:
                intervals = ast.literal_eval(row["anomaly_sequences"])
            except (ValueError, SyntaxError, TypeError):
                intervals = []
                issues.append(f"{channel}: could not parse anomaly intervals")

            if not isinstance(intervals, list):
                issues.append(f"{channel}: anomaly intervals are not a list")
                intervals = []

            test_length = test.shape[0] if test.ndim > 0 else 0
            invalid_intervals = 0

            for interval in intervals:
                if (
                    not isinstance(interval, (list, tuple))
                    or len(interval) != 2
                    or not all(isinstance(v, int) for v in interval)
                ):
                    invalid_intervals += 1
                    continue

                start, end = interval

                if start < 0 or start >= end or end > test_length:
                    invalid_intervals += 1

            if invalid_intervals:
                issues.append(
                    f"{channel}: {invalid_intervals} invalid/out-of-range intervals"
                )

            class_names = re.findall(r"contextual|point", str(row["class"]))
            if len(class_names) != len(intervals):
                issues.append(
                    f"{channel}: interval count ({len(intervals)}) "
                    f"does not match class count ({len(class_names)})"
                )

            try:
                declared_values = int(row["num_values"])
                if declared_values != test_length:
                    issues.append(
                        f"{channel}: CSV num_values={declared_values}, "
                        f"test length={test_length}"
                    )
            except (ValueError, TypeError):
                issues.append(f"{channel}: invalid num_values field")

            records.append({
                "channel": channel,
                "spacecraft": row["spacecraft"],
                "train_shape": str(train.shape),
                "test_shape": str(test.shape),
                "train_dtype": str(train.dtype),
                "test_dtype": str(test.dtype),
                "train_nonfinite": train_nonfinite,
                "test_nonfinite": test_nonfinite,
                "anomaly_intervals": len(intervals),
                "anomaly_classes": ", ".join(class_names),
                "test_min": float(np.min(test)) if test.size else float("nan"),
                "test_max": float(np.max(test)) if test.size else float("nan"),
            })

        except Exception as exc:
            issues.append(f"{channel}: inspection failed: {type(exc).__name__}: {exc}")

    report = pd.DataFrame(records)

    print("\n3. SPACECRAFT DISTRIBUTION")
    print(labels["spacecraft"].value_counts().to_string())

    print("\n4. ARRAY INSPECTION SUMMARY")
    print(f"Successfully inspected channels: {len(records)}")

    if not report.empty:
        print("\nTraining array shapes:")
        print(report["train_shape"].value_counts().to_string())

        print("\nTest array shapes:")
        print(report["test_shape"].value_counts().to_string())

        print("\nData types:")
        print(report["test_dtype"].value_counts().to_string())

        print("\nFirst 12 channels:")
        print(
            report[
                [
                    "channel",
                    "spacecraft",
                    "train_shape",
                    "test_shape",
                    "anomaly_intervals",
                    "anomaly_classes",
                ]
            ].head(12).to_string(index=False)
        )

    print("\n5. DATA QUALITY ISSUES")
    print(f"Total issues detected: {len(issues)}")

    if issues:
        for issue in issues:
            print(f"- {issue}")
    else:
        print("No structural or basic data-quality issues detected.")

    print("\nInspection completed. No dataset files were modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
