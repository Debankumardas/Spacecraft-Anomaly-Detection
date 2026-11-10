from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "nasa_smap_msl"

CHANNEL = "E-1"


def main():
    train_path = DATA / "train" / f"{CHANNEL}.npy"
    test_path = DATA / "test" / f"{CHANNEL}.npy"
    labels_path = DATA / "labeled_anomalies.csv"

    train = np.load(train_path, allow_pickle=False)
    test = np.load(test_path, allow_pickle=False)
    labels = pd.read_csv(labels_path)

    channel_labels = labels[labels["chan_id"] == CHANNEL]

    if len(channel_labels) != 1:
        raise ValueError(
            f"Expected exactly one label row for {CHANNEL}, "
            f"found {len(channel_labels)}"
        )

    label = channel_labels.iloc[0]

    print("=" * 65)
    print(f"CHANNEL FEATURE INSPECTION: {CHANNEL}")
    print("=" * 65)

    print("\n1. DATASET DIMENSIONS")
    print(f"Training shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Data type: {train.dtype}")

    print("\n2. TRAINING FEATURE STATISTICS")
    print("Statistics below use training data only.")

    stats = pd.DataFrame({
        "feature": [f"feature_{i}" for i in range(train.shape[1])],
        "mean": np.mean(train, axis=0),
        "std": np.std(train, axis=0),
        "min": np.min(train, axis=0),
        "median": np.median(train, axis=0),
        "max": np.max(train, axis=0),
        "unique_values": [
            np.unique(train[:, i]).size
            for i in range(train.shape[1])
        ],
    })

    stats["near_constant"] = stats["std"] < 1e-8

    print(stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n3. FEATURE VARIABILITY")
    print(f"Total features: {train.shape[1]}")
    print(f"Near-constant features: {stats['near_constant'].sum()}")
    print(f"Features with non-zero variance: {(stats['std'] > 1e-8).sum()}")

    print("\n4. LABEL INFORMATION")
    print(f"Spacecraft: {label['spacecraft']}")
    print(f"Anomaly intervals: {label['anomaly_sequences']}")
    print(f"Anomaly classes: {label['class']}")
    print(f"Declared test values: {label['num_values']}")

    print("\n5. TRAINING DATA QUALITY")
    print(f"Non-finite training values: {np.size(train) - np.isfinite(train).sum()}")
    print(f"Non-finite test values: {np.size(test) - np.isfinite(test).sum()}")

    print("\nInspection completed. No files were modified.")


if __name__ == "__main__":
    main()
