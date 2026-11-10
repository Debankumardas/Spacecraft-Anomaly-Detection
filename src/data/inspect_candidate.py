import ast
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "nasa_smap_msl"

CHANNEL = "G-6"


def main():
    train = np.load(
        DATA / "train" / f"{CHANNEL}.npy",
        allow_pickle=False
    )

    test = np.load(
        DATA / "test" / f"{CHANNEL}.npy",
        allow_pickle=False
    )

    labels = pd.read_csv(DATA / "labeled_anomalies.csv")
    channel_rows = labels[labels["chan_id"] == CHANNEL]

    if len(channel_rows) != 1:
        raise ValueError(
            f"Expected one label row for {CHANNEL}; found {len(channel_rows)}"
        )

    label = channel_rows.iloc[0]
    intervals = ast.literal_eval(label["anomaly_sequences"])

    print("=" * 70)
    print(f"DETAILED CANDIDATE INSPECTION: {CHANNEL}")
    print("=" * 70)

    print("\n1. DATA DIMENSIONS")
    print(f"Training: {train.shape}")
    print(f"Testing:  {test.shape}")

    print("\n2. FEATURE ANALYSIS")
    print("A feature is classified as variable if it has more than 2 unique values.")

    variable_features = []

    for i in range(train.shape[1]):
        train_feature = train[:, i]
        test_feature = test[:, i]

        unique_count = np.unique(train_feature).size

        if unique_count > 2:
            variable_features.append(i)

        print(
            f"Feature {i:02d} | "
            f"Train unique: {unique_count:5d} | "
            f"Train std: {np.std(train_feature):.6f} | "
            f"Test unique: {np.unique(test_feature).size:5d}"
        )

    print(f"\nVariable feature indices: {variable_features}")

    print("\n3. CONTINUOUS FEATURE STATISTICS")

    for i in variable_features:
        tr = train[:, i]
        te = test[:, i]

        print(f"\nFeature {i}")

        for name, values in [("TRAIN", tr), ("TEST", te)]:
            print(f"  {name}:")
            print(f"    Min:    {np.min(values):.6f}")
            print(f"    Max:    {np.max(values):.6f}")
            print(f"    Mean:   {np.mean(values):.6f}")
            print(f"    Std:    {np.std(values):.6f}")
            print(f"    Median: {np.median(values):.6f}")
            print(
                "    1%, 25%, 75%, 99% quantiles:",
                np.quantile(values, [0.01, 0.25, 0.75, 0.99])
            )

            print(
                "    First 15 values:",
                np.round(values[:15], 5).tolist()
            )

    print("\n4. LABEL INFORMATION")
    print(f"Spacecraft: {label['spacecraft']}")
    print(f"Anomaly intervals: {intervals}")
    print(f"Anomaly classes: {label['class']}")
    print(f"Declared test length: {label['num_values']}")
    print(f"Actual test length: {len(test)}")

    for start, end in intervals:
        print(
            f"  Interval [{start}, {end}): "
            f"{end - start} time steps"
        )

    print("\n5. DATA QUALITY")
    print(f"Training non-finite values: {np.size(train) - np.isfinite(train).sum()}")
    print(f"Test non-finite values: {np.size(test) - np.isfinite(test).sum()}")

    print("\nInspection completed. No files were modified.")


if __name__ == "__main__":
    main()
