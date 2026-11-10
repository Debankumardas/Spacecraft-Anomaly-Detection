from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "raw" / "nasa_smap_msl" / "train"


def main():
    records = []

    for path in sorted(TRAIN_DIR.glob("*.npy")):
        data = np.load(path, mmap_mode="r", allow_pickle=False)

        feature_unique_counts = [
            np.unique(data[:, i]).size
            for i in range(data.shape[1])
        ]

        feature_std = np.std(data, axis=0)

        records.append({
            "channel": path.stem,
            "time_steps": data.shape[0],
            "features": data.shape[1],
            "constant_features": int(np.sum(feature_std < 1e-8)),
            "binary_features": int(
                sum(count <= 2 for count in feature_unique_counts)
            ),
            "continuous_features": int(
                sum(count > 2 for count in feature_unique_counts)
            ),
            "median_unique_values": float(
                np.median(feature_unique_counts)
            ),
            "max_unique_values": int(
                max(feature_unique_counts)
            ),
        })

    results = pd.DataFrame(records)

    results = results.sort_values(
        by=["continuous_features", "median_unique_values", "time_steps"],
        ascending=False
    )

    print("=" * 85)
    print("TELEMETRY CHANNEL VARIABILITY PROFILE")
    print("=" * 85)

    print(f"Channels inspected: {len(results)}")

    print("\nTop 20 channels by feature variability:\n")
    print(results.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.1f}"
    ))

    print("\nOverall feature profile:")
    print(f"Channels with continuous features: "
          f"{(results['continuous_features'] > 0).sum()}")
    print(f"Channels with at least 5 continuous features: "
          f"{(results['continuous_features'] >= 5).sum()}")
    print(f"Channels with only binary/constant features: "
          f"{(results['continuous_features'] == 0).sum()}")

    print("\nProfiling completed. No files were modified.")


if __name__ == "__main__":
    main()
