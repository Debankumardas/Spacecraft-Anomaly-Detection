from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nasa_smap_msl"
    / "train"
    / "E-13.npy"
)

MAX_LAG = 100


def calculate_autocorrelation(signal, max_lag):
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) < 2:
        raise ValueError("Signal must contain at least two samples.")

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite values.")

    if not isinstance(max_lag, int) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer.")

    if max_lag >= len(signal):
        raise ValueError("max_lag must be smaller than signal length.")

    centered = signal - np.mean(signal)
    denominator = np.dot(centered, centered)

    if denominator == 0:
        raise ValueError("Signal has zero variance.")

    correlations = []

    for lag in range(max_lag + 1):
        if lag == 0:
            correlation = 1.0
        else:
            correlation = (
                np.dot(centered[:-lag], centered[lag:])
                / denominator
            )

        correlations.append(correlation)

    return np.array(correlations)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    data = np.load(DATA_PATH, allow_pickle=False)

    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError(f"Unexpected dataset shape: {data.shape}")

    signal = data[:, 0].astype(np.float64)

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite values.")

    autocorrelation = calculate_autocorrelation(signal, MAX_LAG)

    differences = np.diff(signal)

    print("=" * 70)
    print("E-13 TEMPORAL STRUCTURE ANALYSIS")
    print("=" * 70)

    print(f"\nTraining samples: {len(signal)}")
    print(f"Feature: 0")
    print(f"Mean: {signal.mean():.6f}")
    print(f"Standard deviation: {signal.std():.6f}")

    print("\nFIRST DIFFERENCE STATISTICS")
    print(f"Mean: {differences.mean():.6f}")
    print(f"Standard deviation: {differences.std():.6f}")
    print(f"Minimum: {differences.min():.6f}")
    print(f"Maximum: {differences.max():.6f}")

    print("\nAUTOCORRELATION")
    print("-" * 50)

    for lag in [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100]:
        print(f"Lag {lag:3d}: {autocorrelation[lag]:.6f}")

    print("\nFIRST LAG BELOW 0.5")

    below = np.where(autocorrelation[1:] < 0.5)[0]

    if len(below):
        print(f"Lag: {below[0] + 1}")
    else:
        print("No crossing within the inspected lag range.")

    print("\nAnalysis completed. Training data only.")


if __name__ == "__main__":
    main()
