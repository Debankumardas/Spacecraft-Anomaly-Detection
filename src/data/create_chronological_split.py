
"""
Create a chronological training/validation split for E-13 telemetry.

The split uses only the training dataset:
- First 80%: training
- Final 20%: validation

No random shuffling is performed.
"""

from pathlib import Path

import numpy as np


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nasa_smap_msl"
    / "train"
    / "E-13.npy"
)

FEATURE_INDEX = 0
TRAIN_RATIO = 0.80


# --------------------------------------------------
# CHRONOLOGICAL SPLITTING FUNCTION
# --------------------------------------------------

def chronological_split(signal, train_ratio=0.80):
    """
    Split a one-dimensional time series chronologically.

    Returns:
        train_data: Earlier observations.
        validation_data: Later observations.
        split_index: First validation sample index.
    """

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) < 2:
        raise ValueError("Signal must contain at least two samples.")

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite values.")

    if (
        not isinstance(train_ratio, (int, float))
        or isinstance(train_ratio, bool)
        or not np.isfinite(train_ratio)
        or not 0 < train_ratio < 1
    ):
        raise ValueError("train_ratio must be between 0 and 1.")

    split_index = int(len(signal) * train_ratio)

    if split_index == 0 or split_index >= len(signal):
        raise ValueError("Split must produce two non-empty partitions.")

    train_data = signal[:split_index]
    validation_data = signal[split_index:]

    return train_data, validation_data, split_index


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("E-13 CHRONOLOGICAL TRAINING / VALIDATION SPLIT")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    dataset = np.load(DATA_PATH, allow_pickle=False)

    if dataset.ndim != 2:
        raise ValueError("Expected a two-dimensional telemetry array.")

    if not 0 <= FEATURE_INDEX < dataset.shape[1]:
        raise ValueError("Selected feature index does not exist.")

    signal = dataset[:, FEATURE_INDEX]

    train_data, validation_data, split_index = chronological_split(
        signal,
        TRAIN_RATIO
    )

    # --------------------------------------------------
    # VALIDATION CHECKS
    # --------------------------------------------------

    assert len(train_data) + len(validation_data) == len(signal)

    assert np.array_equal(train_data, signal[:split_index])

    assert np.array_equal(
        validation_data,
        signal[split_index:]
    )

    assert split_index == len(train_data)

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print(f"\nDataset: E-13")
    print(f"Feature: {FEATURE_INDEX}")
    print(f"Original samples: {len(signal)}")
    print(f"Training ratio: {TRAIN_RATIO:.0%}")
    print(f"Validation ratio: {1 - TRAIN_RATIO:.0%}")

    print("\nPARTITION DETAILS")
    print("-" * 65)

    print(f"Training samples: {len(train_data)}")
    print(f"Training indices: 0 to {split_index - 1}")

    print(f"\nValidation samples: {len(validation_data)}")
    print(
        f"Validation indices: {split_index} "
        f"to {len(signal) - 1}"
    )

    print("\nBOUNDARY CHECK")
    print("-" * 65)

    print(f"Last training value: {train_data[-1]:.6f}")
    print(f"First validation value: {validation_data[0]:.6f}")

    print("\nDATA INTEGRITY")
    print("-" * 65)

    print(f"Training finite: {np.isfinite(train_data).all()}")
    print(
        f"Validation finite: "
        f"{np.isfinite(validation_data).all()}"
    )
    print(f"Total samples preserved: {len(train_data) + len(validation_data)}")
    print("Chronological order preserved: True")
    print("Random shuffling: False")
    print("Raw dataset modified: False")

    print("\n" + "=" * 65)
    print("CHRONOLOGICAL SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
