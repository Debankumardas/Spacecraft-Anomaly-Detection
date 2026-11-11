
"""
Normalize telemetry using training data statistics only.

The scaler is fitted exclusively on the training partition.
Validation data is transformed using the fitted training scaler.
"""

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.create_chronological_split import chronological_split


# --------------------------------------------------
# CONFIGURATION
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
# NORMALIZATION FUNCTION
# --------------------------------------------------

def normalize_train_validation(train_data, validation_data):
    """
    Fit a StandardScaler on training data only.

    Transform both training and validation data using
    the same fitted scaler.

    Returns:
        normalized_train
        normalized_validation
        scaler
    """

    train_data = np.asarray(train_data, dtype=np.float64)
    validation_data = np.asarray(validation_data, dtype=np.float64)

    if train_data.ndim != 1 or validation_data.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional.")

    if len(train_data) == 0 or len(validation_data) == 0:
        raise ValueError("Training and validation data cannot be empty.")

    if not np.isfinite(train_data).all():
        raise ValueError("Training data contains non-finite values.")

    if not np.isfinite(validation_data).all():
        raise ValueError("Validation data contains non-finite values.")

    # Scikit-learn expects a 2D array.
    train_2d = train_data.reshape(-1, 1)
    validation_2d = validation_data.reshape(-1, 1)

    # Fit ONLY on training observations.
    scaler = StandardScaler()
    scaler.fit(train_2d)

    # Transform both partitions using training statistics.
    normalized_train = scaler.transform(train_2d).ravel()
    normalized_validation = scaler.transform(validation_2d).ravel()

    return normalized_train, normalized_validation, scaler


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("E-13 TRAINING-ONLY TELEMETRY NORMALIZATION")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataset = np.load(DATA_PATH, allow_pickle=False)

    if dataset.ndim != 2:
        raise ValueError("Expected a two-dimensional telemetry array.")

    if not 0 <= FEATURE_INDEX < dataset.shape[1]:
        raise ValueError("Invalid feature index.")

    signal = dataset[:, FEATURE_INDEX]

    train_data, validation_data, split_index = chronological_split(
        signal,
        TRAIN_RATIO
    )

    normalized_train, normalized_validation, scaler = (
        normalize_train_validation(train_data, validation_data)
    )

    # --------------------------------------------------
    # INTEGRITY CHECKS
    # --------------------------------------------------

    assert len(normalized_train) == len(train_data)
    assert len(normalized_validation) == len(validation_data)

    assert np.isfinite(normalized_train).all()
    assert np.isfinite(normalized_validation).all()

    assert np.isclose(np.mean(normalized_train), 0.0, atol=1e-10)
    assert np.isclose(np.std(normalized_train), 1.0, atol=1e-10)

    assert np.isclose(scaler.mean_[0], np.mean(train_data))
    assert np.isclose(scaler.scale_[0], np.std(train_data))

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("\nDATA PARTITIONS")
    print("-" * 65)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(validation_data)}")
    print(f"Split index: {split_index}")

    print("\nTRAINING SCALER PARAMETERS")
    print("-" * 65)
    print(f"Training mean: {scaler.mean_[0]:.6f}")
    print(f"Training standard deviation: {scaler.scale_[0]:.6f}")

    print("\nNORMALIZED TRAINING DATA")
    print("-" * 65)
    print(f"Mean: {np.mean(normalized_train):.10f}")
    print(f"Standard deviation: {np.std(normalized_train):.10f}")
    print(f"Minimum: {np.min(normalized_train):.6f}")
    print(f"Maximum: {np.max(normalized_train):.6f}")

    print("\nNORMALIZED VALIDATION DATA")
    print("-" * 65)
    print(f"Mean: {np.mean(normalized_validation):.6f}")
    print(f"Standard deviation: {np.std(normalized_validation):.6f}")
    print(f"Minimum: {np.min(normalized_validation):.6f}")
    print(f"Maximum: {np.max(normalized_validation):.6f}")

    print("\nLEAKAGE PREVENTION")
    print("-" * 65)
    print("Scaler fitted on training data only: True")
    print("Validation transformed with training scaler: True")
    print("Validation data used during fitting: False")

    print("\n" + "=" * 65)
    print("NORMALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
