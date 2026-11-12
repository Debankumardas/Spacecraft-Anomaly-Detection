
"""
Prepare NASA SMAP telemetry for anomaly detection.

The scaler is fitted exclusively on the chronological training
partition. The complete training and test arrays are transformed
using that same fitted scaler.

No test data is used during scaler fitting.
"""

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nasa_smap_msl"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_RAW_PATH = RAW_DATA_DIR / "train" / "E-13.npy"
TEST_RAW_PATH = RAW_DATA_DIR / "test" / "E-13.npy"

TRAIN_OUTPUT_PATH = PROCESSED_DATA_DIR / "E-13_train.npy"
TEST_OUTPUT_PATH = PROCESSED_DATA_DIR / "E-13_test.npy"
SCALER_OUTPUT_PATH = PROCESSED_DATA_DIR / "E-13_scaler.npz"

TRAIN_RATIO = 0.80


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

def validate_telemetry(data, description):
    """Validate a complete multivariate telemetry array."""

    try:
        data = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{description} must contain numeric values."
        ) from exc

    if data.ndim != 2:
        raise ValueError(
            f"{description} must be a 2D array."
        )

    if data.shape[0] < 2:
        raise ValueError(
            f"{description} must contain at least two observations."
        )

    if data.shape[1] == 0:
        raise ValueError(
            f"{description} must contain at least one feature."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"{description} contains NaN or infinite values."
        )

    return data


# --------------------------------------------------
# SPLIT VALIDATION
# --------------------------------------------------

def calculate_split_index(data_length, train_ratio):
    """Calculate a valid chronological training split index."""

    if (
        isinstance(train_ratio, (bool, np.bool_))
        or not isinstance(
            train_ratio,
            (int, float, np.integer, np.floating),
        )
    ):
        raise ValueError(
            "train_ratio must be a numeric value between 0 and 1."
        )

    if not np.isfinite(train_ratio) or not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be greater than 0 and less than 1."
        )

    split_index = int(data_length * train_ratio)

    if split_index == 0 or split_index >= data_length:
        raise ValueError(
            "Split must produce two non-empty partitions."
        )

    return split_index


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def prepare_telemetry_arrays(
    training_data,
    test_data,
    train_ratio=TRAIN_RATIO,
):
    """
    Normalize complete telemetry arrays.

    The scaler is fitted only on the chronological training
    partition. Both complete arrays are transformed with it.

    Returns:
        normalized_training
        normalized_test
        scaler
        split_index
    """

    training_data = validate_telemetry(
        training_data,
        "Training data",
    )

    test_data = validate_telemetry(
        test_data,
        "Test data",
    )

    if training_data.shape[1] != test_data.shape[1]:
        raise ValueError(
            "Training and test data must have the same "
            "number of features."
        )

    split_index = calculate_split_index(
        len(training_data),
        train_ratio,
    )

    # Fit exclusively on the chronological training partition.
    scaler = StandardScaler()

    scaler.fit(
        training_data[:split_index]
    )

    # Transform complete arrays using the fitted scaler.
    normalized_training = scaler.transform(
        training_data
    )

    normalized_test = scaler.transform(
        test_data
    )

    normalized_training = np.asarray(
        normalized_training,
        dtype=np.float64,
    )

    normalized_test = np.asarray(
        normalized_test,
        dtype=np.float64,
    )

    # --------------------------------------------------
    # INTEGRITY CHECKS
    # --------------------------------------------------

    if normalized_training.shape != training_data.shape:
        raise RuntimeError(
            "Normalized training shape changed unexpectedly."
        )

    if normalized_test.shape != test_data.shape:
        raise RuntimeError(
            "Normalized test shape changed unexpectedly."
        )

    if not np.isfinite(normalized_training).all():
        raise RuntimeError(
            "Normalized training data contains non-finite values."
        )

    if not np.isfinite(normalized_test).all():
        raise RuntimeError(
            "Normalized test data contains non-finite values."
        )

    if scaler.n_samples_seen_ != split_index:
        raise RuntimeError(
            "Scaler was fitted on an unexpected number of samples."
        )

    return (
        normalized_training,
        normalized_test,
        scaler,
        split_index,
    )


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("=" * 70)
    print("NASA SMAP E-13 TELEMETRY PREPROCESSING")
    print("=" * 70)

    # Validate source files.
    if not TRAIN_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw training data not found: {TRAIN_RAW_PATH}"
        )

    if not TEST_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw test data not found: {TEST_RAW_PATH}"
        )

    # Load original arrays.
    training_data = np.load(
        TRAIN_RAW_PATH,
        allow_pickle=False,
    )

    test_data = np.load(
        TEST_RAW_PATH,
        allow_pickle=False,
    )

    training_data = validate_telemetry(
        training_data,
        "Raw training data",
    )

    test_data = validate_telemetry(
        test_data,
        "Raw test data",
    )

    # Prepare normalized arrays.
    (
        normalized_training,
        normalized_test,
        scaler,
        split_index,
    ) = prepare_telemetry_arrays(
        training_data,
        test_data,
        TRAIN_RATIO,
    )

    # Create output directory.
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save processed telemetry.
    np.save(
        TRAIN_OUTPUT_PATH,
        normalized_training,
    )

    np.save(
        TEST_OUTPUT_PATH,
        normalized_test,
    )

    # Save fitted scaler parameters for reproducibility.
    np.savez_compressed(
        SCALER_OUTPUT_PATH,
        mean=scaler.mean_,
        scale=scaler.scale_,
        variance=scaler.var_,
        n_samples_seen=scaler.n_samples_seen_,
        feature_count=scaler.n_features_in_,
        train_ratio=TRAIN_RATIO,
        split_index=split_index,
    )

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("\nSOURCE DATA")
    print("-" * 70)
    print(f"Training shape: {training_data.shape}")
    print(f"Test shape: {test_data.shape}")

    print("\nTRAINING PARTITION")
    print("-" * 70)
    print(f"Scaler fitting samples: {split_index}")
    print(
        f"Remaining training samples: "
        f"{len(training_data) - split_index}"
    )

    print("\nSCALER")
    print("-" * 70)
    print(f"Features: {scaler.n_features_in_}")
    print(f"Samples used for fitting: {scaler.n_samples_seen_}")
    print("Scaler fitted on test data: False")

    print("\nPROCESSED DATA")
    print("-" * 70)
    print(f"Training shape: {normalized_training.shape}")
    print(f"Test shape: {normalized_test.shape}")
    print(
        f"Training partition mean: "
        f"{np.mean(normalized_training[:split_index], axis=0).mean():.10f}"
    )
    print(
        f"Training partition average std: "
        f"{np.std(normalized_training[:split_index], axis=0).mean():.10f}"
    )

    print("\nSAVED FILES")
    print("-" * 70)
    print(TRAIN_OUTPUT_PATH)
    print(TEST_OUTPUT_PATH)
    print(SCALER_OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("TELEMETRY PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
