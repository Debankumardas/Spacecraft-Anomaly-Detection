
"""
Analyze one-step persistence prediction errors for E-13 telemetry.

Training and validation errors are calculated separately.
The validation predictions use the final training observation
as their initial history.
"""

from pathlib import Path

import numpy as np

from src.data.create_chronological_split import chronological_split
from src.data.normalize_telemetry import normalize_train_validation
from src.models.persistence_baseline import predict_persistence


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHANNEL_ID = "E-13"
FEATURE_INDEX = 0
TRAIN_RATIO = 0.80


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nasa_smap_msl"
    / "train"
    / f"{CHANNEL_ID}.npy"
)


def calculate_absolute_errors(actual, predicted):
    """Calculate absolute prediction errors after validating inputs."""

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional.")

    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError("Inputs cannot be empty.")

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths must match.")

    if not np.isfinite(actual).all():
        raise ValueError("Actual values contain non-finite values.")

    if not np.isfinite(predicted).all():
        raise ValueError("Predicted values contain non-finite values.")

    return np.abs(actual - predicted)


def summarize_errors(errors):
    """Calculate descriptive statistics for absolute errors."""

    errors = np.asarray(errors, dtype=np.float64)

    if errors.ndim != 1 or len(errors) == 0:
        raise ValueError("Errors must be a non-empty one-dimensional array.")

    if not np.isfinite(errors).all():
        raise ValueError("Errors contain non-finite values.")

    percentiles = np.percentile(
        errors,
        [50, 75, 90, 95, 99],
    )

    return {
        "count": len(errors),
        "mean": np.mean(errors),
        "median": percentiles[0],
        "std": np.std(errors),
        "minimum": np.min(errors),
        "maximum": np.max(errors),
        "p75": percentiles[1],
        "p90": percentiles[2],
        "p95": percentiles[3],
        "p99": percentiles[4],
    }


def print_summary(title, summary):
    """Print a formatted error summary."""

    print(f"\n{title}")
    print("-" * 55)

    for name, value in summary.items():
        if name == "count":
            print(f"{name:12}: {value}")
        else:
            print(f"{name:12}: {value:.6f}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    telemetry = np.load(DATA_PATH, allow_pickle=False)

    if telemetry.ndim != 2:
        raise ValueError("Expected a two-dimensional telemetry array.")

    if not 0 <= FEATURE_INDEX < telemetry.shape[1]:
        raise ValueError("Selected feature index does not exist.")

    signal = telemetry[:, FEATURE_INDEX]

    training_data, validation_data, split_index = chronological_split(
        signal,
        train_ratio=TRAIN_RATIO,
    )

    normalized_train, normalized_validation, _ = (
        normalize_train_validation(training_data, validation_data)
    )

    # Training predictions use previous actual training observations.
    training_predictions = predict_persistence(
        training_data[:-1],
        training_data[1:],
    )

    training_errors = calculate_absolute_errors(
        training_data[1:],
        training_predictions,
    )

    # Validation starts with the final training observation.
    validation_predictions = predict_persistence(
        training_data[-1:],
        validation_data,
    )

    validation_errors = calculate_absolute_errors(
        validation_data,
        validation_predictions,
    )

    # Repeat the error analysis in normalized units.
    normalized_training_predictions = predict_persistence(
        normalized_train[:-1],
        normalized_train[1:],
    )

    normalized_training_errors = calculate_absolute_errors(
        normalized_train[1:],
        normalized_training_predictions,
    )

    normalized_validation_predictions = predict_persistence(
        normalized_train[-1:],
        normalized_validation,
    )

    normalized_validation_errors = calculate_absolute_errors(
        normalized_validation,
        normalized_validation_predictions,
    )

    training_summary = summarize_errors(training_errors)
    validation_summary = summarize_errors(validation_errors)

    normalized_training_summary = summarize_errors(
        normalized_training_errors
    )
    normalized_validation_summary = summarize_errors(
        normalized_validation_errors
    )

    print("=" * 65)
    print("PERSISTENCE PREDICTION-ERROR ANALYSIS")
    print("=" * 65)

    print(f"Channel: {CHANNEL_ID}")
    print(f"Feature: {FEATURE_INDEX}")
    print(f"Total observations: {len(signal)}")
    print(f"Training observations: {len(training_data)}")
    print(f"Validation observations: {len(validation_data)}")
    print(f"Chronological split index: {split_index}")

    print_summary("TRAINING ERRORS - ORIGINAL SCALE", training_summary)
    print_summary("VALIDATION ERRORS - ORIGINAL SCALE", validation_summary)

    print_summary(
        "TRAINING ERRORS - NORMALIZED SCALE",
        normalized_training_summary,
    )

    print_summary(
        "VALIDATION ERRORS - NORMALIZED SCALE",
        normalized_validation_summary,
    )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
