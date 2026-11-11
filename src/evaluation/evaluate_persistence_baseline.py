
"""
Evaluate a one-step persistence forecasting baseline on E-13 telemetry.

The model predicts each observation using the immediately preceding
actual observation. Evaluation uses the chronological validation segment.
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


def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE between actual and predicted values."""

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    return mae, rmse


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    telemetry = np.load(DATA_PATH, allow_pickle=False)

    if telemetry.ndim != 2:
        raise ValueError("Expected a two-dimensional telemetry array.")

    if FEATURE_INDEX >= telemetry.shape[1]:
        raise ValueError("Selected feature index does not exist.")

    signal = telemetry[:, FEATURE_INDEX]

    training_data, validation_data, split_index = chronological_split(
        signal,
        train_ratio=TRAIN_RATIO,
    )

    normalized_train, normalized_validation, scaler = (
        normalize_train_validation(training_data, validation_data)
    )

    # Evaluate predictions in the original telemetry scale.
    raw_predictions = predict_persistence(
        training_data,
        validation_data,
    )

    raw_mae, raw_rmse = calculate_metrics(
        validation_data,
        raw_predictions,
    )

    # Also report errors in training-standard-deviation units.
    normalized_predictions = predict_persistence(
        normalized_train,
        normalized_validation,
    )

    normalized_mae, normalized_rmse = calculate_metrics(
        normalized_validation,
        normalized_predictions,
    )

    print("=" * 65)
    print("PERSISTENCE BASELINE EVALUATION")
    print("=" * 65)

    print(f"Channel: {CHANNEL_ID}")
    print(f"Feature: {FEATURE_INDEX}")
    print(f"Total observations: {len(signal)}")
    print(f"Training observations: {len(training_data)}")
    print(f"Validation observations: {len(validation_data)}")
    print(f"Chronological split index: {split_index}")

    print("\nORIGINAL TELEMETRY SCALE")
    print(f"MAE:  {raw_mae:.6f}")
    print(f"RMSE: {raw_rmse:.6f}")

    print("\nNORMALIZED SCALE")
    print(f"MAE:  {normalized_mae:.6f}")
    print(f"RMSE: {normalized_rmse:.6f}")

    print("\nSCALING INFORMATION")
    print(f"Training mean: {scaler.mean_[0]:.6f}")
    print(f"Training standard deviation: {scaler.scale_[0]:.6f}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
