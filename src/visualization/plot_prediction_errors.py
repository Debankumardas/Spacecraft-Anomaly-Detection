
"""
Visualize E-13 telemetry and persistence prediction errors.

Generates:
1. Telemetry and absolute prediction errors over time.
2. Training and validation error distributions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.create_chronological_split import chronological_split
from src.models.persistence_baseline import predict_persistence
from src.evaluation.analyze_prediction_errors import (
    calculate_absolute_errors,
)


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

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


def load_and_prepare_data():
    """Load telemetry and calculate training and validation errors."""

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

    training_predictions = predict_persistence(
        training_data[:-1],
        training_data[1:],
    )

    training_errors = calculate_absolute_errors(
        training_data[1:],
        training_predictions,
    )

    validation_predictions = predict_persistence(
        training_data[-1:],
        validation_data,
    )

    validation_errors = calculate_absolute_errors(
        validation_data,
        validation_predictions,
    )

    # Align error arrays with the original observation indices.
    training_error_indices = np.arange(1, len(training_data))

    validation_error_indices = np.arange(
        split_index,
        len(signal),
    )

    return {
        "signal": signal,
        "split_index": split_index,
        "training_errors": training_errors,
        "validation_errors": validation_errors,
        "training_error_indices": training_error_indices,
        "validation_error_indices": validation_error_indices,
    }


def plot_telemetry_and_errors(data):
    """Plot telemetry and absolute prediction errors over time."""

    signal = data["signal"]
    split_index = data["split_index"]

    training_indices = data["training_error_indices"]
    validation_indices = data["validation_error_indices"]

    training_errors = data["training_errors"]
    validation_errors = data["validation_errors"]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
    )

    axes[0].plot(
        np.arange(len(signal)),
        signal,
        linewidth=0.8,
        label="E-13 Feature 0",
    )

    axes[0].axvline(
        split_index,
        linestyle="--",
        linewidth=1.5,
        label="Training / Validation boundary",
    )

    axes[0].set_title("E-13 Telemetry Signal")
    axes[0].set_ylabel("Telemetry value")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        training_indices,
        training_errors,
        linewidth=0.7,
        label="Training absolute error",
    )

    axes[1].plot(
        validation_indices,
        validation_errors,
        linewidth=0.7,
        label="Validation absolute error",
    )

    axes[1].axvline(
        split_index,
        linestyle="--",
        linewidth=1.5,
        label="Training / Validation boundary",
    )

    axes[1].set_title("One-Step Persistence Absolute Prediction Error")
    axes[1].set_xlabel("Observation index")
    axes[1].set_ylabel("Absolute error")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()

    output_path = FIGURE_DIR / "e13_persistence_errors.png"

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_error_distributions(data):
    """Compare training and validation error distributions."""

    training_errors = data["training_errors"]
    validation_errors = data["validation_errors"]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    axes[0].hist(
        training_errors,
        bins=50,
        alpha=0.65,
        label="Training",
        density=True,
    )

    axes[0].hist(
        validation_errors,
        bins=50,
        alpha=0.65,
        label="Validation",
        density=True,
    )

    axes[0].set_title("Absolute Error Distribution")
    axes[0].set_xlabel("Absolute prediction error")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(
        training_errors,
        bins=50,
        alpha=0.65,
        label="Training",
        density=True,
    )

    axes[1].hist(
        validation_errors,
        bins=50,
        alpha=0.65,
        label="Validation",
        density=True,
    )

    axes[1].set_yscale("log")
    axes[1].set_title("Absolute Error Distribution (Log Density)")
    axes[1].set_xlabel("Absolute prediction error")
    axes[1].set_ylabel("Density (log scale)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()

    output_path = FIGURE_DIR / "e13_error_distribution.png"

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_and_prepare_data()

    telemetry_plot = plot_telemetry_and_errors(data)
    distribution_plot = plot_error_distributions(data)

    print("=" * 65)
    print("PREDICTION-ERROR VISUALIZATION")
    print("=" * 65)

    print(f"Channel: {CHANNEL_ID}")
    print(f"Feature: {FEATURE_INDEX}")
    print(f"Training error count: {len(data['training_errors'])}")
    print(f"Validation error count: {len(data['validation_errors'])}")

    print("\nGenerated figures:")

    print(f"1. {telemetry_plot}")
    print(f"2. {distribution_plot}")

    print("\nVisualization completed.")


if __name__ == "__main__":
    main()
