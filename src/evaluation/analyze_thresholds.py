"""
Training-calibrated threshold analysis for E-13 telemetry.

Candidate thresholds are calculated exclusively from training
prediction errors and then evaluated on validation errors.
"""

from pathlib import Path

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


def calculate_candidate_thresholds(training_errors):
    """Calculate candidate thresholds using training errors only."""

    training_errors = np.asarray(training_errors, dtype=np.float64)

    if training_errors.ndim != 1:
        raise ValueError("Training errors must be one-dimensional.")

    if len(training_errors) == 0:
        raise ValueError("Training errors cannot be empty.")

    if not np.isfinite(training_errors).all():
        raise ValueError("Training errors contain non-finite values.")

    if np.any(training_errors < 0):
        raise ValueError("Training errors cannot be negative.")

    thresholds = {
        "95th percentile": np.percentile(training_errors, 95),
        "97.5th percentile": np.percentile(training_errors, 97.5),
        "99th percentile": np.percentile(training_errors, 99),
        "Mean + 3 standard deviations": (
            np.mean(training_errors) + 3 * np.std(training_errors)
        ),
    }

    return thresholds

def validate_threshold(threshold):
    """Validate and return a finite, non-negative numeric threshold."""

    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.integer, np.floating))
    ):
        raise ValueError("Threshold must be a finite number.")

    if not np.isfinite(threshold):
        raise ValueError("Threshold must be a finite number.")

    if threshold < 0:
        raise ValueError("Threshold cannot be negative.")

    return float(threshold)

def evaluate_threshold(validation_errors, threshold):
    """Evaluate a fixed threshold against validation errors."""

    validation_errors = np.asarray(
        validation_errors,
        dtype=np.float64,
    )

    if validation_errors.ndim != 1:
        raise ValueError("Validation errors must be one-dimensional.")

    if len(validation_errors) == 0:
        raise ValueError("Validation errors cannot be empty.")

    if not np.isfinite(validation_errors).all():
        raise ValueError("Validation errors contain non-finite values.")

    if np.any(validation_errors < 0):
        raise ValueError("Validation errors cannot be negative.")

    threshold = validate_threshold(threshold)

    alerts = validation_errors >= threshold
    alert_count = int(np.sum(alerts))
    total_count = len(validation_errors)

    return {
        "threshold": float(threshold),
        "alert_count": alert_count,
        "alert_percentage": (alert_count / total_count) * 100,
        "alerts_per_1000": (alert_count / total_count) * 1000,
    }

def group_alert_events(validation_errors, threshold, start_index=0):
    """Group consecutive threshold-exceeding errors into events."""

    validation_errors = np.asarray(
        validation_errors,
        dtype=np.float64,
    )

    if validation_errors.ndim != 1:
        raise ValueError("Validation errors must be one-dimensional.")

    if len(validation_errors) == 0:
        raise ValueError("Validation errors cannot be empty.")

    if not np.isfinite(validation_errors).all():
        raise ValueError("Validation errors contain non-finite values.")

    if np.any(validation_errors < 0):
        raise ValueError("Validation errors cannot be negative.")

    threshold = validate_threshold(threshold)

    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, (int, np.integer))
        or start_index < 0
    ):
        raise ValueError("Start index must be a non-negative integer.")

    alert_indices = np.flatnonzero(
        validation_errors >= threshold
    )

    if len(alert_indices) == 0:
        return []

    # Separate alert indices wherever there is a gap.
    event_groups = np.split(
        alert_indices,
        np.where(np.diff(alert_indices) > 1)[0] + 1,
    )

    events = []

    for group in event_groups:
        event_errors = validation_errors[group]

        events.append({
            "start_index": int(group[0] + start_index),
            "end_index": int(group[-1] + start_index),
            "duration": int(group[-1] - group[0] + 1),
            "alert_points": int(len(group)),
            "peak_error": float(np.max(event_errors)),
            "mean_error": float(np.mean(event_errors)),
        })

    return events

def load_prediction_errors():
    """Load E-13 telemetry and calculate training and validation errors."""

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

    validation_predictions = predict_persistence(
        training_data[-1:],
        validation_data,
    )

    training_errors = calculate_absolute_errors(
        training_data[1:],
        training_predictions,
    )

    validation_errors = calculate_absolute_errors(
        validation_data,
        validation_predictions,
    )

    return training_errors, validation_errors, split_index



def main():
    training_errors, validation_errors, split_index = (
        load_prediction_errors()
    )

    # Calibrate thresholds using training errors only.
    candidate_thresholds = calculate_candidate_thresholds(
        training_errors
    )

    print("=" * 75)
    print("TRAINING-CALIBRATED ANOMALY THRESHOLD ANALYSIS")
    print("=" * 75)

    print(f"Channel: {CHANNEL_ID}")
    print(f"Feature: {FEATURE_INDEX}")
    print(f"Training errors: {len(training_errors)}")
    print(f"Validation errors: {len(validation_errors)}")
    print(f"Training / validation split index: {split_index}")

    print("\nCANDIDATE THRESHOLDS")
    print("-" * 75)

    for name, threshold in candidate_thresholds.items():
        print(f"{name:<35}: {threshold:.6f}")

    print("\nVALIDATION ALERT AND EVENT SUMMARY")
    print("-" * 95)

    print(
        f"{'Threshold method':<35}"
        f"{'Alerts':>9}"
        f"{'Alert %':>11}"
        f"{'Events':>9}"
        f"{'Isolated':>11}"
        f"{'Longest':>11}"
    )

    print("-" * 95)

    all_events = {}

    for name, threshold in candidate_thresholds.items():
        results = evaluate_threshold(
            validation_errors,
            threshold,
        )

        events = group_alert_events(
            validation_errors,
            threshold,
            start_index=split_index,
        )

        all_events[name] = events

        isolated_events = sum(
            event["duration"] == 1
            for event in events
        )

        longest_event = max(
            (event["duration"] for event in events),
            default=0,
        )

        print(
            f"{name:<35}"
            f"{results['alert_count']:>9}"
            f"{results['alert_percentage']:>10.2f}%"
            f"{len(events):>9}"
            f"{isolated_events:>11}"
            f"{longest_event:>11}"
        )

    print("\nDETAILED VALIDATION EVENTS")
    print("=" * 75)

    for name, events in all_events.items():
        print(f"\n{name}")
        print("-" * 75)

        if not events:
            print("No alert events detected.")
            continue

        print(
            f"{'Start':>8}"
            f"{'End':>8}"
            f"{'Duration':>10}"
            f"{'Alert pts':>11}"
            f"{'Peak error':>13}"
            f"{'Mean error':>13}"
        )

        for event in events:
            print(
                f"{event['start_index']:>8}"
                f"{event['end_index']:>8}"
                f"{event['duration']:>10}"
                f"{event['alert_points']:>11}"
                f"{event['peak_error']:>13.6f}"
                f"{event['mean_error']:>13.6f}"
            )

    print("\n" + "=" * 75)
    print("Analysis completed.")
    print(
        "Note: These are prediction-error alert events, "
        "not confirmed spacecraft anomalies."
    )


if __name__ == "__main__":
    main()
