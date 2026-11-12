
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.create_chronological_split import chronological_split
from src.models.persistence_baseline import predict_persistence
from src.evaluation.analyze_thresholds import (
    calculate_candidate_thresholds,
    group_alert_events,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "E-13_train.npy"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "E-13_test.npy"
)

ANOMALY_LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nasa_smap_msl"
    / "labeled_anomalies.csv"
)

CHANNEL = "E-13"
FEATURE_INDEX = 0
TRAIN_RATIO = 0.80


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

def validate_telemetry_data(data, description):
    """Validate a multivariate telemetry array."""

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(
            f"{description} must be a 2D telemetry array."
        )

    if data.shape[0] < 2:
        raise ValueError(
            f"{description} must contain at least two observations."
        )

    if data.shape[1] <= FEATURE_INDEX:
        raise ValueError(
            f"{description} does not contain feature "
            f"{FEATURE_INDEX}."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"{description} contains NaN or infinite values."
        )

    return data


# --------------------------------------------------
# GROUND-TRUTH MASK
# --------------------------------------------------

def build_ground_truth_mask(intervals, total_samples):
    """
    Build a Boolean ground-truth mask.

    Interval boundaries are zero-based and inclusive.
    """

    if (
        not isinstance(total_samples, (int, np.integer))
        or isinstance(total_samples, (bool, np.bool_))
        or total_samples <= 0
    ):
        raise ValueError(
            "total_samples must be a positive integer."
        )

    if not isinstance(intervals, (list, tuple)):
        raise ValueError(
            "intervals must be a list or tuple."
        )

    mask = np.zeros(total_samples, dtype=bool)

    for interval in intervals:

        if not isinstance(
            interval,
            (list, tuple, np.ndarray),
        ):
            raise ValueError(
                "Each interval must contain two boundaries."
            )

        if len(interval) != 2:
            raise ValueError(
                "Each interval must contain exactly two boundaries."
            )

        start, end = interval

        if (
            not isinstance(start, (int, np.integer))
            or isinstance(start, (bool, np.bool_))
            or not isinstance(end, (int, np.integer))
            or isinstance(end, (bool, np.bool_))
        ):
            raise ValueError(
                "Interval boundaries must be integers."
            )

        if start < 0 or end < start:
            raise ValueError(
                "Interval boundaries are invalid."
            )

        if end >= total_samples:
            raise ValueError(
                "Interval exceeds the available samples."
            )

        # Inclusive end boundary.
        mask[start:end + 1] = True

    return mask


# --------------------------------------------------
# POINT-LEVEL EVALUATION
# --------------------------------------------------

def calculate_point_metrics(actual, predicted):
    """
    Calculate point-level classification metrics.

    Both inputs must be one-dimensional binary arrays.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError(
            "Actual and predicted masks must be 1D arrays."
        )

    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError(
            "Actual and predicted masks cannot be empty."
        )

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted masks must have equal lengths."
        )

    if not np.isfinite(actual).all():
        raise ValueError(
            "Actual mask contains NaN or infinite values."
        )

    if not np.isfinite(predicted).all():
        raise ValueError(
            "Predicted mask contains NaN or infinite values."
        )

    if not np.isin(actual, [0, 1, False, True]).all():
        raise ValueError(
            "Actual mask must contain only binary values."
        )

    if not np.isin(predicted, [0, 1, False, True]).all():
        raise ValueError(
            "Predicted mask must contain only binary values."
        )

    actual = actual.astype(bool)
    predicted = predicted.astype(bool)

    true_positive = int(
        np.sum(actual & predicted)
    )

    false_positive = int(
        np.sum(~actual & predicted)
    )

    true_negative = int(
        np.sum(~actual & ~predicted)
    )

    false_negative = int(
        np.sum(actual & ~predicted)
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0.0
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


# --------------------------------------------------
# EVENT-LEVEL EVALUATION
# --------------------------------------------------

def evaluate_event_detection(
    predicted_events,
    anomaly_intervals,
):
    """
    Evaluate event-level anomaly detection.

    A predicted event matches a ground-truth interval
    when they overlap by at least one sample.

    Matching is one-to-one. Each predicted event and
    each ground-truth interval can match only once.
    """

    if not isinstance(predicted_events, (list, tuple)):
        raise ValueError(
            "predicted_events must be a list or tuple."
        )

    if not isinstance(anomaly_intervals, (list, tuple)):
        raise ValueError(
            "anomaly_intervals must be a list or tuple."
        )

    def validate_interval(start, end, description):

        if (
            not isinstance(start, (int, np.integer))
            or isinstance(start, (bool, np.bool_))
            or not isinstance(end, (int, np.integer))
            or isinstance(end, (bool, np.bool_))
        ):
            raise ValueError(
                f"{description} boundaries must be integers."
            )

        if start < 0 or end < start:
            raise ValueError(
                f"{description} boundaries are invalid."
            )

    validated_predictions = []

    for event in predicted_events:

        if not isinstance(event, dict):
            raise ValueError(
                "Each predicted event must be a dictionary."
            )

        if (
            "start_index" not in event
            or "end_index" not in event
        ):
            raise ValueError(
                "Predicted events must contain start_index "
                "and end_index."
            )

        start = event["start_index"]
        end = event["end_index"]

        validate_interval(
            start,
            end,
            "Predicted event",
        )

        validated_predictions.append(
            (int(start), int(end))
        )

    validated_anomalies = []

    for interval in anomaly_intervals:

        if not isinstance(
            interval,
            (list, tuple, np.ndarray),
        ):
            raise ValueError(
                "Each anomaly interval must contain two boundaries."
            )

        if len(interval) != 2:
            raise ValueError(
                "Each anomaly interval must contain exactly "
                "two boundaries."
            )

        start, end = interval

        validate_interval(
            start,
            end,
            "Anomaly interval",
        )

        validated_anomalies.append(
            (int(start), int(end))
        )

    # Find all possible overlaps.
    possible_matches = []

    for predicted_start, predicted_end in validated_predictions:

        matches = []

        for anomaly_index, (
            anomaly_start,
            anomaly_end,
        ) in enumerate(validated_anomalies):

            overlaps = (
                predicted_start <= anomaly_end
                and predicted_end >= anomaly_start
            )

            if overlaps:
                matches.append(anomaly_index)

        possible_matches.append(matches)

    # Maximum one-to-one matching using augmenting paths.
    matched_predictions = {}

    def assign_match(prediction_index, visited):

        for anomaly_index in possible_matches[prediction_index]:

            if anomaly_index in visited:
                continue

            visited.add(anomaly_index)

            if (
                anomaly_index not in matched_predictions
                or assign_match(
                    matched_predictions[anomaly_index],
                    visited,
                )
            ):
                matched_predictions[anomaly_index] = (
                    prediction_index
                )

                return True

        return False

    for prediction_index in range(
        len(validated_predictions)
    ):
        assign_match(
            prediction_index,
            set(),
        )

    total_anomaly_intervals = len(validated_anomalies)

    detected_intervals = len(matched_predictions)

    missed_intervals = (
        total_anomaly_intervals - detected_intervals
    )

    unmatched_predicted_events = (
        len(validated_predictions) - detected_intervals
    )

    interval_recall = (
        detected_intervals / total_anomaly_intervals
        if total_anomaly_intervals > 0
        else 0.0
    )

    return {
        "total_anomaly_intervals": total_anomaly_intervals,
        "detected_intervals": detected_intervals,
        "missed_intervals": missed_intervals,
        "interval_recall": interval_recall,
        "unmatched_predicted_events": unmatched_predicted_events,
    }


# --------------------------------------------------
# LOAD ANOMALY ANNOTATIONS
# --------------------------------------------------

def load_anomaly_intervals():
    """Load and validate annotated anomaly intervals."""

    if not ANOMALY_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly labels file not found: "
            f"{ANOMALY_LABELS_PATH}"
        )

    labels = pd.read_csv(ANOMALY_LABELS_PATH)

    required_columns = {
        "chan_id",
        "anomaly_sequences",
    }

    missing_columns = (
        required_columns - set(labels.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    channel_rows = labels[
        labels["chan_id"] == CHANNEL
    ]

    if len(channel_rows) != 1:
        raise ValueError(
            f"Expected exactly one annotation row for "
            f"{CHANNEL}, found {len(channel_rows)}."
        )

    raw_intervals = channel_rows.iloc[0][
        "anomaly_sequences"
    ]

    if not isinstance(raw_intervals, str):
        raise ValueError(
            "Anomaly sequences must be stored as a string."
        )

    try:
        anomaly_intervals = ast.literal_eval(
            raw_intervals
        )

    except (ValueError, SyntaxError) as error:
        raise ValueError(
            "Could not parse anomaly interval annotations."
        ) from error

    if not isinstance(anomaly_intervals, list):
        raise ValueError(
            "Anomaly annotations must be a list."
        )

    return anomaly_intervals


# --------------------------------------------------
# CALIBRATION PREDICTION ERRORS
# --------------------------------------------------


def load_calibration_errors():
    """
    Calculate prediction errors on the held-out chronological
    calibration partition of the training telemetry.

    The scaler is fitted only on the first 80% of the original
    training data. These errors come from the remaining 20%.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}"
        )

    training_data = np.load(
        TRAIN_PATH,
        allow_pickle=False,
    )

    training_data = validate_telemetry_data(
        training_data,
        "Training data",
    )

    training_signal = training_data[:, FEATURE_INDEX]

    fit_signal, calibration_signal, split_index = chronological_split(
        training_signal,
        train_ratio=TRAIN_RATIO,
    )

    if len(fit_signal) != split_index:
        raise RuntimeError(
            "Chronological training split returned an unexpected size."
        )

    if len(fit_signal) < 1 or len(calibration_signal) == 0:
        raise ValueError(
            "Both fitting and calibration partitions must be non-empty."
        )

    # The final fitting observation provides context for the
    # first calibration prediction.
    predictions = predict_persistence(
        fit_signal[-1:],
        calibration_signal,
    )

    prediction_errors = np.abs(
        calibration_signal - predictions
    )

    prediction_errors = np.asarray(
        prediction_errors,
        dtype=np.float64,
    )

    if (
        prediction_errors.ndim != 1
        or len(prediction_errors) != len(calibration_signal)
        or len(prediction_errors) == 0
    ):
        raise ValueError(
            "Calibration errors have an unexpected shape."
        )

    if not np.isfinite(prediction_errors).all():
        raise ValueError(
            "Calibration errors contain NaN or infinite values."
        )

    return prediction_errors


# --------------------------------------------------
# TEST PREDICTION ERRORS
# --------------------------------------------------

def load_test_errors(training_data):
    """
    Calculate online one-step persistence errors on test data.

    The final training observation is used for the first
    test prediction.

    Each subsequent prediction uses the previous actual
    test observation.
    """

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_PATH}"
        )

    training_data = validate_telemetry_data(
        training_data,
        "Training data",
    )

    test_data = np.load(
        TEST_PATH,
        allow_pickle=False,
    )

    test_data = validate_telemetry_data(
        test_data,
        "Test data",
    )

    if training_data.shape[1] != test_data.shape[1]:
        raise ValueError(
            "Training and test data must have "
            "the same number of features."
        )

    training_signal = training_data[:, FEATURE_INDEX]

    test_signal = test_data[:, FEATURE_INDEX]

    predictions = predict_persistence(
        training_signal,
        test_signal,
    )

    prediction_errors = np.abs(
        test_signal - predictions
    )

    prediction_errors = np.asarray(
        prediction_errors,
        dtype=np.float64,
    )

    if (
        prediction_errors.ndim != 1
        or len(prediction_errors) != len(test_signal)
    ):
        raise ValueError(
            "Test prediction errors have an unexpected shape."
        )

    if not np.isfinite(prediction_errors).all():
        raise ValueError(
            "Test errors contain NaN or infinite values."
        )

    return prediction_errors


# --------------------------------------------------
# OUTPUT FORMATTING
# --------------------------------------------------

def print_point_metrics(metrics):

    print(
        f"TP: {metrics['true_positive']}, "
        f"FP: {metrics['false_positive']}, "
        f"TN: {metrics['true_negative']}, "
        f"FN: {metrics['false_negative']}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall: {metrics['recall']:.4f}"
    )

    print(
        f"F1-score: {metrics['f1']:.4f}"
    )

    print(
        f"Specificity: {metrics['specificity']:.4f}"
    )


def print_event_metrics(metrics):

    print(
        f"Total ground-truth intervals: "
        f"{metrics['total_anomaly_intervals']}"
    )

    print(
        f"Detected intervals: "
        f"{metrics['detected_intervals']}"
    )

    print(
        f"Missed intervals: "
        f"{metrics['missed_intervals']}"
    )

    print(
        f"Unmatched predicted events: "
        f"{metrics['unmatched_predicted_events']}"
    )

    print(
        f"Interval recall: "
        f"{metrics['interval_recall']:.4f}"
    )


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("=" * 65)

    print(
        "E-13 TEST-SET ANOMALY DETECTION EVALUATION"
    )

    print("=" * 65)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}"
        )

    training_data = np.load(
        TRAIN_PATH,
        allow_pickle=False,
    )

    training_data = validate_telemetry_data(
        training_data,
        "Training data",
    )

    # Calculate calibration and test prediction errors.
    calibration_errors = load_calibration_errors()

    test_errors = load_test_errors(
        training_data
    )

    # Load annotations.
    anomaly_intervals = load_anomaly_intervals()

    # Construct ground-truth mask.
    ground_truth_mask = build_ground_truth_mask(
        anomaly_intervals,
        total_samples=len(test_errors),
    )

    # Verify expected dataset dimensions.
    if len(test_errors) != 8640:
        raise ValueError(
            f"Expected 8640 test observations, "
            f"found {len(test_errors)}."
        )

    print(f"\nChannel: {CHANNEL}")

    print(f"Feature index: {FEATURE_INDEX}")

    print(
        f"Calibration errors: {len(calibration_errors)}"
    )

    print(
        f"Test observations: {len(test_errors)}"
    )

    print(
        f"Ground-truth anomaly samples: "
        f"{int(ground_truth_mask.sum())}"
    )

    print(
        f"Normal samples: "
        f"{int((~ground_truth_mask).sum())}"
    )

    print(
        f"Annotated anomaly intervals: "
        f"{len(anomaly_intervals)}"
    )

    # Calibrate thresholds using held-out training observations.
    print("\nThreshold calibration:")

    print(
        "Thresholds are calculated using the final 20% "
        "of the training telemetry."
    )

    candidate_thresholds = calculate_candidate_thresholds(
        calibration_errors
    )

    if not candidate_thresholds:
        raise ValueError(
            "No candidate thresholds were generated."
        )

    for threshold_name, threshold in candidate_thresholds.items():

        print("\n" + "-" * 65)

        print(
            f"Threshold: {threshold_name}"
        )

        print(
            f"Threshold value: {threshold:.6f}"
        )

        # Generate binary predictions.
        predicted_mask = test_errors >= threshold

        # Point-level evaluation.
        point_metrics = calculate_point_metrics(
            ground_truth_mask,
            predicted_mask,
        )

        print("\nPoint-level metrics:")

        print_point_metrics(
            point_metrics
        )

        # Event-level evaluation.
        predicted_events = group_alert_events(
            validation_errors=test_errors,
            threshold=threshold,
            start_index=0,
        )

        event_metrics = evaluate_event_detection(
            predicted_events,
            anomaly_intervals,
        )

        print("\nEvent-level metrics:")

        print_event_metrics(
            event_metrics
        )

    # Evaluation notes.
    print("\n" + "=" * 65)

    print("EVALUATION NOTES")

    print("=" * 65)

    print(
        "1. Thresholds are calibrated using prediction errors "
        "from the final 20% of training telemetry."
    )

    print(
        "2. Test anomaly labels are used only for final evaluation."
    )

    print(
        "3. Test predictions use online one-step persistence "
        "with previous actual observations."
    )

    print(
        "4. An event is considered detected when at least "
        "one predicted sample overlaps its annotated interval."
    )

    print(
        "5. Interval recall does not measure complete anomaly "
        "coverage or detection delay."
    )

    print(
        "6. Unmatched predicted events are alerts that do not "
        "receive a one-to-one ground-truth match."
    )

    print(
        "7. This persistence model is a baseline, not a "
        "production-ready anomaly detection model."
    )

    print("=" * 65)


if __name__ == "__main__":
    main()
