from pathlib import Path

import numpy as np

from src.models.isolation_forest_detector import (
    IsolationForestDetector,
)

from src.evaluation.analyze_thresholds import (
    calculate_candidate_thresholds,
    group_alert_events,
)

from src.evaluation.evaluate_test_anomalies import (
    build_ground_truth_mask,
    calculate_point_metrics,
    evaluate_event_detection,
    load_anomaly_intervals,
    print_point_metrics,
    print_event_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "E-13_train.npy"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "E-13_test.npy"

TRAIN_RATIO = 0.80
RANDOM_STATE = 42
N_ESTIMATORS = 100


def main():

    print("=" * 75)
    print("ISOLATION FOREST: E-13 TEST-SET EVALUATION")
    print("=" * 75)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}")

    training_data = np.load(TRAIN_PATH, allow_pickle=False)
    test_data = np.load(TEST_PATH, allow_pickle=False)

    if training_data.ndim != 2 or test_data.ndim != 2:
        raise ValueError("Training and test data must be two-dimensional.")

    if training_data.shape[1] != test_data.shape[1]:
        raise ValueError("Training and test feature counts do not match.")

    if not np.isfinite(training_data).all():
        raise ValueError("Training data contains non-finite values.")

    if not np.isfinite(test_data).all():
        raise ValueError("Test data contains non-finite values.")

    split_index = int(len(training_data) * TRAIN_RATIO)

    if split_index <= 0 or split_index >= len(training_data):
        raise ValueError("Invalid chronological training split.")

    fitting_data = training_data[:split_index]
    calibration_data = training_data[split_index:]

    print(f"\nTraining observations: {len(training_data)}")
    print(f"Model fitting observations: {len(fitting_data)}")
    print(f"Calibration observations: {len(calibration_data)}")
    print(f"Test observations: {len(test_data)}")
    print(f"Original features: {training_data.shape[1]}")

    # Fit the detector exclusively on the first 80% of training data.
    detector = IsolationForestDetector(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    detector.fit(fitting_data)

    print(f"Features retained: {int(detector.feature_mask_.sum())}")

    # Calibration and test scores are generated independently.
    calibration_scores = detector.score_samples(calibration_data)
    test_scores = detector.score_samples(test_data)

    if not np.isfinite(calibration_scores).all():
        raise RuntimeError("Calibration scores contain non-finite values.")

    if not np.isfinite(test_scores).all():
        raise RuntimeError("Test scores contain non-finite values.")

    # Thresholds use calibration scores only.
    candidate_thresholds = calculate_candidate_thresholds(
        calibration_scores
    )

    anomaly_intervals = load_anomaly_intervals()

    ground_truth_mask = build_ground_truth_mask(
        anomaly_intervals,
        total_samples=len(test_scores),
    )

    print("\nCALIBRATION SUMMARY")
    print("-" * 75)
    print(f"Calibration score minimum: {calibration_scores.min():.6f}")
    print(f"Calibration score maximum: {calibration_scores.max():.6f}")
    print(f"Calibration score mean: {calibration_scores.mean():.6f}")
    print(f"Calibration score std: {calibration_scores.std():.6f}")

    print("\nTEST DATA")
    print("-" * 75)
    print(f"Ground-truth anomaly samples: {int(ground_truth_mask.sum())}")
    print(f"Normal samples: {int((~ground_truth_mask).sum())}")
    print(f"Annotated anomaly intervals: {len(anomaly_intervals)}")

    print("\nTHRESHOLD EVALUATION")
    print("=" * 75)

    for threshold_name, threshold in candidate_thresholds.items():

        print("\n" + "-" * 75)
        print(f"Threshold method: {threshold_name}")
        print(f"Threshold value: {threshold:.6f}")

        predicted_mask = test_scores >= threshold

        point_metrics = calculate_point_metrics(
            ground_truth_mask,
            predicted_mask,
        )

        print("\nPoint-level metrics:")
        print_point_metrics(point_metrics)

        predicted_events = group_alert_events(
            validation_errors=test_scores,
            threshold=threshold,
            start_index=0,
        )

        event_metrics = evaluate_event_detection(
            predicted_events,
            anomaly_intervals,
        )

        print("\nEvent-level metrics:")
        print_event_metrics(event_metrics)

        print(f"Predicted anomaly samples: {int(predicted_mask.sum())}")
        print(f"Predicted anomaly percentage: {predicted_mask.mean() * 100:.3f}%")

    print("\n" + "=" * 75)
    print("EVALUATION NOTES")
    print("=" * 75)
    print("1. Isolation Forest was fitted on the first 80% of training data.")
    print("2. Thresholds were calibrated using the final 20% of training data.")
    print("3. Test anomaly labels were used only for final evaluation.")
    print("4. Higher anomaly scores indicate more unusual observations.")
    print("5. Event matching uses the existing one-to-one overlap method.")
    print("6. Calibration data may contain anomalies; this can affect thresholds.")
    print("7. This is a multivariate point-anomaly detector, not a temporal model.")
    print("=" * 75)


if __name__ == "__main__":
    main()
