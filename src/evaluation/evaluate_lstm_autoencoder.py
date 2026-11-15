"""Train and evaluate the E-13 LSTM autoencoder detector."""

import time

import numpy as np
import torch

from src.data.sequence_generator import create_sequences
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.evaluation.analyze_thresholds import (
    calculate_candidate_thresholds,
    evaluate_threshold,
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


def prepare_data(sequence_length=30):
    """Create chronological training, validation, calibration and test sets."""

    train = np.load(
        "data/processed/E-13_train.npy",
        allow_pickle=False,
    )
    test = np.load(
        "data/processed/E-13_test.npy",
        allow_pickle=False,
    )

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("Telemetry data must be two-dimensional.")

    if train.shape[1] != test.shape[1]:
        raise ValueError("Training and test feature counts must match.")

    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Telemetry contains non-finite values.")

    fit_end = int(len(train) * 0.80)
    fitting_data = train[:fit_end]
    calibration_data = train[fit_end:]

    validation_end = int(len(fitting_data) * 0.80)
    model_training_data = fitting_data[:validation_end]
    validation_data = fitting_data[validation_end:]

    training_sequences = create_sequences(
        model_training_data,
        sequence_length,
    )

    validation_sequences = create_sequences(
        validation_data,
        sequence_length,
        context=model_training_data,
    )

    calibration_sequences = create_sequences(
        calibration_data,
        sequence_length,
        context=fitting_data,
    )

    test_sequences = create_sequences(
        test,
        sequence_length,
        context=train[-(sequence_length - 1):],
    )

    return (
        training_sequences,
        validation_sequences,
        calibration_sequences,
        test_sequences,
    )


def evaluate_lstm_scores(calibration_scores, test_scores):
    """Evaluate fixed calibration thresholds against test annotations."""

    if (
        not np.isfinite(calibration_scores).all()
        or not np.isfinite(test_scores).all()
    ):
        raise ValueError("Reconstruction scores must be finite.")
    thresholds = calculate_candidate_thresholds(calibration_scores)

    # Test annotations are loaded only after thresholds are fixed.
    anomaly_intervals = load_anomaly_intervals()
    actual_mask = build_ground_truth_mask(
        anomaly_intervals,
        len(test_scores),
    )

    print("\n" + "=" * 65)
    print("LSTM AUTOENCODER THRESHOLD EVALUATION")
    print("=" * 65)

    for name, threshold in thresholds.items():
        calibration_result = evaluate_threshold(
            calibration_scores,
            threshold,
        )

        predicted_mask = test_scores >= threshold

        point_metrics = calculate_point_metrics(
            actual_mask,
            predicted_mask,
        )

        predicted_events = group_alert_events(
            test_scores,
            threshold,
            start_index=0,
        )

        event_metrics = evaluate_event_detection(
            predicted_events,
            anomaly_intervals,
        )

        print(f"\n{'-' * 65}")
        print(f"Threshold: {name}")
        print(f"Threshold value: {threshold:.8f}")

        print("\nCalibration:")
        print(
            f"Alerts: {calibration_result['alert_count']} "
            f"({calibration_result['alert_percentage']:.2f}%)"
        )

        print("\nTest point-level metrics:")
        print_point_metrics(point_metrics)

        print("\nTest event-level metrics:")
        print_event_metrics(event_metrics)

        print(f"Total predicted alert events: {len(predicted_events)}")

def main():
    torch.set_num_threads(2)

    print("Preparing E-13 telemetry sequences...")

    (
        training_sequences,
        validation_sequences,
        calibration_sequences,
        test_sequences,
    ) = prepare_data()

    print("Training sequences:", training_sequences.shape)
    print("Validation sequences:", validation_sequences.shape)
    print("Calibration sequences:", calibration_sequences.shape)
    print("Test sequences:", test_sequences.shape)

    detector = LSTMAutoencoder(
        hidden_size=32,
        learning_rate=0.001,
        batch_size=64,
        max_epochs=30,
        patience=5,
        random_state=42,
    )

    print("\nTraining LSTM autoencoder...")
    start_time = time.perf_counter()

    detector.fit(
        training_sequences,
        validation_sequences,
    )

    training_time = time.perf_counter() - start_time

    print("\nTRAINING SUMMARY")
    print("Epochs completed:", len(detector.training_history_))
    print(
        "Best validation loss:",
        f"{min(item['validation_loss'] for item in detector.training_history_):.8f}",
    )
    print("Training time:", f"{training_time:.2f} seconds")

    print("\nGenerating reconstruction scores...")

    calibration_scores = detector.score_samples(
        calibration_sequences
    )
    test_scores = detector.score_samples(test_sequences)

    print("Calibration scores:", len(calibration_scores))
    print("Test scores:", len(test_scores))
    print("Calibration scores finite:", np.isfinite(calibration_scores).all())
    print("Test scores finite:", np.isfinite(test_scores).all())

    print("\nCalibration score mean:", f"{calibration_scores.mean():.8f}")
    print("Calibration score std:", f"{calibration_scores.std():.8f}")
    print("Test score mean:", f"{test_scores.mean():.8f}")
    print("Test score std:", f"{test_scores.std():.8f}")

    evaluate_lstm_scores(
        calibration_scores,
        test_scores,
    )


if __name__ == "__main__":
    main()
