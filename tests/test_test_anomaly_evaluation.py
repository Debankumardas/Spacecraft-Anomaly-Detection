
import numpy as np
import pytest

from src.evaluation import evaluate_test_anomalies as evaluator

from src.evaluation.evaluate_test_anomalies import (
    build_ground_truth_mask,
    calculate_point_metrics,
    evaluate_event_detection,
    load_calibration_errors,
)


# --------------------------------------------------
# Ground-truth mask tests
# --------------------------------------------------


def test_ground_truth_mask_uses_inclusive_intervals():
    mask = build_ground_truth_mask(
        intervals=[[2, 4]],
        total_samples=7,
    )

    expected = np.array(
        [False, False, True, True, True, False, False]
    )

    np.testing.assert_array_equal(mask, expected)


def test_ground_truth_mask_handles_multiple_intervals():
    mask = build_ground_truth_mask(
        intervals=[[1, 2], [5, 6]],
        total_samples=8,
    )

    expected = np.array(
        [False, True, True, False, False, True, True, False]
    )

    np.testing.assert_array_equal(mask, expected)


def test_ground_truth_mask_handles_overlapping_intervals():
    mask = build_ground_truth_mask(
        intervals=[[1, 4], [3, 5]],
        total_samples=7,
    )

    expected = np.array(
        [False, True, True, True, True, True, False]
    )

    np.testing.assert_array_equal(mask, expected)


def test_ground_truth_mask_handles_empty_intervals():
    mask = build_ground_truth_mask(
        intervals=[],
        total_samples=5,
    )

    assert not mask.any()


@pytest.mark.parametrize(
    "intervals",
    [
        [[-1, 2]],
        [[2, 7]],
        [[4, 2]],
        [[1]],
        [[1, 2, 3]],
        ["invalid"],
        [[1.5, 3]],
        [[True, 3]],
    ],
)
def test_ground_truth_mask_rejects_invalid_intervals(intervals):
    with pytest.raises(ValueError):
        build_ground_truth_mask(intervals, total_samples=6)


@pytest.mark.parametrize(
    "total_samples",
    [0, -1, 2.5, True, "5"],
)
def test_ground_truth_mask_rejects_invalid_sample_counts(total_samples):
    with pytest.raises(ValueError):
        build_ground_truth_mask([], total_samples)


def test_ground_truth_mask_rejects_invalid_interval_container():
    with pytest.raises(ValueError):
        build_ground_truth_mask("[1, 2]", total_samples=5)


# --------------------------------------------------
# Point-level metric tests
# --------------------------------------------------


def test_point_metrics_calculate_perfect_predictions():
    actual = np.array([1, 1, 0, 0])
    predicted = np.array([1, 1, 0, 0])

    metrics = calculate_point_metrics(actual, predicted)

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["true_negative"] == 2
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)


def test_point_metrics_calculate_mixed_predictions():
    actual = np.array([1, 1, 0, 0])
    predicted = np.array([1, 0, 1, 0])

    metrics = calculate_point_metrics(actual, predicted)

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)


def test_point_metrics_handle_no_predicted_alerts():
    actual = np.array([1, 1, 0, 0])
    predicted = np.array([0, 0, 0, 0])

    metrics = calculate_point_metrics(actual, predicted)

    assert metrics["true_positive"] == 0
    assert metrics["false_negative"] == 2
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_point_metrics_handle_all_predicted_alerts():
    actual = np.array([1, 0, 0, 0])
    predicted = np.array([1, 1, 1, 1])

    metrics = calculate_point_metrics(actual, predicted)

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 3
    assert metrics["true_negative"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == pytest.approx(0.25)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(0.0)


def test_point_metrics_handle_no_actual_anomalies():
    actual = np.array([0, 0, 0])
    predicted = np.array([0, 1, 0])

    metrics = calculate_point_metrics(actual, predicted)

    assert metrics["true_positive"] == 0
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 2
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


@pytest.mark.parametrize(
    "actual, predicted",
    [
        (np.array([]), np.array([])),
        (np.array([0, 1]), np.array([0])),
        (np.array([[0, 1]]), np.array([[0, 1]])),
        (np.array([0, 2]), np.array([0, 1])),
        (np.array([0, np.nan]), np.array([0, 1])),
        (np.array([0, 1]), np.array([0, np.inf])),
    ],
)
def test_point_metrics_reject_invalid_masks(actual, predicted):
    with pytest.raises(ValueError):
        calculate_point_metrics(actual, predicted)


# --------------------------------------------------
# Event-level evaluation tests
# --------------------------------------------------


def test_event_detection_identifies_all_anomaly_intervals():
    predicted_events = [
        {"start_index": 10, "end_index": 12},
        {"start_index": 30, "end_index": 31},
    ]

    anomaly_intervals = [[10, 15], [28, 35]]

    results = evaluate_event_detection(
        predicted_events,
        anomaly_intervals,
    )

    assert results["total_anomaly_intervals"] == 2
    assert results["detected_intervals"] == 2
    assert results["missed_intervals"] == 0
    assert results["interval_recall"] == pytest.approx(1.0)
    assert results["unmatched_predicted_events"] == 0


def test_event_detection_identifies_missed_intervals():
    predicted_events = [
        {"start_index": 10, "end_index": 12},
    ]

    anomaly_intervals = [[10, 15], [30, 35]]

    results = evaluate_event_detection(
        predicted_events,
        anomaly_intervals,
    )

    assert results["detected_intervals"] == 1
    assert results["missed_intervals"] == 1
    assert results["interval_recall"] == pytest.approx(0.5)


def test_event_detection_identifies_false_alert_events():
    predicted_events = [
        {"start_index": 10, "end_index": 12},
        {"start_index": 50, "end_index": 52},
    ]

    anomaly_intervals = [[10, 15]]

    results = evaluate_event_detection(
        predicted_events,
        anomaly_intervals,
    )

    assert results["detected_intervals"] == 1
    assert results["unmatched_predicted_events"] == 1


def test_event_detection_handles_no_predicted_events():
    results = evaluate_event_detection(
        predicted_events=[],
        anomaly_intervals=[[10, 15]],
    )

    assert results["detected_intervals"] == 0
    assert results["missed_intervals"] == 1
    assert results["interval_recall"] == 0.0
    assert results["unmatched_predicted_events"] == 0


def test_event_detection_handles_no_anomaly_intervals():
    results = evaluate_event_detection(
        predicted_events=[
            {"start_index": 10, "end_index": 12},
        ],
        anomaly_intervals=[],
    )

    assert results["total_anomaly_intervals"] == 0
    assert results["detected_intervals"] == 0
    assert results["missed_intervals"] == 0
    assert results["interval_recall"] == 0.0
    assert results["unmatched_predicted_events"] == 1


def test_event_detection_matches_each_anomaly_interval_only_once():
    predicted_events = [
        {"start_index": 10, "end_index": 25},
    ]

    anomaly_intervals = [
        [10, 15],
        [20, 25],
    ]

    results = evaluate_event_detection(
        predicted_events,
        anomaly_intervals,
    )

    assert results["detected_intervals"] == 1
    assert results["missed_intervals"] == 1
    assert results["unmatched_predicted_events"] == 0


def test_event_detection_matches_each_predicted_event_only_once():
    predicted_events = [
        {"start_index": 10, "end_index": 12},
        {"start_index": 13, "end_index": 15},
    ]

    anomaly_intervals = [
        [10, 15],
    ]

    results = evaluate_event_detection(
        predicted_events,
        anomaly_intervals,
    )

    assert results["detected_intervals"] == 1
    assert results["missed_intervals"] == 0
    assert results["unmatched_predicted_events"] == 1

@pytest.mark.parametrize(
    "predicted_events",
    [
        "invalid",
        [None],
        [{"start_index": 10}],
        [{"start_index": 10, "end_index": 12.5}],
        [{"start_index": 10, "end_index": 5}],
        [{"start_index": -1, "end_index": 5}],
        [{"start_index": True, "end_index": 5}],
    ],
)
def test_event_detection_rejects_invalid_predicted_events(
    predicted_events,
):
    with pytest.raises(ValueError):
        evaluate_event_detection(
            predicted_events,
            [[10, 15]],
        )


@pytest.mark.parametrize(
    "anomaly_intervals",
    [
        "invalid",
        [[10]],
        [[10, 15, 20]],
        [[10.5, 15]],
        [[15, 10]],
        [[-1, 10]],
        [[True, 15]],
    ],
)
def test_event_detection_rejects_invalid_anomaly_intervals(
    anomaly_intervals,
):
    with pytest.raises(ValueError):
        evaluate_event_detection(
            [{"start_index": 10, "end_index": 12}],
            anomaly_intervals,
        )


# --------------------------------------------------
# Calibration error tests
# --------------------------------------------------


def test_calibration_errors_use_held_out_training_partition(
    tmp_path,
    monkeypatch,
):
    training_data = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [100.0],
            [110.0],
        ]
    )

    training_path = tmp_path / "training.npy"

    np.save(training_path, training_data)

    monkeypatch.setattr(
        evaluator,
        "TRAIN_PATH",
        training_path,
    )

    monkeypatch.setattr(
        evaluator,
        "TRAIN_RATIO",
        0.8,
    )

    errors = load_calibration_errors()

    # First calibration prediction: 7.0 -> 100.0
    # Second calibration prediction: 100.0 -> 110.0
    expected_errors = np.array([93.0, 10.0])

    np.testing.assert_allclose(
        errors,
        expected_errors,
    )


def test_calibration_errors_include_first_held_out_observation(
    tmp_path,
    monkeypatch,
):
    training_data = np.arange(
        10,
        dtype=np.float64,
    ).reshape(-1, 1)

    training_path = tmp_path / "training.npy"

    np.save(training_path, training_data)

    monkeypatch.setattr(
        evaluator,
        "TRAIN_PATH",
        training_path,
    )

    monkeypatch.setattr(
        evaluator,
        "TRAIN_RATIO",
        0.8,
    )

    errors = load_calibration_errors()

    assert len(errors) == 2

    np.testing.assert_allclose(
        errors,
        np.array([1.0, 1.0]),
    )
