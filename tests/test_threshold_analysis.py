
import numpy as np
import pytest

from src.evaluation.analyze_thresholds import (
    calculate_candidate_thresholds,
    evaluate_threshold,
    group_alert_events,
)


def test_candidate_thresholds_are_calculated_from_training_errors():
    training_errors = np.array([1.0, 2.0, 3.0, 4.0])

    thresholds = calculate_candidate_thresholds(training_errors)

    assert set(thresholds) == {
        "95th percentile",
        "97.5th percentile",
        "99th percentile",
        "Mean + 3 standard deviations",
    }

    assert thresholds["95th percentile"] == pytest.approx(
        np.percentile(training_errors, 95)
    )

    assert thresholds["97.5th percentile"] == pytest.approx(
        np.percentile(training_errors, 97.5)
    )

    assert thresholds["99th percentile"] == pytest.approx(
        np.percentile(training_errors, 99)
    )

    assert thresholds["Mean + 3 standard deviations"] == pytest.approx(
        np.mean(training_errors) + 3 * np.std(training_errors)
    )


def test_candidate_thresholds_reject_empty_input():
    with pytest.raises(ValueError):
        calculate_candidate_thresholds(np.array([]))


def test_candidate_thresholds_reject_non_finite_values():
    with pytest.raises(ValueError):
        calculate_candidate_thresholds(
            np.array([1.0, np.nan, 2.0])
        )


def test_candidate_thresholds_reject_negative_errors():
    with pytest.raises(ValueError):
        calculate_candidate_thresholds(
            np.array([1.0, -0.5, 2.0])
        )


def test_evaluate_threshold_counts_alerts_correctly():
    errors = np.array([0.1, 0.7, 0.8, 0.2, 0.9])

    results = evaluate_threshold(errors, threshold=0.7)

    assert results["threshold"] == pytest.approx(0.7)
    assert results["alert_count"] == 3
    assert results["alert_percentage"] == pytest.approx(60.0)
    assert results["alerts_per_1000"] == pytest.approx(600.0)


def test_group_alert_events_correctly():
    errors = np.array([0.1, 0.8, 0.9, 0.1, 0.7, 0.1, 0.1, 1.2])

    events = group_alert_events(
        errors,
        threshold=0.7,
        start_index=100,
    )

    assert len(events) == 3

    assert events[0] == {
        "start_index": 101,
        "end_index": 102,
        "duration": 2,
        "alert_points": 2,
        "peak_error": pytest.approx(0.9),
        "mean_error": pytest.approx(0.85),
    }

    assert events[1]["start_index"] == 104
    assert events[1]["end_index"] == 104
    assert events[1]["duration"] == 1

    assert events[2]["start_index"] == 107
    assert events[2]["end_index"] == 107


def test_group_alert_events_returns_empty_list_without_alerts():
    errors = np.array([0.1, 0.2, 0.3])

    events = group_alert_events(errors, threshold=0.5)

    assert events == []


def test_group_alert_events_handles_all_alerts():
    errors = np.array([0.8, 0.9, 1.0])

    events = group_alert_events(errors, threshold=0.7)

    assert len(events) == 1
    assert events[0]["start_index"] == 0
    assert events[0]["end_index"] == 2
    assert events[0]["duration"] == 3
    assert events[0]["alert_points"] == 3


def test_group_alert_events_rejects_empty_input():
    with pytest.raises(ValueError):
        group_alert_events(np.array([]), threshold=0.5)


def test_group_alert_events_rejects_non_finite_values():
    with pytest.raises(ValueError):
        group_alert_events(
            np.array([0.1, np.inf, 0.8]),
            threshold=0.5,
        )


def test_group_alert_events_rejects_negative_errors():
    with pytest.raises(ValueError):
        group_alert_events(
            np.array([0.1, -0.2, 0.8]),
            threshold=0.5,
        )


def test_group_alert_events_rejects_negative_start_index():
    with pytest.raises(ValueError):
        group_alert_events(
            np.array([0.1, 0.8]),
            threshold=0.5,
            start_index=-1,
        )

@pytest.mark.parametrize(
    "threshold",
    ["invalid", None, True, np.nan, np.inf, -0.5],
)
def test_evaluate_threshold_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        evaluate_threshold(
            np.array([0.1, 0.8]),
            threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    ["invalid", None, True, np.nan, np.inf, -0.5],
)
def test_group_alert_events_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        group_alert_events(
            np.array([0.1, 0.8]),
            threshold,
        )