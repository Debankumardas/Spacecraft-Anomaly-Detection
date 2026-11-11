
import numpy as np
import pytest

from src.models.persistence_baseline import predict_persistence


def test_predictions_use_previous_actual_observation():
    history = np.array([8, 9, 10], dtype=float)
    targets = np.array([12, 11, 15, 14], dtype=float)

    predictions = predict_persistence(history, targets)

    expected = np.array([10, 12, 11, 15], dtype=float)

    assert np.array_equal(predictions, expected)


def test_first_prediction_uses_last_training_observation():
    history = np.array([3, 5, 7], dtype=float)
    targets = np.array([9, 11], dtype=float)

    predictions = predict_persistence(history, targets)

    assert predictions[0] == 7


def test_prediction_count_matches_target_count():
    history = np.array([1, 2, 3], dtype=float)
    targets = np.array([4, 5, 6, 7], dtype=float)

    predictions = predict_persistence(history, targets)

    assert len(predictions) == len(targets)


def test_single_target_is_supported():
    history = np.array([1, 2, 3], dtype=float)
    targets = np.array([5], dtype=float)

    predictions = predict_persistence(history, targets)

    assert np.array_equal(predictions, np.array([3.0]))


def test_empty_history_is_rejected():
    with pytest.raises(ValueError):
        predict_persistence(np.array([]), np.array([1, 2]))


def test_empty_targets_are_rejected():
    with pytest.raises(ValueError):
        predict_persistence(np.array([1, 2]), np.array([]))


def test_two_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        predict_persistence(
            np.ones((3, 1)),
            np.ones(2)
        )


def test_non_finite_history_is_rejected():
    with pytest.raises(ValueError):
        predict_persistence(
            np.array([1, np.nan, 3]),
            np.array([4, 5])
        )


def test_non_finite_targets_are_rejected():
    with pytest.raises(ValueError):
        predict_persistence(
            np.array([1, 2, 3]),
            np.array([4, np.inf])
        )
