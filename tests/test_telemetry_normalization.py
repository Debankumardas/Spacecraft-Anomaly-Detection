
import numpy as np
import pytest

from src.data.normalize_telemetry import normalize_train_validation


def test_training_data_is_standardized():
    train = np.array([1, 2, 3, 4, 5], dtype=float)
    validation = np.array([6, 7], dtype=float)

    normalized_train, _, scaler = normalize_train_validation(
        train, validation
    )

    assert np.isclose(np.mean(normalized_train), 0.0)
    assert np.isclose(np.std(normalized_train), 1.0)
    assert np.isclose(scaler.mean_[0], np.mean(train))


def test_scaler_uses_training_statistics_only():
    train = np.array([1, 2, 3, 4, 5], dtype=float)
    validation = np.array([6, 7], dtype=float)

    _, _, scaler = normalize_train_validation(train, validation)

    assert np.isclose(scaler.mean_[0], 3.0)
    assert np.isclose(scaler.scale_[0], np.std(train))


def test_extreme_validation_values_do_not_change_scaler():
    train = np.array([1, 2, 3, 4, 5], dtype=float)

    validation_a = np.array([6, 7], dtype=float)
    validation_b = np.array([100000, -100000], dtype=float)

    _, _, scaler_a = normalize_train_validation(train, validation_a)
    _, _, scaler_b = normalize_train_validation(train, validation_b)

    assert np.array_equal(scaler_a.mean_, scaler_b.mean_)
    assert np.array_equal(scaler_a.scale_, scaler_b.scale_)


def test_validation_uses_training_scaler():
    train = np.array([1, 2, 3, 4, 5], dtype=float)
    validation = np.array([6, 7], dtype=float)

    _, normalized_validation, scaler = normalize_train_validation(
        train, validation
    )

    expected = (validation - scaler.mean_[0]) / scaler.scale_[0]

    assert np.allclose(normalized_validation, expected)


def test_empty_training_data_is_rejected():
    with pytest.raises(ValueError):
        normalize_train_validation(np.array([]), np.array([1, 2]))


def test_empty_validation_data_is_rejected():
    with pytest.raises(ValueError):
        normalize_train_validation(np.array([1, 2]), np.array([]))


def test_non_finite_training_data_is_rejected():
    with pytest.raises(ValueError):
        normalize_train_validation(
            np.array([1, np.nan, 3]), np.array([4, 5])
        )


def test_non_finite_validation_data_is_rejected():
    with pytest.raises(ValueError):
        normalize_train_validation(
            np.array([1, 2, 3]), np.array([4, np.inf])
        )


def test_two_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        normalize_train_validation(
            np.ones((5, 1)), np.ones(2)
        )
