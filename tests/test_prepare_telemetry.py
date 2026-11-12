
"""Tests for NASA SMAP telemetry preprocessing."""

import numpy as np
import pytest

from src.data.prepare_telemetry import (
    calculate_split_index,
    prepare_telemetry_arrays,
    validate_telemetry,
)


# --------------------------------------------------
# TEST DATA
# --------------------------------------------------

@pytest.fixture
def sample_data():
    """Create reproducible synthetic telemetry arrays."""

    rng = np.random.default_rng(42)

    training_data = rng.normal(
        loc=10.0,
        scale=3.0,
        size=(100, 5),
    )

    test_data = rng.normal(
        loc=12.0,
        scale=4.0,
        size=(40, 5),
    )

    return training_data, test_data


# --------------------------------------------------
# TELEMETRY VALIDATION TESTS
# --------------------------------------------------

def test_validate_telemetry_accepts_valid_data(sample_data):
    training_data, _ = sample_data

    result = validate_telemetry(
        training_data,
        "Training data",
    )

    assert result.shape == (100, 5)
    assert result.dtype == np.float64
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "invalid_data",
    [
        np.array([1.0, 2.0, 3.0]),
        np.ones((2, 3, 4)),
    ],
)
def test_validate_telemetry_rejects_non_2d_data(invalid_data):
    with pytest.raises(ValueError, match="2D array"):
        validate_telemetry(
            invalid_data,
            "Test data",
        )


def test_validate_telemetry_rejects_less_than_two_rows():
    data = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="at least two observations"):
        validate_telemetry(
            data,
            "Test data",
        )


def test_validate_telemetry_rejects_zero_features():
    data = np.empty((5, 0))

    with pytest.raises(ValueError, match="at least one feature"):
        validate_telemetry(
            data,
            "Test data",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_telemetry_rejects_non_finite_values(invalid_value):
    data = np.ones((5, 3))
    data[2, 1] = invalid_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_telemetry(
            data,
            "Test data",
        )


def test_validate_telemetry_rejects_non_numeric_data():
    data = np.array(
        [
            ["one", "two"],
            ["three", "four"],
        ]
    )

    with pytest.raises(ValueError, match="numeric values"):
        validate_telemetry(
            data,
            "Test data",
        )


# --------------------------------------------------
# SPLIT INDEX TESTS
# --------------------------------------------------

@pytest.mark.parametrize(
    ("data_length", "train_ratio", "expected"),
    [
        (100, 0.80, 80),
        (100, 0.70, 70),
        (100, 0.50, 50),
        (10, 0.80, 8),
        (2880, 0.80, 2304),
    ],
)
def test_calculate_split_index_returns_correct_index(
    data_length,
    train_ratio,
    expected,
):
    result = calculate_split_index(
        data_length,
        train_ratio,
    )

    assert result == expected


@pytest.mark.parametrize(
    "invalid_ratio",
    [
        0,
        1,
        -0.1,
        1.1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        False,
        "0.8",
        None,
    ],
)
def test_calculate_split_index_rejects_invalid_ratio(invalid_ratio):
    with pytest.raises(ValueError, match="train_ratio"):
        calculate_split_index(
            100,
            invalid_ratio,
        )


def test_calculate_split_index_rejects_empty_partition():
    with pytest.raises(ValueError, match="non-empty partitions"):
        calculate_split_index(
            2,
            0.1,
        )


# --------------------------------------------------
# PREPROCESSING TESTS
# --------------------------------------------------

def test_prepare_telemetry_preserves_shapes(sample_data):
    training_data, test_data = sample_data

    (
        normalized_training,
        normalized_test,
        scaler,
        split_index,
    ) = prepare_telemetry_arrays(
        training_data,
        test_data,
    )

    assert normalized_training.shape == training_data.shape
    assert normalized_test.shape == test_data.shape

    assert normalized_training.dtype == np.float64
    assert normalized_test.dtype == np.float64

    assert scaler.n_features_in_ == 5
    assert split_index == 80


def test_scaler_is_fitted_only_on_training_partition(sample_data):
    training_data, test_data = sample_data

    (
        _,
        _,
        scaler,
        split_index,
    ) = prepare_telemetry_arrays(
        training_data,
        test_data,
        train_ratio=0.80,
    )

    expected_mean = np.mean(
        training_data[:split_index],
        axis=0,
    )

    expected_variance = np.var(
        training_data[:split_index],
        axis=0,
    )

    np.testing.assert_allclose(
        scaler.mean_,
        expected_mean,
    )

    np.testing.assert_allclose(
        scaler.var_,
        expected_variance,
    )

    assert scaler.n_samples_seen_ == split_index


def test_test_data_does_not_change_scaler_statistics(sample_data):
    training_data, test_data = sample_data

    # First preprocessing run.
    _, _, original_scaler, _ = prepare_telemetry_arrays(
        training_data,
        test_data,
    )

    # Replace test data with extreme values.
    extreme_test_data = np.full(
        test_data.shape,
        1_000_000.0,
    )

    # Second preprocessing run.
    _, _, extreme_scaler, _ = prepare_telemetry_arrays(
        training_data,
        extreme_test_data,
    )

    # Test data must not influence fitted scaler statistics.
    np.testing.assert_allclose(
        original_scaler.mean_,
        extreme_scaler.mean_,
    )

    np.testing.assert_allclose(
        original_scaler.scale_,
        extreme_scaler.scale_,
    )

    np.testing.assert_allclose(
        original_scaler.var_,
        extreme_scaler.var_,
    )


def test_training_partition_is_standardized(sample_data):
    training_data, test_data = sample_data

    (
        normalized_training,
        _,
        _,
        split_index,
    ) = prepare_telemetry_arrays(
        training_data,
        test_data,
    )

    fitting_partition = normalized_training[:split_index]

    np.testing.assert_allclose(
        np.mean(fitting_partition, axis=0),
        np.zeros(training_data.shape[1]),
        atol=1e-10,
    )

    np.testing.assert_allclose(
        np.std(fitting_partition, axis=0),
        np.ones(training_data.shape[1]),
        atol=1e-10,
    )


def test_remaining_training_data_uses_fitted_scaler(sample_data):
    training_data, test_data = sample_data

    (
        normalized_training,
        _,
        scaler,
        split_index,
    ) = prepare_telemetry_arrays(
        training_data,
        test_data,
    )

    expected = scaler.transform(
        training_data[split_index:]
    )

    np.testing.assert_allclose(
        normalized_training[split_index:],
        expected,
    )


def test_test_data_uses_training_scaler(sample_data):
    training_data, test_data = sample_data

    (
        _,
        normalized_test,
        scaler,
        _,
    ) = prepare_telemetry_arrays(
        training_data,
        test_data,
    )

    expected = scaler.transform(test_data)

    np.testing.assert_allclose(
        normalized_test,
        expected,
    )


def test_preprocessing_rejects_different_feature_counts():
    training_data = np.ones((10, 5))
    test_data = np.ones((10, 4))

    with pytest.raises(
        ValueError,
        match="same number of features",
    ):
        prepare_telemetry_arrays(
            training_data,
            test_data,
        )


def test_preprocessing_rejects_invalid_training_data():
    training_data = np.ones((10, 5))
    training_data[2, 1] = np.nan

    test_data = np.ones((10, 5))

    with pytest.raises(ValueError, match="NaN or infinite"):
        prepare_telemetry_arrays(
            training_data,
            test_data,
        )


def test_preprocessing_rejects_invalid_test_data():
    training_data = np.ones((10, 5))
    test_data = np.ones((10, 5))

    test_data[2, 1] = np.inf

    with pytest.raises(ValueError, match="NaN or infinite"):
        prepare_telemetry_arrays(
            training_data,
            test_data,
        )


def test_preprocessing_rejects_invalid_train_ratio(sample_data):
    training_data, test_data = sample_data

    with pytest.raises(ValueError, match="train_ratio"):
        prepare_telemetry_arrays(
            training_data,
            test_data,
            train_ratio=1.0,
        )


def test_preprocessing_returns_finite_arrays(sample_data):
    training_data, test_data = sample_data

    (
        normalized_training,
        normalized_test,
        _,
        _,
    ) = prepare_telemetry_arrays(
        training_data,
        test_data,
    )

    assert np.isfinite(normalized_training).all()
    assert np.isfinite(normalized_test).all()
