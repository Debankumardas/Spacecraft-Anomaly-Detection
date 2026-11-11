
import numpy as np
import pytest

from src.data.create_chronological_split import chronological_split


def test_chronological_split_sizes():
    signal = np.arange(2880, dtype=np.float64)

    train, validation, split_index = chronological_split(signal, 0.80)

    assert len(train) == 2304
    assert len(validation) == 576
    assert split_index == 2304


def test_chronological_order_is_preserved():
    signal = np.arange(100, dtype=np.float64)

    train, validation, split_index = chronological_split(signal)

    assert np.array_equal(train, signal[:split_index])
    assert np.array_equal(validation, signal[split_index:])


def test_no_samples_are_lost():
    signal = np.arange(100, dtype=np.float64)

    train, validation, _ = chronological_split(signal)

    combined = np.concatenate((train, validation))

    assert np.array_equal(combined, signal)


def test_two_dimensional_input_is_rejected():
    signal = np.ones((100, 2))

    with pytest.raises(ValueError):
        chronological_split(signal)


def test_single_sample_is_rejected():
    signal = np.array([1.0])

    with pytest.raises(ValueError):
        chronological_split(signal)


def test_non_finite_values_are_rejected():
    signal = np.array([1.0, 2.0, np.nan, 4.0])

    with pytest.raises(ValueError):
        chronological_split(signal)


@pytest.mark.parametrize("ratio", [0, 1, -0.5, 1.5, True, np.nan])
def test_invalid_train_ratios_are_rejected(ratio):
    signal = np.arange(100, dtype=np.float64)

    with pytest.raises(ValueError):
        chronological_split(signal, ratio)


def test_empty_partitions_are_rejected():
    signal = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        chronological_split(signal, 0.01)
