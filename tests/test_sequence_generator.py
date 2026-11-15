import numpy as np
import pytest

from src.data.sequence_generator import create_sequences


def test_creates_expected_sequence_shape():
    data = np.arange(100).reshape(20, 5)

    sequences = create_sequences(data, sequence_length=5)

    assert sequences.shape == (16, 5, 5)


def test_sequences_preserve_chronological_order():
    data = np.arange(20).reshape(10, 2)

    sequences = create_sequences(data, sequence_length=3)

    np.testing.assert_array_equal(
        sequences[0],
        data[:3],
    )

    np.testing.assert_array_equal(
        sequences[1],
        data[1:4],
    )


def test_context_produces_one_sequence_per_target():
    context = np.arange(58).reshape(29, 2)
    data = np.arange(20).reshape(10, 2)

    sequences = create_sequences(
        data,
        sequence_length=30,
        context=context,
    )

    assert sequences.shape == (10, 30, 2)


def test_first_context_sequence_ends_at_first_target():
    context = np.arange(58).reshape(29, 2)
    data = np.arange(20).reshape(10, 2)

    sequences = create_sequences(
        data,
        sequence_length=30,
        context=context,
    )

    np.testing.assert_array_equal(
        sequences[0, :-1],
        context,
    )

    np.testing.assert_array_equal(
        sequences[0, -1],
        data[0],
    )


def test_context_uses_only_most_recent_observations():
    context = np.arange(80).reshape(40, 2)
    data = np.ones((3, 2))

    sequences = create_sequences(
        data,
        sequence_length=5,
        context=context,
    )

    np.testing.assert_array_equal(
        sequences[0, :-1],
        context[-4:],
    )


def test_context_and_data_feature_counts_must_match():
    context = np.ones((4, 3))
    data = np.ones((5, 2))

    with pytest.raises(ValueError, match="feature counts"):
        create_sequences(data, sequence_length=5, context=context)


def test_insufficient_context_raises_error():
    context = np.ones((3, 2))
    data = np.ones((5, 2))

    with pytest.raises(ValueError, match="Context must contain"):
        create_sequences(data, sequence_length=5, context=context)


def test_insufficient_data_raises_error():
    data = np.ones((4, 2))

    with pytest.raises(ValueError, match="at least sequence_length"):
        create_sequences(data, sequence_length=5)


def test_non_finite_data_raises_error():
    data = np.ones((5, 2))
    data[2, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        create_sequences(data, sequence_length=3)


def test_invalid_sequence_length_raises_error():
    data = np.ones((5, 2))

    for length in [0, 1, -1, 2.5, True]:
        with pytest.raises(ValueError):
            create_sequences(data, sequence_length=length)


def test_input_data_is_not_modified():
    data = np.arange(20, dtype=float).reshape(10, 2)
    original = data.copy()

    create_sequences(data, sequence_length=3)

    np.testing.assert_array_equal(data, original)


def test_context_is_not_modified():
    context = np.ones((4, 2))
    data = np.ones((5, 2))
    original = context.copy()

    create_sequences(data, sequence_length=5, context=context)

    np.testing.assert_array_equal(context, original)
