import numpy as np


def create_sequences(data, sequence_length=30, context=None):
    """
    Create sliding windows ending at each target observation.

    When context is provided, its final sequence_length - 1
    observations are used as historical context. This allows
    every target observation to receive a complete sequence.

    Without context, the first sequence contains sequence_length
    observations from data itself.
    """

    if (
        isinstance(sequence_length, (bool, np.bool_))
        or not isinstance(sequence_length, (int, np.integer))
        or sequence_length < 2
    ):
        raise ValueError(
            "Sequence length must be an integer of at least 2."
        )

    try:
        data = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Data must contain numeric values.") from exc

    if data.ndim != 2:
        raise ValueError("Data must be a two-dimensional array.")

    if data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("Data must contain observations and features.")

    if not np.isfinite(data).all():
        raise ValueError("Data contains non-finite values.")

    if context is None:
        if len(data) < sequence_length:
            raise ValueError(
                "Data must contain at least sequence_length observations."
            )

        combined = data
        first_start = 0
        number_of_sequences = len(data) - sequence_length + 1

    else:
        try:
            context = np.asarray(context, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Context must contain numeric values."
            ) from exc

        if context.ndim != 2:
            raise ValueError(
                "Context must be a two-dimensional array."
            )

        if context.shape[1] != data.shape[1]:
            raise ValueError(
                "Context and data feature counts must match."
            )

        if not np.isfinite(context).all():
            raise ValueError(
                "Context contains non-finite values."
            )

        required_context = sequence_length - 1

        if len(context) < required_context:
            raise ValueError(
                "Context must contain at least sequence_length - 1 "
                "observations."
            )

        context = context[-required_context:]

        combined = np.concatenate((context, data), axis=0)
        first_start = 0
        number_of_sequences = len(data)

    sequences = np.stack(
        [
            combined[start:start + sequence_length]
            for start in range(
                first_start,
                first_start + number_of_sequences,
            )
        ]
    )

    return sequences
