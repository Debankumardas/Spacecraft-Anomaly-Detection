
"""
One-step persistence forecasting baseline.

Each prediction uses the immediately preceding actual observation.
The first validation prediction uses the final training observation.
"""

import numpy as np


def predict_persistence(history, targets):
    """
    Generate one-step-ahead persistence predictions.

    Parameters:
        history: Previous observations available before the target sequence.
        targets: Actual observations to predict sequentially.

    Returns:
        A NumPy array containing one prediction per target.
    """

    history = np.asarray(history, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if history.ndim != 1 or targets.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional.")

    if len(history) == 0:
        raise ValueError("History cannot be empty.")

    if len(targets) == 0:
        raise ValueError("Targets cannot be empty.")

    if not np.isfinite(history).all():
        raise ValueError("History contains non-finite values.")

    if not np.isfinite(targets).all():
        raise ValueError("Targets contain non-finite values.")

    # First prediction uses the final historical observation.
    # Later predictions use the previous actual target.
    predictions = np.concatenate(
        ([history[-1]], targets[:-1])
    )

    return predictions
