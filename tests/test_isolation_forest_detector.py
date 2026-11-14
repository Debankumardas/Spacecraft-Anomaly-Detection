import numpy as np
import pytest

from src.models.isolation_forest_detector import (
    IsolationForestDetector,
)


def make_data():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 5))
    data[:, 4] = 3.0
    return data


def test_detector_fits_and_removes_constant_features():
    model = IsolationForestDetector(random_state=42)
    model.fit(make_data())

    assert model.is_fitted_
    assert model.n_features_in_ == 5
    assert model.feature_mask_.sum() == 4


def test_scores_are_finite_and_correct_length():
    data = make_data()
    model = IsolationForestDetector(random_state=42).fit(data[:80])

    scores = model.score_samples(data[80:])

    assert scores.shape == (20,)
    assert np.isfinite(scores).all()


def test_predictions_are_boolean():
    data = make_data()
    model = IsolationForestDetector(random_state=42).fit(data[:80])

    predictions = model.predict_anomalies(data[80:], 0.5)

    assert predictions.shape == (20,)
    assert predictions.dtype == bool


def test_scoring_before_fitting_raises_error():
    model = IsolationForestDetector()

    with pytest.raises(RuntimeError):
        model.score_samples(make_data())


def test_wrong_feature_count_raises_error():
    model = IsolationForestDetector().fit(make_data())

    with pytest.raises(ValueError):
        model.score_samples(np.ones((10, 4)))


def test_non_finite_data_raises_error():
    data = make_data()
    data[0, 0] = np.nan

    with pytest.raises(ValueError):
        IsolationForestDetector().fit(data)


def test_one_dimensional_data_raises_error():
    with pytest.raises(ValueError):
        IsolationForestDetector().fit(np.ones(100))


def test_all_constant_features_raise_error():
    with pytest.raises(ValueError):
        IsolationForestDetector().fit(np.ones((20, 4)))


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, True, "bad"])
def test_invalid_threshold_raises_error(threshold):
    data = make_data()
    model = IsolationForestDetector().fit(data[:80])

    with pytest.raises(ValueError):
        model.predict_anomalies(data[80:], threshold)
