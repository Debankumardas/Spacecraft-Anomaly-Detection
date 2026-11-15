import numpy as np
import pytest

from src.models.lstm_autoencoder import LSTMAutoencoder


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)

    training = rng.normal(size=(40, 10, 3)).astype(np.float32)
    validation = rng.normal(size=(10, 10, 3)).astype(np.float32)

    return training, validation


@pytest.fixture
def fitted_model(sample_data):
    training, validation = sample_data

    model = LSTMAutoencoder(
        hidden_size=8,
        batch_size=16,
        max_epochs=3,
        patience=2,
    )

    model.fit(training, validation)

    return model


def test_model_fits_successfully(sample_data):
    training, validation = sample_data

    model = LSTMAutoencoder(
        hidden_size=8,
        max_epochs=2,
    )

    result = model.fit(training, validation)

    assert result is model
    assert model.is_fitted_
    assert model.n_features_in_ == 3
    assert model.sequence_length_ == 10


def test_training_history_contains_loss(sample_data):
    training, validation = sample_data

    model = LSTMAutoencoder(
        hidden_size=8,
        max_epochs=3,
    )

    model.fit(training, validation)

    assert len(model.training_history_) >= 1
    assert len(model.training_history_) <= 3
    assert all(
        np.isfinite(item["training_loss"])
        and np.isfinite(item["validation_loss"])
        for item in model.training_history_
    )


def test_scores_are_finite_and_non_negative(fitted_model, sample_data):
    _, validation = sample_data

    scores = fitted_model.score_samples(validation)

    assert scores.shape == (10,)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0)


def test_predictions_are_boolean(fitted_model, sample_data):
    _, validation = sample_data

    predictions = fitted_model.predict_anomalies(
        validation,
        threshold=0.5,
    )

    assert predictions.dtype == bool
    assert predictions.shape == (10,)


def test_scoring_before_fit_raises_error(sample_data):
    _, validation = sample_data

    model = LSTMAutoencoder()

    with pytest.raises(RuntimeError, match="fitted"):
        model.score_samples(validation)


def test_mismatched_sequence_dimensions_raise_error(sample_data):
    training, validation = sample_data

    model = LSTMAutoencoder(
        hidden_size=8,
        max_epochs=2,
    )

    model.fit(training, validation)

    wrong_data = np.ones((5, 9, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="dimensions"):
        model.score_samples(wrong_data)


def test_non_finite_sequences_raise_error(sample_data):
    training, validation = sample_data

    training[0, 0, 0] = np.nan

    model = LSTMAutoencoder()

    with pytest.raises(ValueError, match="non-finite"):
        model.fit(training, validation)


def test_mismatched_training_and_validation_raise_error():
    training = np.ones((10, 10, 3))
    validation = np.ones((5, 10, 2))

    model = LSTMAutoencoder()

    with pytest.raises(ValueError, match="dimensions"):
        model.fit(training, validation)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, True, "bad"])
def test_invalid_thresholds_raise_error(fitted_model, sample_data, threshold):
    _, validation = sample_data

    with pytest.raises(ValueError, match="Threshold"):
        fitted_model.predict_anomalies(validation, threshold)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"hidden_size": True},
        {"batch_size": 0},
        {"batch_size": True},
        {"learning_rate": np.nan},
        {"learning_rate": np.inf},
        {"learning_rate": 0},
        {"max_epochs": 0},
        {"max_epochs": True},
        {"patience": 0},
        {"random_state": -1},
        {"random_state": True},
    ],
)
def test_invalid_constructor_parameters_raise_error(kwargs):
    with pytest.raises(ValueError):
        LSTMAutoencoder(**kwargs)