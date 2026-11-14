
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """Multivariate Isolation Forest for spacecraft telemetry."""

    def __init__(
        self,
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=2,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_mask_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    @staticmethod
    def _validate_data(data, description):
        try:
            data = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{description} must contain numeric values."
            ) from exc

        if data.ndim != 2:
            raise ValueError(
                f"{description} must be a two-dimensional array."
            )

        if data.shape[0] < 2 or data.shape[1] < 1:
            raise ValueError(
                f"{description} must contain at least two observations "
                "and one feature."
            )

        if not np.isfinite(data).all():
            raise ValueError(
                f"{description} contains non-finite values."
            )

        return data

    def fit(self, training_data):
        """Fit using only the supplied model-fitting partition."""

        training_data = self._validate_data(
            training_data,
            "Training data",
        )

        self.n_features_in_ = training_data.shape[1]

        # Remove features with no variation in the fitting partition.
        feature_std = np.std(training_data, axis=0)
        self.feature_mask_ = feature_std > 0

        if not np.any(self.feature_mask_):
            raise ValueError(
                "All training features are constant."
            )

        filtered_training = training_data[:, self.feature_mask_]

        self.model.fit(filtered_training)
        self.is_fitted_ = True

        return self

    def score_samples(self, data):
        """
        Return anomaly scores.

        Higher values indicate more anomalous observations.
        """

        if not self.is_fitted_:
            raise RuntimeError(
                "Detector must be fitted before scoring."
            )

        data = self._validate_data(data, "Input data")

        if data.shape[1] != self.n_features_in_:
            raise ValueError(
                "Input feature count does not match training data."
            )

        filtered_data = data[:, self.feature_mask_]

        # Isolation Forest assigns lower scores to anomalies.
        return -self.model.score_samples(filtered_data)

    def predict_anomalies(self, data, threshold):
        """Convert anomaly scores into binary anomaly predictions."""

        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(
                threshold,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(threshold)
        ):
            raise ValueError(
                "Threshold must be a finite numeric value."
            )

        if threshold < 0:
            raise ValueError(
                "Threshold cannot be negative."
            )

        if threshold < 0:
            raise ValueError(
                "Threshold cannot be negative."
            )

        scores = self.score_samples(data)

        return scores >= float(threshold)
