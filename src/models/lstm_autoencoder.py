import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMNetwork(nn.Module):
    """Sequence-to-sequence LSTM autoencoder."""

    def __init__(self, n_features, hidden_size):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, sequences):
        _, (hidden, _) = self.encoder(sequences)

        sequence_length = sequences.shape[1]

        latent = hidden[-1].unsqueeze(1).repeat(
            1, sequence_length, 1
        )

        decoded, _ = self.decoder(latent)

        return self.output_layer(decoded)


class LSTMAutoencoder:
    """LSTM-based reconstruction model for telemetry anomaly detection."""

    def __init__(
        self,
        hidden_size=32,
        learning_rate=0.001,
        batch_size=64,
        max_epochs=30,
        patience=5,
        random_state=42,
    ):
        integer_parameters = {
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "random_state": random_state,
        }

        for name, value in integer_parameters.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
            ):
                raise ValueError(f"{name} must be an integer.")

        if hidden_size < 1:
            raise ValueError("Hidden size must be positive.")

        if batch_size < 1:
            raise ValueError("Batch size must be positive.")

        if max_epochs < 1 or patience < 1:
            raise ValueError("Epochs and patience must be positive.")

        if random_state < 0:
            raise ValueError("Random state must be non-negative.")

        if (
            isinstance(learning_rate, (bool, np.bool_))
            or not isinstance(
                learning_rate,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError(
                "Learning rate must be a finite positive number."
            )

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

        self.model = None
        self.n_features_in_ = None
        self.sequence_length_ = None
        self.is_fitted_ = False
        self.training_history_ = []

    @staticmethod
    def _validate_sequences(sequences, description):
        try:
            sequences = np.asarray(sequences, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{description} must contain numeric values."
            ) from exc

        if sequences.ndim != 3:
            raise ValueError(
                f"{description} must have three dimensions "
                "(samples, timesteps, features)."
            )

        if any(size < 1 for size in sequences.shape):
            raise ValueError(
                f"{description} cannot contain empty dimensions."
            )

        if sequences.shape[1] < 2:
            raise ValueError("Sequences must contain at least two timesteps.")

        if not np.isfinite(sequences).all():
            raise ValueError(
                f"{description} contains non-finite values."
            )

        return sequences

    def fit(self, training_sequences, validation_sequences):
        """Train using training sequences and chronological validation."""

        training_sequences = self._validate_sequences(
            training_sequences, "Training sequences"
        )

        validation_sequences = self._validate_sequences(
            validation_sequences, "Validation sequences"
        )

        if training_sequences.shape[1:] != validation_sequences.shape[1:]:
            raise ValueError(
                "Training and validation sequence dimensions must match."
            )

        self.n_features_in_ = training_sequences.shape[2]
        self.sequence_length_ = training_sequences.shape[1]

        torch.manual_seed(self.random_state)
        torch.set_num_threads(2)

        self.model = LSTMNetwork(
            n_features=self.n_features_in_,
            hidden_size=self.hidden_size,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        loss_function = nn.MSELoss()

        training_tensor = torch.from_numpy(
            training_sequences.copy()
        )

        validation_tensor = torch.from_numpy(
            validation_sequences.copy()
        )

        loader = DataLoader(
            TensorDataset(training_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        best_validation_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        self.training_history_ = []

        for epoch in range(self.max_epochs):
            self.model.train()
            training_losses = []

            for (batch,) in loader:
                optimizer.zero_grad()

                reconstructed = self.model(batch)
                loss = loss_function(reconstructed, batch)

                loss.backward()
                optimizer.step()

                training_losses.append(loss.item())

            self.model.eval()

            with torch.no_grad():
                reconstructed = self.model(validation_tensor)

                validation_loss = loss_function(
                    reconstructed,
                    validation_tensor,
                ).item()

            training_loss = float(np.mean(training_losses))

            self.training_history_.append({
                "epoch": epoch + 1,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            })

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("LSTM training failed.")

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_fitted_ = True

        return self

    def score_samples(self, sequences):
        """
        Return reconstruction MSE for each sequence's final timestep.
        Each score corresponds to the sequence's target observation.
        Higher scores indicate more unusual observations.
        """

        if not self.is_fitted_:
            raise RuntimeError(
                "Model must be fitted before scoring."
            )

        sequences = self._validate_sequences(
            sequences, "Input sequences"
        )

        if sequences.shape[1:] != (
            self.sequence_length_,
            self.n_features_in_,
        ):
            raise ValueError(
                "Input sequence dimensions do not match training data."
            )

        tensor = torch.from_numpy(sequences.copy())

        self.model.eval()

        with torch.no_grad():
            reconstructed = self.model(tensor)

            errors = torch.mean(
                (reconstructed[:, -1, :] - tensor[:, -1, :]) ** 2,
                dim=1,
            )

        return errors.numpy()

    def predict_anomalies(self, sequences, threshold):
        """Convert reconstruction scores into binary predictions."""

        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(
                threshold,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError(
                "Threshold must be a finite non-negative number."
            )

        return self.score_samples(sequences) >= float(threshold)
