from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nasa_smap_msl"
    / "train"
    / "E-13.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "E-13_temporal_structure.png"
)

MAX_LAG = 200
ROLLING_WINDOW = 100


def calculate_autocorrelation(signal, max_lag):
    centered = signal - np.mean(signal)
    denominator = np.dot(centered, centered)

    if denominator == 0:
        raise ValueError("Signal has zero variance.")

    return np.array([
        np.dot(centered[:len(centered) - lag or None], centered[lag:])
        / denominator
        for lag in range(max_lag + 1)
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    data = np.load(DATA_PATH, allow_pickle=False)
    signal = data[:, 0].astype(np.float64)

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite values.")

    if len(signal) <= MAX_LAG:
        raise ValueError("Signal is too short for the requested lag range.")

    autocorrelation = calculate_autocorrelation(signal, MAX_LAG)

    rolling_mean = np.convolve(
        signal,
        np.ones(ROLLING_WINDOW) / ROLLING_WINDOW,
        mode="valid",
    )

    rolling_std = np.array([
        np.std(signal[i:i + ROLLING_WINDOW])
        for i in range(len(signal) - ROLLING_WINDOW + 1)
    ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        constrained_layout=True,
    )

    # 1. Original telemetry signal
    axes[0].plot(signal, linewidth=0.8)
    axes[0].set_title("E-13 Training Telemetry — Feature 0")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Telemetry Value")
    axes[0].grid(alpha=0.3)

    # 2. Autocorrelation
    axes[1].stem(
        range(MAX_LAG + 1),
        autocorrelation,
        linefmt="C0-",
        markerfmt="C0o",
        basefmt="k-",
    )
    axes[1].axhline(0, linewidth=0.8)
    axes[1].axhline(0.5, linestyle="--", linewidth=0.8)
    axes[1].set_title("Autocorrelation — Training Data")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Correlation")
    axes[1].grid(alpha=0.3)

    # 3. Rolling statistics
    axes[2].plot(
        rolling_mean,
        label=f"Rolling Mean ({ROLLING_WINDOW} samples)",
    )
    axes[2].plot(
        rolling_std,
        label=f"Rolling Standard Deviation ({ROLLING_WINDOW} samples)",
    )
    axes[2].set_title("Rolling Statistics")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Value")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.close(fig)

    print("=" * 70)
    print("TEMPORAL STRUCTURE VISUALIZATION")
    print("=" * 70)
    print(f"Training samples: {len(signal)}")
    print(f"Autocorrelation lags: {MAX_LAG}")
    print(f"Rolling window: {ROLLING_WINDOW}")
    print(f"Rolling mean range: {rolling_mean.min():.6f} to {rolling_mean.max():.6f}")
    print(f"Rolling std range: {rolling_std.min():.6f} to {rolling_std.max():.6f}")
    print(f"Figure saved: {OUTPUT_PATH}")
    print("\nTraining data only.")


if __name__ == "__main__":
    main()
