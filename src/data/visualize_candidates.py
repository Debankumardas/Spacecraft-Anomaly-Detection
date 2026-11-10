from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "nasa_smap_msl"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

CHANNELS = ["E-13", "P-1"]
ZOOM_SAMPLES = 1000


def load_signal(split, channel):
    path = DATA_ROOT / split / f"{channel}.npy"

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path, allow_pickle=False)

    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError(f"Unexpected array shape for {channel}: {data.shape}")

    return data[:, 0]


def plot_channel(channel):
    train = load_signal("train", channel)
    test = load_signal("test", channel)

    fig, axes = plt.subplots(3, 1, figsize=(15, 11))

    fig.suptitle(
        f"{channel} — Telemetry Feature 0",
        fontsize=16,
        fontweight="bold"
    )

    axes[0].plot(train, linewidth=0.7)
    axes[0].set_title("Training Telemetry")
    axes[0].set_ylabel("Telemetry value")
    axes[0].set_xlabel("Sample index")
    axes[0].grid(alpha=0.3)

    axes[1].plot(test, linewidth=0.6)
    axes[1].set_title("Complete Test Telemetry")
    axes[1].set_ylabel("Telemetry value")
    axes[1].set_xlabel("Sample index")
    axes[1].grid(alpha=0.3)

    zoom_length = min(ZOOM_SAMPLES, len(test))

    axes[2].plot(test[:zoom_length], linewidth=0.8)
    axes[2].set_title(f"First {zoom_length} Test Samples — Detailed View")
    axes[2].set_ylabel("Telemetry value")
    axes[2].set_xlabel("Sample index")
    axes[2].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUT_DIR / f"{channel}_telemetry_inspection.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    print(f"\nChannel: {channel}")
    print(f"Training shape: {train.shape}")
    print(f"Testing shape: {test.shape}")
    print(f"Training range: {train.min():.5f} to {train.max():.5f}")
    print(f"Testing range: {test.min():.5f} to {test.max():.5f}")
    print(f"Saved figure: {output_path}")

    plt.show()
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for channel in CHANNELS:
        plot_channel(channel)

    print("\nAll visualizations completed.")


if __name__ == "__main__":
    main()
