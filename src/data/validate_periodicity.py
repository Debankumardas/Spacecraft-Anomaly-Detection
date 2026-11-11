from pathlib import Path

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

N_SEGMENTS = 4
MAX_LAG = 150
SEARCH_MIN_LAG = 30
SEARCH_MAX_LAG = 70


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


def find_local_peaks(values, min_lag, max_lag):
    peaks = []

    for lag in range(min_lag + 1, max_lag):
        if (
            values[lag] > 0
            and values[lag] > values[lag - 1]
            and values[lag] >= values[lag + 1]
        ):
            peaks.append((lag, values[lag]))

    return sorted(peaks, key=lambda item: item[1], reverse=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    data = np.load(DATA_PATH, allow_pickle=False)
    signal = data[:, 0].astype(np.float64)

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite values.")

    segment_length = len(signal) // N_SEGMENTS

    if N_SEGMENTS < 1:
        raise ValueError("N_SEGMENTS must be positive.")

    if segment_length <= MAX_LAG:
        raise ValueError("Each segment must be longer than MAX_LAG.")

    print("=" * 70)
    print("E-13 PERIODICITY VALIDATION")
    print("=" * 70)
    print(f"Total training samples: {len(signal)}")
    print(f"Segments: {N_SEGMENTS}")
    print(f"Samples per segment: {segment_length}")
    print(f"Peak search range: {SEARCH_MIN_LAG}-{SEARCH_MAX_LAG}")

    for i in range(N_SEGMENTS):
        start = i * segment_length
        end = (
            (i + 1) * segment_length
            if i < N_SEGMENTS - 1
            else len(signal)
        )

        segment = signal[start:end]
        acf = calculate_autocorrelation(segment, MAX_LAG)

        peaks = find_local_peaks(
            acf,
            SEARCH_MIN_LAG,
            SEARCH_MAX_LAG,
        )

        print("\n" + "-" * 70)
        print(f"SEGMENT {i + 1}: samples {start} to {end - 1}")
        print(f"Mean: {segment.mean():.6f}")
        print(f"Standard deviation: {segment.std():.6f}")
        print(f"Autocorrelation at lag 50: {acf[50]:.6f}")

        if peaks:
            print("Top local autocorrelation peaks:")
            for lag, value in peaks[:3]:
                print(f"  Lag {lag:3d}: {value:.6f}")
        else:
            print("No local peaks found in the search range.")

    print("\n" + "=" * 70)
    print("Validation completed using training data only.")
    print("=" * 70)


if __name__ == "__main__":
    main()
