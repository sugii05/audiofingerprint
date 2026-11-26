import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

from .spectogram import generate_spectrogram, load_wav, make_demo_signal


def find_peaks(magnitudes_db, min_db=-35, neighborhood=3):
    magnitudes_db = np.asarray(magnitudes_db)
    if magnitudes_db.ndim != 2 or not np.all(np.isfinite(magnitudes_db)):
        raise ValueError("Use a finite, two-dimensional spectrogram.")
    if neighborhood < 1:
        raise ValueError("Neighborhood must be at least 1.")

    peaks = []
    for time_index in range(magnitudes_db.shape[1]):
        column = magnitudes_db[:, time_index]
        candidates = np.flatnonzero(
            (column[1:-1] > column[:-2])
            & (column[1:-1] >= column[2:])
            & (column[1:-1] > min_db)
        ) + 1
        for frequency_index in candidates:
            area = magnitudes_db[
                max(0, frequency_index - neighborhood) : frequency_index + neighborhood + 1,
                max(0, time_index - neighborhood) : time_index + neighborhood + 1,
            ]
            if column[frequency_index] == area.max():
                peaks.append((time_index, int(frequency_index)))
    return peaks


def generate_hashes(peaks, fan_out=5, max_time_delta=30):
    if fan_out < 1 or max_time_delta < 1:
        raise ValueError("Fan-out and maximum time difference must be positive.")
    peaks = sorted(set(peaks))
    fingerprints = []
    for index, (time1, frequency1) in enumerate(peaks):
        pairs = 0
        for target in range(index + 1, len(peaks)):
            time2, frequency2 = peaks[target]
            delta = time2 - time1
            if delta == 0:
                continue
            if delta > max_time_delta:
                break
            key = f"{frequency1}:{frequency2}:{delta}"
            fingerprint = hashlib.sha1(key.encode("ascii")).hexdigest()[:20]
            fingerprints.append({"hash": fingerprint, "anchor_frame": time1})
            pairs += 1
            if pairs == fan_out:
                break
    return fingerprints


def main():
    parser = argparse.ArgumentParser(description="Generate simple audio fingerprints.")
    parser.add_argument("audio", nargs="?", type=Path, help="16-bit PCM WAV; omit for demo")
    parser.add_argument("--output", type=Path, default=Path("output/fingerprints.json"))
    args = parser.parse_args()

    try:
        samples, sample_rate = load_wav(args.audio) if args.audio else make_demo_signal()
        window_size, hop_size = 1024, 256
        _, _, magnitudes_db = generate_spectrogram(samples, sample_rate, window_size, hop_size)
        peaks = find_peaks(magnitudes_db)
        fingerprints = generate_hashes(peaks)
        result = {
            "sample_rate": sample_rate,
            "window_size": window_size,
            "hop_size": hop_size,
            "peaks": peaks,
            "fingerprints": fingerprints,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, wave.Error, EOFError) as error:
        parser.error(str(error))

    print(f"Peaks: {len(peaks)}")
    print(f"Fingerprints: {len(fingerprints)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
