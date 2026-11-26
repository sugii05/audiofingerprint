import argparse
from pathlib import Path
import wave

import numpy as np


def make_demo_signal():
    sample_rate = 8000
    times = np.arange(4 * sample_rate) / sample_rate
    changing_tone = np.where(
        times < 2,
        np.sin(2 * np.pi * 440 * times),
        np.sin(2 * np.pi * 880 * times),
    )
    samples = 0.5 * changing_tone + 0.25 * np.cos(2 * np.pi * 120 * times)
    return samples, sample_rate


def load_wav(path):
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise ValueError("Use an uncompressed 16-bit PCM WAV file.")
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        raw_samples = audio.readframes(audio.getnframes())

    samples = np.frombuffer(raw_samples, dtype="<i2").astype(np.float64)
    samples = samples.reshape(-1, channels).mean(axis=1) / 32768.0
    return samples, sample_rate


def generate_spectrogram(samples, sample_rate, window_size=1024, hop_size=256):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("Audio must be a nonempty, finite, mono signal.")
    if sample_rate <= 0 or window_size < 4 or not 1 <= hop_size <= window_size:
        raise ValueError("Use a positive sample rate and 1 <= hop <= window (>= 4).")

    frame_count = 1 + max(0, (len(samples) - window_size + hop_size - 1) // hop_size)
    padded_length = (frame_count - 1) * hop_size + window_size
    samples = np.pad(samples, (0, padded_length - len(samples)))
    window = np.hanning(window_size)
    magnitudes = np.empty((window_size // 2 + 1, frame_count))

    for frame_index in range(frame_count):
        start = frame_index * hop_size
        frame = samples[start : start + window_size]
        spectrum = np.fft.rfft(frame * window)
        magnitudes[:, frame_index] = np.abs(spectrum)

    peak = magnitudes.max()
    if peak == 0:
        magnitudes_db = np.full_like(magnitudes, -80.0)
    else:
        magnitudes_db = 20 * np.log10(np.maximum(magnitudes / peak, 1e-4))

    times = (np.arange(frame_count) * hop_size + window_size / 2) / sample_rate
    frequencies = np.fft.rfftfreq(window_size, d=1 / sample_rate)
    return times, frequencies, magnitudes_db


def save_spectrogram(times, frequencies, magnitudes_db, hop_seconds, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_edges = np.append(times - hop_seconds / 2, times[-1] + hop_seconds / 2)
    frequency_step = frequencies[1] - frequencies[0]
    frequency_edges = np.append(
        frequencies - frequency_step / 2, frequencies[-1] + frequency_step / 2
    )
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    mesh = axis.pcolormesh(
        time_edges, frequency_edges, magnitudes_db,
        shading="flat", cmap="magma", vmin=-80, vmax=0,
    )
    axis.set(title="Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
    axis.set_ylim(0, frequencies[-1])
    figure.colorbar(mesh, ax=axis, label="Magnitude (dB relative to peak)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: build a spectrogram from a demo signal or a 16-bit PCM WAV file."
    )
    parser.add_argument("audio", nargs="?", type=Path, help="16-bit PCM WAV; omit for demo")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("output/spectrogram.png"))
    args = parser.parse_args()

    try:
        samples, sample_rate = load_wav(args.audio) if args.audio else make_demo_signal()
        times, frequencies, magnitudes_db = generate_spectrogram(
            samples, sample_rate, args.window_size, args.hop_size
        )
        save_spectrogram(
            times, frequencies, magnitudes_db, args.hop_size / sample_rate, args.output
        )
    except (OSError, ValueError, wave.Error, EOFError) as error:
        parser.error(str(error))

    print(f"Audio: {len(samples) / sample_rate:.2f} s at {sample_rate} Hz")
    print(f"Spectrogram: {len(frequencies)} frequency bins x {len(times)} time frames")
    print(f"Bin spacing: {sample_rate / args.window_size:.2f} Hz")
    print(f"Frame step: {args.hop_size / sample_rate:.3f} s")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
