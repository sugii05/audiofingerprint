import tempfile
import unittest
from pathlib import Path
import wave

import numpy as np

from src.spectogram import generate_spectrogram, load_wav, make_demo_signal


class SpectrogramChecks(unittest.TestCase):
    def test_demo_tracks_the_frequency_change(self):
        samples, sample_rate = make_demo_signal()
        times, frequencies, db = generate_spectrogram(samples, sample_rate)
        self.assertEqual(db.shape, (513, 122))
        self.assertAlmostEqual(times[0], 0.064)
        np.testing.assert_allclose(np.diff(times), 0.032)
        for seconds, expected_hz in [(1, 440), (3, 880)]:
            column = db[:, np.argmin(np.abs(times - seconds))]
            self.assertAlmostEqual(frequencies[np.argmax(column)], expected_hz, delta=7.8125)
            low_band = (frequencies > 80) & (frequencies < 160)
            low_peak = frequencies[low_band][np.argmax(column[low_band])]
            self.assertAlmostEqual(low_peak, 120, delta=7.8125)

    def test_fft_matches_a_known_bin_centered_tone(self):
        samples = np.sin(2 * np.pi * 1000 * np.arange(1024) / 8000)
        _, frequencies, db = generate_spectrogram(samples, 8000)
        self.assertEqual(frequencies[np.argmax(db[:, 0])], 1000)
        self.assertEqual(db.max(), 0)
        self.assertGreaterEqual(db.min(), -80)

    def test_silence_and_short_audio_stay_finite(self):
        _, _, db = generate_spectrogram(np.zeros(100), 8000)
        self.assertEqual(db.shape, (513, 1))
        np.testing.assert_array_equal(db, -80)

    def test_partial_last_frame_keeps_the_end_of_the_audio(self):
        samples = np.zeros(1025)
        samples[-1] = 1
        _, _, db = generate_spectrogram(samples, 8000)
        self.assertEqual(db.shape, (513, 2))
        self.assertEqual(db[:, -1].max(), 0)

    def test_pcm_wav_scaling_and_channel_average(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            for channels, pcm in [
                (1, np.array([-32768, 0, 16384], dtype="<i2")),
                (2, np.array([[-32768, 0], [16384, 16384], [1000, -1000]], dtype="<i2")),
            ]:
                with self.subTest(channels=channels):
                    with wave.open(str(path), "wb") as audio:
                        audio.setparams((channels, 2, 8000, 0, "NONE", "not compressed"))
                        audio.writeframes(pcm.tobytes())
                    samples, rate = load_wav(path)
                    expected = pcm.astype(float).reshape(-1, channels).mean(axis=1) / 32768
                    np.testing.assert_allclose(samples, expected)
                    self.assertEqual(rate, 8000)

    def test_empty_audio_and_invalid_window_settings_are_rejected(self):
        for samples, window, hop in [([], 1024, 256), ([0], 2, 1), ([0], 1024, 0), ([0], 8, 9)]:
            with self.subTest(window=window, hop=hop, samples=samples):
                with self.assertRaises(ValueError):
                    generate_spectrogram(samples, 8000, window, hop)


if __name__ == "__main__":
    unittest.main()
