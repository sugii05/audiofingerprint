import hashlib
import unittest

import numpy as np

from src.hashing import find_peaks, generate_hashes
from src.spectogram import generate_spectrogram, make_demo_signal


class HashingChecks(unittest.TestCase):
    def test_peaks_reject_weaker_neighbors_and_quiet_bins(self):
        db = np.full((20, 12), -80.0)
        db[4, 1] = -5
        db[5, 1] = -10
        db[12, 8] = -12
        db[4, 10] = -60
        self.assertEqual(find_peaks(db), [(1, 4), (8, 12)])

    def test_known_pair_uses_frequencies_and_time_difference(self):
        expected = hashlib.sha1(b"10:20:3").hexdigest()[:20]
        self.assertEqual(
            generate_hashes([(2, 10), (5, 20)]),
            [{"hash": expected, "anchor_frame": 2}],
        )

    def test_time_shift_preserves_hashes_and_shifts_anchors(self):
        peaks = [(0, 10), (3, 20), (6, 30), (10, 40)]
        original = generate_hashes(peaks)
        shifted = generate_hashes([(time + 100, frequency) for time, frequency in peaks])
        self.assertEqual([p["hash"] for p in original], [p["hash"] for p in shifted])
        self.assertEqual(
            [p["anchor_frame"] + 100 for p in original],
            [p["anchor_frame"] for p in shifted],
        )

    def test_pairing_limits_and_same_time_peaks(self):
        peaks = [(10, 40), (0, 11), (2, 20), (0, 10), (3, 30), (0, 10)]
        fingerprints = generate_hashes(peaks, fan_out=1, max_time_delta=3)
        expected_keys = [b"10:20:2", b"11:20:2", b"20:30:1"]
        self.assertEqual(
            [p["hash"] for p in fingerprints],
            [hashlib.sha1(key).hexdigest()[:20] for key in expected_keys],
        )
        self.assertEqual([p["anchor_frame"] for p in fingerprints], [0, 0, 2])

    def test_silence_and_single_peaks_produce_no_hashes(self):
        _, _, db = generate_spectrogram(np.zeros(8000), 8000)
        self.assertEqual(find_peaks(db), [])
        self.assertEqual(generate_hashes([]), [])
        self.assertEqual(generate_hashes([(0, 10)]), [])
        self.assertEqual(generate_hashes([(0, 10), (0, 20)]), [])

    def test_demo_produces_repeatable_fingerprints(self):
        samples, rate = make_demo_signal()
        _, _, db = generate_spectrogram(samples, rate)
        peaks = find_peaks(db)
        fingerprints = generate_hashes(peaks)
        self.assertGreater(len(fingerprints), 0)
        self.assertEqual(fingerprints, generate_hashes(find_peaks(db)))
        self.assertEqual(fingerprints, generate_hashes(list(reversed(peaks))))


if __name__ == "__main__":
    unittest.main()
