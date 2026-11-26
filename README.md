# Audio fingerprint

A small Python prototype for spectrograms and combinatorial audio hashing.

```sh
python3 -m pip install -r requirements.txt
python3 -m src.spectogram
python3 -m src.hashing
```

Both commands use demo audio by default. Add a 16-bit PCM WAV path to use your own:

```sh
python3 -m src.hashing song.wav
```

Results go in `output/`: `spectrogram.png` and `fingerprints.json`.
Hashing pairs strong peaks with up to five later peaks within 30 frames.
Each hash uses `(frequency_bin_1, frequency_bin_2, frame_difference)`;
the anchor frame is stored separately. Compare audio using the same sample rate,
window size, and hop size. Song matching is not implemented yet.

Run checks with `python3 -m unittest discover -s tests -v`.
