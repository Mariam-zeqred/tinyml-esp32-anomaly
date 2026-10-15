# Walkthrough (for Mariam)

## Why it is built this way

The incubator already streams temperature and heater state every 15 s. The question this repo
answers is whether a model small enough to live inside that firmware can tell "the chamber is
not responding the way it normally does" without any labelled fault. An autoencoder is the
simplest tool for that: train it to reproduce normal windows, and anything it reproduces badly
is suspicious. Using temperature *differences* makes the model indifferent to the setpoint, and feeding the heater pattern as context (but not
asking the model to reproduce it) lets it learn the heater lag. The first version reconstructed
the heater bits too and failed: a 4-unit bottleneck cannot copy 32 on/off values, so that
error swamped the temperature error. The committed version has a baseline ten times lower.

Everything that can be checked without a board is checked on the host, and the same C window
code runs in the test and in the sketch. The float reference kernel exists only so `make check`
can prove the exported bytes are the trained network; the ESP32 runs TensorFlow Lite Micro.

## File map

- `src/tinyml/windows.py` — the feature definition and the block-wise train/validation split.
  `lib/anomaly_core/Window.cpp` is its C twin; the host test ties them together.
- `src/tinyml/train.py` — Keras model, int8 export with a representative dataset, the
  float-vs-int8 check. `evaluate.py` — threshold, 100 fault trials, figures, `results.json`.
  `faults.py` — the five fault kinds, copied from repo #1 with its thermal parameters.
- `src/tinyml/export_c.py` — writes `model_data.cc` (the model), `model_params.h` (window,
  quantisation, threshold, tolerances), `model_weights.h` (reference kernel), `test_vectors.h`
  (245 recorded windows with the interpreter's scores).
- `src/Detector.cpp` — the TFLM interpreter with a two-op resolver. `src/Replay.cpp` — where
  readings come from; swap it for your sensor. `src/main.cpp` — boot self-check, then the loop.
- `test/test_native/test_core.cpp` — ring buffer, quantisation, score, model blob, and the
  245-window comparison.

## Three questions a professor may ask

**"You trained and set the threshold on the same day. Isn't the evaluation circular?"**
Partly, and the README says so. The split is by 30-minute blocks with straddling windows
dropped, so validation readings never enter training, and the threshold uses validation scores
only. But the faults are injected into that same day, and regimes seen once are shared. The
right next step is a second day, and the repo is set up so that one `make all` re-runs
everything on it.

**"Why does int8 quantisation matter for such a small model?"**
Because it changed a verdict. Calibrating the int8 ranges on training windows only left the
output unable to represent the fastest cool-down of the day, which lay in a validation block.
The clipped reconstruction scored 0.003 against 0.00014 in float (numbers from that discarded
export). The fix was to calibrate on all clean windows; the lesson is that the representative
dataset must cover the range the model will see, not just the training set.

**"How do you know the board runs the same model?"**
`tests/test_artifacts.py` checks that `model_data.cc` is byte-identical to the
`.tflite` file and that the header's threshold matches `results.json`. The host test feeds 245
recorded windows through the C window code and a float reference of the dequantised weights
and compares with the interpreter's scores within a measured tolerance. The sketch repeats that
comparison with TensorFlow Lite Micro at boot and prints PASS or FAIL.

## One thing you could extend

Add humidity as a second reconstructed channel. The window code, the exporter and the test
vectors are all written per feature index, so the change is the feature layout in
`windows.py` and `Window.cpp`, then `make all`. The 2026-10-03 trace already contains `hum`.
