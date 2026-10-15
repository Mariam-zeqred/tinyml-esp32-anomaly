"""Train the autoencoder on *normal* windows and export it as a full-integer TFLite model.

    python -m tinyml.train     # -> model/anomaly_ae_int8.tflite, model/train_report.json

Architecture: 63 -> 16 -> 4 -> 16 -> 31 dense, tanh hidden units, linear output (~1.7 k
weights): the input is temperature deltas + heater pattern, the output is the temperature
deltas only. The anomaly score of a window is the mean squared reconstruction error over the
31 temperature features (normalised units). Seeded; deterministic on one machine.
"""

from __future__ import annotations

import json
import os
import random
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

from tinyml.load import ROOT, load_trace  # noqa: E402
from tinyml.windows import N_FEATURES, N_TARGETS, split_mask, trace_features  # noqa: E402

MODEL_DIR = ROOT / "model"
TFLITE_FILE = MODEL_DIR / "anomaly_ae_int8.tflite"
KERAS_FILE = MODEL_DIR / "anomaly_ae_float.keras"  # the float model, for the int8-vs-float check
REPORT_FILE = MODEL_DIR / "train_report.json"

HIDDEN = 16
BOTTLENECK = 4
EPOCHS = 400
BATCH = 32
LEARNING_RATE = 3e-3
SEED = 0


def score(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Mean squared error between the window's temperature features and their reconstruction
    (what the C code computes too)."""
    tgt = x[:, :N_TARGETS].astype(np.float64)
    return np.mean((tgt - x_hat.astype(np.float64)) ** 2, axis=1)


def build_model():
    import tensorflow as tf

    layers = tf.keras.layers
    inp = tf.keras.Input(shape=(N_FEATURES,), name="window")
    x = layers.Dense(HIDDEN, activation="tanh", name="enc")(inp)
    x = layers.Dense(BOTTLENECK, activation="tanh", name="code")(x)
    x = layers.Dense(HIDDEN, activation="tanh", name="dec")(x)
    out = layers.Dense(N_TARGETS, activation=None, name="recon")(x)
    model = tf.keras.Model(inp, out, name="anomaly_ae")
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    return model


def to_int8_tflite(model, x_rep: np.ndarray) -> bytes:
    """Full-integer quantisation: int8 weights, activations, input and output.

    ``x_rep`` sets the activation ranges (not the weights). It must cover the whole range the
    model will see: with training windows only, the output could not represent temperature
    drops faster than -0.37 °C per 8 min, and the one fast cool-down of the day (after the first
    overshoot, in a validation block) was clipped into a false alarm. So all clean windows of
    the day are used here, and the README says so.
    """
    import tensorflow as tf

    def representative():
        for i in range(len(x_rep)):
            yield [x_rep[i : i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def run_tflite(tflite: bytes, x: np.ndarray) -> np.ndarray:
    """Reconstructions from the int8 model, dequantised to float (one window at a time, like
    the microcontroller)."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    recon = np.zeros((len(x), N_TARGETS), dtype=np.float32)
    for i in range(len(x)):
        q = np.clip(np.round(x[i] / s_in) + z_in, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None, :])
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
        recon[i] = (y - z_out) * s_out
    return recon


def main() -> None:
    import tensorflow as tf

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    trace = load_trace()
    x = trace_features(trace)
    mask = split_mask(len(trace))
    x_train, x_val = x[mask == 0], x[mask == 1]

    model = build_model()
    t0 = time.time()
    hist = model.fit(
        x_train,
        x_train[:, :N_TARGETS],
        epochs=EPOCHS,
        batch_size=BATCH,
        shuffle=True,
        validation_data=(x_val, x_val[:, :N_TARGETS]),
        verbose=0,
    )
    train_s = time.time() - t0

    s_train_f = score(x_train, model.predict(x_train, verbose=0))
    s_val_f = score(x_val, model.predict(x_val, verbose=0))

    tflite = to_int8_tflite(model, x)
    MODEL_DIR.mkdir(exist_ok=True)
    TFLITE_FILE.write_bytes(tflite)
    model.save(KERAS_FILE)
    s_val_q = score(x_val, run_tflite(tflite, x_val))
    s_train_q = score(x_train, run_tflite(tflite, x_train))

    n_params = int(sum(int(np.prod(w.shape)) for w in model.get_weights()))
    report = {
        "data": "data/hive-1C5A00_history_2026-10-03.json (1 device, 1 day)",
        "windows": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "dropped": int((mask == -1).sum()),
        },
        "architecture": [N_FEATURES, HIDDEN, BOTTLENECK, HIDDEN, N_TARGETS],
        "parameters": n_params,
        "epochs": EPOCHS,
        "train_seconds": round(train_s, 1),
        "loss_final": {
            "train": float(hist.history["loss"][-1]),
            "val": float(hist.history["val_loss"][-1]),
        },
        "score_float": {
            "train_median": float(np.median(s_train_f)),
            "val_median": float(np.median(s_val_f)),
            "val_p99": float(np.quantile(s_val_f, 0.99)),
            "val_max": float(s_val_f.max()),
        },
        "score_int8": {
            "train_median": float(np.median(s_train_q)),
            "val_median": float(np.median(s_val_q)),
            "val_p99": float(np.quantile(s_val_q, 0.99)),
            "val_max": float(s_val_q.max()),
            "abs_diff_vs_float_val_p50": float(np.median(np.abs(s_val_q - s_val_f))),
            "abs_diff_vs_float_val_p99": float(np.quantile(np.abs(s_val_q - s_val_f), 0.99)),
            "abs_diff_vs_float_val_max": float(np.abs(s_val_q - s_val_f).max()),
        },
        "tflite_bytes": len(tflite),
        "tensorflow": tf.__version__,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
