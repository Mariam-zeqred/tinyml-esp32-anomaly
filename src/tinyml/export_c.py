"""Turn the int8 TFLite model into the C sources the firmware and the host test compile.

    python -m tinyml.export_c

Writes into ``lib/anomaly_core/``:
    model_data.cc / .h   the .tflite bytes as a C array (what TensorFlow Lite Micro loads)
    model_params.h       window constants, input/output quantisation, threshold, tolerance
    model_weights.h      the same weights dequantised to float, for the host reference kernel
    test_vectors.h       raw reading segments with the scores the TFLite interpreter gives them

The threshold comes from ``figures/results.json`` (``tinyml.evaluate``). The tolerances written
to ``model_params.h`` are measured: the host test and the sketch's self-check must agree with
the int8 interpreter to within ``max(ABS, REL * score)``, where ABS is 1.5x the worst gap of the
float reference kernel (dequantised weights, int8 input/output clipping emulated) over the real
day and REL is 1.5x its worst relative gap over the fault segments.
"""

from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tinyml import faults as fl  # noqa: E402
from tinyml.evaluate import EXAMPLE_START, RESULTS_FILE  # noqa: E402
from tinyml.load import ROOT, Trace, load_trace  # noqa: E402
from tinyml.train import TFLITE_FILE, run_tflite, score  # noqa: E402
from tinyml.windows import DELTA_SCALE_C, N_FEATURES, N_TARGETS, WINDOW, features  # noqa: E402

OUT_DIR = ROOT / "lib" / "anomaly_core"
LAYERS = ("enc", "code", "dec", "recon")


def read_layers(tflite: bytes) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Dequantised (W[out, in], b[out]) per layer plus the input/output quantisation."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    details = {d["name"]: d for d in interp.get_tensor_details()}
    layers = []
    for name in LAYERS:
        w_d = next(
            d for n, d in details.items() if f"/{name}_1/MatMul" in n and len(d["shape"]) == 2
        )
        b_d = next(d for n, d in details.items() if f"/{name}_1/BiasAdd" in n)
        w_q = interp.get_tensor(w_d["index"]).astype(np.float64)
        b_q = interp.get_tensor(b_d["index"]).astype(np.float64)
        w = w_q * w_d["quantization_parameters"]["scales"][:, None]
        b = b_q * b_d["quantization_parameters"]["scales"]
        layers.append((w, b))
    inp = interp.get_input_details()[0]["quantization"]
    out = interp.get_output_details()[0]["quantization"]
    io = {"in_scale": inp[0], "in_zero": inp[1], "out_scale": out[0], "out_zero": out[1]}
    return layers, io


def reference(layers, io: dict[str, float], x: np.ndarray) -> np.ndarray:
    """What ``RefModel.cpp`` computes: the input rounded and clipped like the int8 input
    tensor, float dense layers (tanh except on the last one), the output clipped to the range
    the int8 output tensor can hold. Inside the network nothing is rounded, which is the only
    difference left from the interpreter."""
    q = np.clip(np.round(x / io["in_scale"]) + io["in_zero"], -128, 127)
    h = (q - io["in_zero"]) * io["in_scale"]
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.tanh(h)
    lo = (-128 - io["out_zero"]) * io["out_scale"]
    hi = (127 - io["out_zero"]) * io["out_scale"]
    return np.clip(h, lo, hi)


def c_array(name: str, values: np.ndarray, ctype: str, per_line: int = 8, fmt: str = "{}") -> str:
    flat = values.ravel()
    rows = [
        "  " + ", ".join(fmt.format(v) for v in flat[i : i + per_line]) + ","
        for i in range(0, len(flat), per_line)
    ]
    return f"static const {ctype} {name}[{len(flat)}] = {{\n" + "\n".join(rows) + "\n};\n"


def segments(trace: Trace) -> list[tuple[str, Trace, int, int]]:
    """(name, trace to read from, first reading, last reading) for the test vectors."""
    out = [("clean_cooldown_val", trace, 150, 213), ("clean_steady_38_8_val", trace, 1100, 1163)]
    for kind, lo, hi in (
        ("spike", 960, 1023),
        ("heater_stuck_on", 980, 1075),
        ("door_open", 980, 1091),
    ):
        out.append((kind, fl.inject(trace, fl.make_fault(kind, EXAMPLE_START)), lo, hi))
    return out


def main() -> None:
    tflite = TFLITE_FILE.read_bytes()
    results = json.loads(RESULTS_FILE.read_text())
    threshold = float(results["threshold"]["value"])
    trace = load_trace()
    layers, io = read_layers(tflite)

    # tolerance = max(absolute, relative * interpreter score): the absolute part is the worst
    # reference-vs-interpreter gap over the real day (scores near the threshold), the relative
    # part the worst ratio over the fault segments (scores far above it).
    x_all = features(trace.temp, trace.heater)
    s_int8 = score(x_all, run_tflite(tflite, x_all))
    worst_abs = float(np.abs(s_int8 - score(x_all, reference(layers, io, x_all))).max())
    worst_rel = 0.0
    for _, src, lo, hi in segments(trace):
        x_seg = features(src.temp[lo : hi + 1], src.heater[lo : hi + 1])
        s_seg = score(x_seg, run_tflite(tflite, x_seg))
        gap = np.abs(s_seg - score(x_seg, reference(layers, io, x_seg)))
        worst_rel = max(worst_rel, float((gap / np.maximum(s_seg, threshold)).max()))
    tol_abs, tol_rel = worst_abs * 1.5, worst_rel * 1.5
    gen = "// Generated by `python -m tinyml.export_c`; do not edit.\n"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "model_data.h").write_text(
        gen + "#pragma once\n"
        "extern const unsigned char g_anomaly_model_tflite[];\n"
        "extern const unsigned int g_anomaly_model_tflite_len;\n"
    )
    body = c_array(
        "model_bytes", np.frombuffer(tflite, dtype=np.uint8), "unsigned char", 16, "0x{:02x}"
    )
    body = body.replace(
        "static const unsigned char model_bytes",
        "alignas(16) const unsigned char g_anomaly_model_tflite",
    )
    (OUT_DIR / "model_data.cc").write_text(
        gen
        + '#include "model_data.h"\n\n'
        + body
        + f"const unsigned int g_anomaly_model_tflite_len = {len(tflite)};\n"
    )

    params = [
        f"#define AN_WINDOW {WINDOW}",
        f"#define AN_N_FEATURES {N_FEATURES}",
        f"#define AN_N_TARGETS {N_TARGETS}",
        f"#define AN_DELTA_SCALE_C {DELTA_SCALE_C}f",
        f"#define AN_IN_SCALE {io['in_scale']:.9g}f",
        f"#define AN_IN_ZERO {int(io['in_zero'])}",
        f"#define AN_OUT_SCALE {io['out_scale']:.9g}f",
        f"#define AN_OUT_ZERO {int(io['out_zero'])}",
        f"#define AN_THRESHOLD {threshold:.9g}f  // {results['threshold']['rule']}",
        f"#define AN_SCORE_TOL_ABS {tol_abs:.3g}f  // 1.5 x worst |float ref - int8|, real day",
        f"#define AN_SCORE_TOL_REL {tol_rel:.3g}f  // 1.5 x worst relative gap, fault segments",
        f"#define AN_MODEL_BYTES {len(tflite)}",
        f"#define AN_N_PARAMS {sum(w.size + b.size for w, b in layers)}",
    ]
    (OUT_DIR / "model_params.h").write_text(gen + "#pragma once\n" + "\n".join(params) + "\n")

    weights = [
        gen,
        "#pragma once",
        "// Dequantised int8 weights of model/anomaly_ae_int8.tflite, W[out][in] row-major.",
        "",
    ]
    dims = []
    for name, (w, b) in zip(LAYERS, layers, strict=True):
        dims.append(f"#define AN_{name.upper()}_OUT {w.shape[0]}")
        weights.append(c_array(f"AN_W_{name.upper()}", w.astype(np.float32), "float", 8, "{:.8e}f"))
        weights.append(c_array(f"AN_B_{name.upper()}", b.astype(np.float32), "float", 8, "{:.8e}f"))
    (OUT_DIR / "model_weights.h").write_text("\n".join(weights[:3] + dims + [""] + weights[3:]))

    vec = [
        gen,
        "#pragma once",
        "#include <stdint.h>",
        '#include "model_params.h"',
        "",
        "// Raw reading segments (real day, two with an injected fault) and the score the TFLite",
        "// int8 interpreter gives each complete window. `sure` = 1 where the score is more than",
        "// 25 % away from the threshold, so rounding differences cannot flip the verdict.",
        "typedef struct {",
        "  const char* name;",
        "  int n;",
        "  const float* temp;",
        "  const uint8_t* heater;",
        "  const float* score;  // n - AN_WINDOW + 1 entries",
        "  const uint8_t* flag;",
        "  const uint8_t* sure;",
        "} an_test_segment_t;",
        "",
    ]
    names = []
    n_windows = 0
    for name, src, lo, hi in segments(trace):
        temp, heater = src.temp[lo : hi + 1], src.heater[lo : hi + 1]
        s = score(features(temp, heater), run_tflite(tflite, features(temp, heater)))
        n_windows += len(s)
        vec.append(c_array(f"seg_{name}_temp", temp.astype(np.float32), "float", 8, "{:.2f}f"))
        vec.append(c_array(f"seg_{name}_heater", heater.astype(np.uint8), "uint8_t", 16))
        vec.append(c_array(f"seg_{name}_score", s.astype(np.float32), "float", 6, "{:.6e}f"))
        vec.append(c_array(f"seg_{name}_flag", (s > threshold).astype(np.uint8), "uint8_t", 16))
        vec.append(
            c_array(
                f"seg_{name}_sure",
                (np.abs(s - threshold) > 0.25 * threshold).astype(np.uint8),
                "uint8_t",
                16,
            )
        )
        fields = ", ".join(f"seg_{name}_{f}" for f in ("temp", "heater", "score", "flag", "sure"))
        names.append(f'  {{"{name}", {len(temp)}, {fields}}},')
    vec.append(f"#define AN_N_TEST_SEGMENTS {len(names)}")
    vec.append(
        "static const an_test_segment_t AN_TEST_SEGMENTS[AN_N_TEST_SEGMENTS] = {\n"
        + "\n".join(names)
        + "\n};\n"
    )
    (OUT_DIR / "test_vectors.h").write_text("\n".join(vec))

    summary = {
        "tflite_bytes": len(tflite),
        "n_params": int(sum(w.size + b.size for w, b in layers)),
        "ref_vs_int8_abs_diff_max_clean_day": worst_abs,
        "ref_vs_int8_rel_diff_max_fault_segments": worst_rel,
        "score_tolerance": {"abs": tol_abs, "rel": tol_rel},
        "test_windows": n_windows,
        "written": sorted(p.name for p in OUT_DIR.iterdir() if p.suffix in (".h", ".cc")),
    }
    (ROOT / "model" / "export_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
