"""Calibrate the threshold on clean validation windows, inject faults, score, draw figures.

    python -m tinyml.evaluate      # -> figures/results.json, figures/01_*.png, figures/02_*.png

Needs ``model/anomaly_ae_int8.tflite`` from ``tinyml.train``. Every number in the README's
results tables comes from ``figures/results.json`` written here.
"""

from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tinyml import faults as fl  # noqa: E402
from tinyml.load import ROOT, Trace, load_trace  # noqa: E402
from tinyml.train import KERAS_FILE, TFLITE_FILE, run_tflite, score  # noqa: E402
from tinyml.windows import WINDOW, episodes, split_mask, trace_features, window_ends  # noqa: E402

FIG_DIR = ROOT / "figures"
RESULTS_FILE = FIG_DIR / "results.json"

#: The threshold is this quantile of the int8 scores on clean *validation* windows.
THRESHOLD_QUANTILE = 0.995
TRIALS_PER_KIND = 20
SEED = 0
#: Fault start (reading index) used for the example figure and the C test vectors.
EXAMPLE_START = 1000


def scores_for(trace: Trace, tflite: bytes) -> np.ndarray:
    x = trace_features(trace)
    return score(x, run_tflite(tflite, x))


def trial(
    trace: Trace, tflite: bytes, fault: fl.Fault, threshold: float, clean_flags: np.ndarray
) -> dict[str, float | bool | int]:
    s = scores_for(fl.inject(trace, fault), tflite)
    flags = s > threshold
    affected = fl.affected_windows(fault, len(trace), WINDOW)
    hit = flags & affected
    ends = window_ends(len(trace))
    before = ends < fault.start - fl.PRE_MARGIN
    delay_min = float((ends[hit][0] - fault.start) * 15 / 60) if hit.any() else None
    return {
        "detected": bool(hit.any()),
        "detected_null": bool((clean_flags & affected).any()),
        "delay_min": delay_min,
        "peak_score_ratio": float(s[affected].max() / threshold),
        # flags before the fault cannot be caused by it: pure false alarms. Flags after the
        # settle margin are mostly the chamber still recovering (tau ~ 90 min), see README.
        "fp_episodes_before": len(episodes(flags & before)),
        "fp_episodes_after": len(episodes(flags & ~affected & ~before)),
    }


def summarise(rows: list[dict]) -> dict[str, float | int | None]:
    delays = [r["delay_min"] for r in rows if r["delay_min"] is not None]
    return {
        "trials": len(rows),
        "detection_rate": float(np.mean([r["detected"] for r in rows])),
        "null_detection_rate": float(np.mean([r["detected_null"] for r in rows])),
        "median_delay_min": float(np.median(delays)) if delays else None,
        "median_peak_score_ratio": float(np.median([r["peak_score_ratio"] for r in rows])),
        "fp_episodes_before_fault_per_trial": float(
            np.mean([r["fp_episodes_before"] for r in rows])
        ),
        "episodes_after_settle_per_trial": float(np.mean([r["fp_episodes_after"] for r in rows])),
    }


def float_vs_int8(x: np.ndarray, s_int8: np.ndarray, threshold: float) -> dict[str, float | int]:
    """How often int8 rounding changes the verdict on the clean day."""
    import tensorflow as tf

    model = tf.keras.models.load_model(KERAS_FILE, compile=False)
    s_float = score(x, model.predict(x, verbose=0))
    return {
        "windows": int(len(x)),
        "decision_flips": int(((s_float > threshold) != (s_int8 > threshold)).sum()),
        "abs_diff_p50": float(np.median(np.abs(s_float - s_int8))),
        "abs_diff_max": float(np.abs(s_float - s_int8).max()),
    }


def draw(trace: Trace, s_clean: np.ndarray, threshold: float, mask: np.ndarray, tflite: bytes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)
    minutes = trace.minutes
    ends = window_ends(len(trace))

    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(minutes, trace.temp, lw=1, label="inside °C")
    ax[0].plot(minutes, trace.target, lw=1, ls="--", label="target")
    ax[0].set_ylabel("°C")
    ax[0].legend(loc="lower right")
    ax[1].step(minutes, trace.heater, lw=0.8)
    ax[1].set_ylabel("heater")
    ax[1].set_yticks([0, 1])
    ax[2].plot(minutes[ends], s_clean, lw=0.8, label="int8 score")
    ax[2].axhline(threshold, color="r", lw=1, label=f"threshold {threshold:.3f}")
    for i0, i1 in episodes(mask == 1):
        ax[2].axvspan(minutes[ends[i0]], minutes[ends[i1]], color="0.9", zorder=0)
    ax[2].set_ylabel("MSE")
    ax[2].set_xlabel("minutes since first reading (grey: validation blocks)")
    ax[2].legend(loc="upper right")
    fig.suptitle("Clean day: 1 367 real readings, 1 336 windows, no fault injected")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_clean_day_scores.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(len(fl.KINDS), 2, figsize=(12, 2.4 * len(fl.KINDS)), sharex=True)
    for row, kind in enumerate(fl.KINDS):
        fault = fl.make_fault(kind, EXAMPLE_START)
        bad = fl.inject(trace, fault)
        s = scores_for(bad, tflite)
        lo, hi = max(0, fault.start - 160), min(len(trace) - 1, fault.end + 160)
        sel = slice(lo, hi)
        axes[row, 0].plot(minutes[sel], trace.temp[sel], lw=1, color="0.6", label="real")
        axes[row, 0].plot(minutes[sel], bad.temp[sel], lw=1, color="C3", label="with fault")
        axes[row, 0].axvspan(minutes[fault.start], minutes[fault.end], color="C3", alpha=0.12)
        axes[row, 0].set_ylabel(f"{kind}\n°C", fontsize=9)
        wsel = (ends >= lo) & (ends <= hi)
        axes[row, 1].plot(minutes[ends][wsel], s_clean[wsel], lw=0.8, color="0.6", label="real")
        axes[row, 1].plot(minutes[ends][wsel], s[wsel], lw=0.8, color="C3", label="with fault")
        axes[row, 1].axhline(threshold, color="r", lw=1)
        axes[row, 1].axvspan(minutes[fault.start], minutes[fault.end], color="C3", alpha=0.12)
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_ylabel("score (log)", fontsize=9)
        if row == 0:
            axes[row, 0].legend(loc="upper left", fontsize=8)
    axes[-1, 0].set_xlabel("minutes")
    axes[-1, 1].set_xlabel("minutes")
    fig.suptitle(f"One injected fault of each kind at reading {EXAMPLE_START} (default length)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_fault_examples.png", dpi=110)
    plt.close(fig)


def main() -> None:
    trace = load_trace()
    tflite = TFLITE_FILE.read_bytes()
    mask = split_mask(len(trace))
    s_clean = scores_for(trace, tflite)
    threshold = float(np.quantile(s_clean[mask == 1], THRESHOLD_QUANTILE))
    clean_flags = s_clean > threshold

    per_kind: dict[str, dict] = {}
    for kind in fl.KINDS:
        length, _ = fl.DEFAULTS[kind]
        starts = fl.random_starts(len(trace), length, TRIALS_PER_KIND, WINDOW, SEED)
        rows = [
            trial(trace, tflite, fl.make_fault(kind, st), threshold, clean_flags) for st in starts
        ]
        per_kind[kind] = summarise(rows)
        print(kind, per_kind[kind])

    ends = window_ends(len(trace))
    results = {
        "model": TFLITE_FILE.name,
        "tflite_bytes": len(tflite),
        "threshold": {
            "value": threshold,
            "rule": f"quantile {THRESHOLD_QUANTILE} of int8 scores on clean validation windows",
            "val_windows": int((mask == 1).sum()),
        },
        "clean_day": {
            "windows": int(len(s_clean)),
            "flagged_windows_all": int(clean_flags.sum()),
            "flagged_windows_val": int(clean_flags[mask == 1].sum()),
            "false_alarm_episodes_all": [
                {"minute": round(float(trace.minutes[ends[a]]), 1), "windows": b - a + 1}
                for a, b in episodes(clean_flags)
            ],
            "hours": round(float(trace.minutes[-1] / 60), 2),
            "score_median": float(np.median(s_clean)),
            "score_max": float(s_clean.max()),
        },
        "float_vs_int8_clean_day": float_vs_int8(trace_features(trace), s_clean, threshold),
        "faults": {
            "trials_per_kind": TRIALS_PER_KIND,
            "seed": SEED,
            "defaults_readings_magnitude": {k: list(v) for k, v in fl.DEFAULTS.items()},
            "thermal_model": {"a": fl.A_C_PER_MIN, "tau_h": fl.TAU_H_MIN, "b": fl.B_PER_MIN},
            "per_kind": per_kind,
        },
    }
    FIG_DIR.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")
    draw(trace, s_clean, threshold, mask, tflite)
    print(json.dumps({k: v for k, v in results.items() if k != "faults"}, indent=2))


if __name__ == "__main__":
    main()
