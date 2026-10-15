"""Windowing and features. Mirrors ``lib/anomaly_core/Window.cpp`` exactly: the host test
feeds raw readings through the C code and compares with vectors produced here.

A window is the last ``WINDOW`` readings (32 x 15 s = 8 min). Its feature vector has
``N_FEATURES = 2*WINDOW - 1 = 63`` entries:

    f[0 .. WINDOW-2]      (temp[k] - temp[0]) / DELTA_SCALE_C   for k = 1 .. WINDOW-1
    f[WINDOW-1 .. 2W-2]   heater[k]                              for k = 0 .. WINDOW-1

Differences, not absolute temperature, so a window at 30 °C and one at 38.8 °C look alike when
the chamber behaves alike. The heater channel is *context*: the autoencoder sees it but only
reconstructs the temperature part, so it learns "the heater was on, so the temperature should
start rising a few minutes later" (the heater lag is ~5 min, see repo #1) without having to
reproduce a 0/1 pattern through a 4-unit bottleneck, which first attempts showed it cannot.
"""

from __future__ import annotations

import numpy as np

from tinyml.load import Trace

WINDOW = 32
N_FEATURES = 2 * WINDOW - 1
N_TARGETS = WINDOW - 1  # the model reconstructs only the temperature part, f[0 .. WINDOW-2]
DELTA_SCALE_C = 2.0  # 8-min temperature changes on the real day span -0.57 .. +3.50 °C

# Train/validation split: contiguous blocks of this many readings, alternating; windows that
# straddle two blocks are dropped so no reading is shared between the two sets.
BLOCK = 120  # 30 min


def features(temp: np.ndarray, heater: np.ndarray) -> np.ndarray:
    """Feature vectors for every complete window; row i ends at reading i + WINDOW - 1."""
    temp = np.asarray(temp, dtype=np.float64)
    heater = np.asarray(heater, dtype=np.float64)
    n = len(temp) - WINDOW + 1
    if n <= 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    idx = np.arange(WINDOW)[None, :] + np.arange(n)[:, None]
    t = temp[idx]
    h = heater[idx]
    deltas = (t[:, 1:] - t[:, :1]) / DELTA_SCALE_C
    return np.concatenate([deltas, h], axis=1).astype(np.float32)


def window_ends(n_readings: int) -> np.ndarray:
    """Index of the last reading of each window, aligned with :func:`features` rows."""
    return np.arange(WINDOW - 1, n_readings)


def split_mask(n_readings: int, block: int = BLOCK) -> np.ndarray:
    """Per window: 0 = train, 1 = validation, -1 = dropped (straddles two blocks)."""
    ends = window_ends(n_readings)
    starts = ends - WINDOW + 1
    b_start, b_end = starts // block, ends // block
    out = np.where(b_end % 2 == 0, 0, 1)
    out[b_start != b_end] = -1
    return out


def trace_features(trace: Trace) -> np.ndarray:
    return features(trace.temp, trace.heater)


def episodes(flags: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive True values as (first, last) window indices."""
    flags = np.asarray(flags, dtype=bool)
    out: list[tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out
