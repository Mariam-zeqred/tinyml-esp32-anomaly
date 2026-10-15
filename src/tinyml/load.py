"""Read the incubator history export into NumPy arrays.

The file ``data/hive-1C5A00_history_2026-10-03.json`` is a copy of the real trace analysed in
the ``incubator-telemetry-analysis`` repo: 1 367 readings at ~15 s from one queen-bee incubator
on 3 Oct 2026 (one device, one day, door shut, no known fault). Fields used here: ``ts`` (epoch
seconds), ``temp`` (inside °C), ``heater`` (relay on/off), ``targetTemp``, ``tempOut``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HISTORY_FILE = ROOT / "data" / "hive-1C5A00_history_2026-10-03.json"


@dataclass(frozen=True)
class Trace:
    """One day of readings, sorted by time. All arrays have the same length."""

    ts: np.ndarray  # epoch seconds (int64)
    temp: np.ndarray  # inside temperature, °C
    heater: np.ndarray  # relay state, 0/1 (int8)
    target: np.ndarray  # setpoint, °C
    temp_out: np.ndarray  # outside temperature, °C

    def __len__(self) -> int:
        return int(len(self.ts))

    @property
    def minutes(self) -> np.ndarray:
        """Minutes since the first reading (used by the fault model)."""
        return (self.ts - self.ts[0]) / 60.0

    def replace(self, **kw: np.ndarray) -> Trace:
        return Trace(**{**self.__dict__, **kw})


def load_trace(path: str | Path = HISTORY_FILE) -> Trace:
    with Path(path).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    rows = sorted(raw, key=lambda r: r["ts"])
    # the cloud export can repeat a row when the app re-syncs; keep the first
    seen: set[int] = set()
    rows = [r for r in rows if not (r["ts"] in seen or seen.add(r["ts"]))]
    return Trace(
        ts=np.array([r["ts"] for r in rows], dtype=np.int64),
        temp=np.array([r["temp"] for r in rows], dtype=np.float64),
        heater=np.array([1 if r["heater"] else 0 for r in rows], dtype=np.int8),
        target=np.array([r["targetTemp"] for r in rows], dtype=np.float64),
        temp_out=np.array([r["tempOut"] for r in rows], dtype=np.float64),
    )
