"""Inject synthetic faults into the real trace, the same five kinds (and defaults) as
``incubator-telemetry-analysis/src/incubator/faults.py`` so the two repos' results compare.

The real day has no known fault, so faults of known kind, position and length are added:

``stuck_sensor``     the temperature sensor repeats its last value for 15 min
``spike``            one reading is off by +1.5 °C
``heater_stuck_on``  the relay is closed for 10 min although the firmware logs it open
``heater_stuck_off`` the relay is open for 20 min although the firmware logs it closed
``door_open``        the loss to ambient is x4 for 15 min

The three physical faults change the chamber temperature in a way the real trace cannot show,
so their effect is synthesised with the second-order thermal model fitted in repo #1 and added
to the measured temperature (the model is linear, so the *deviation* can be simulated alone).
Parameters are copied from that repo's ``figures/model_params.json`` (``second_order``).
Repo #1's sixth kind, ``dropout`` (no readings for 10 min), is not a job for this model: the
firmware's sampling watchdog catches it, so it is left out here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinyml.load import Trace

# incubator-telemetry-analysis, figures/model_params.json -> "second_order" (physical branch):
#   dE/dt = (a*u - E)/tau_h ;  dT/dt = E - b*(T - T_out)
A_C_PER_MIN = 0.5908
TAU_H_MIN = 4.9175
B_PER_MIN = 0.0111

KINDS = ("stuck_sensor", "spike", "heater_stuck_on", "heater_stuck_off", "door_open")
#: duration in readings (15 s each) and magnitude, as in repo #1
DEFAULTS: dict[str, tuple[int, float]] = {
    "stuck_sensor": (60, 0.0),
    "spike": (1, 1.5),
    "heater_stuck_on": (40, 0.0),
    "heater_stuck_off": (80, 0.0),
    "door_open": (60, 4.0),
}
PRE_MARGIN = 4  # readings (1 min) before the fault that count as "affected"
SETTLE_MARGIN = 20  # readings (5 min) after it; the chamber takes minutes to recover


@dataclass(frozen=True)
class Fault:
    kind: str
    start: int  # reading index
    length: int  # readings
    magnitude: float = 0.0

    @property
    def end(self) -> int:
        return self.start + self.length - 1


def make_fault(kind: str, start: int) -> Fault:
    if kind not in KINDS:
        raise ValueError(f"unknown fault kind {kind!r}; choose from {KINDS}")
    length, mag = DEFAULTS[kind]
    return Fault(kind, start, length, mag)


def deviation(trace: Trace, fault: Fault) -> np.ndarray:
    """Temperature deviation (°C per reading) the thermal model predicts for a physical fault:
    with ``du`` the change in relay state and ``m`` the loss multiplier inside the window,
    dDE/dt = (a*du - DE)/tau_h ; dD/dt = DE - m*b*D - (m-1)*b*(T_meas - T_out).
    """
    n = len(trace)
    minutes = trace.minutes
    inside = np.zeros(n, dtype=bool)
    inside[fault.start : fault.end + 1] = True
    u = trace.heater.astype(np.float64)
    du = np.zeros(n)
    mult = np.ones(n)
    if fault.kind == "heater_stuck_on":
        du[inside] = 1.0 - u[inside]
    elif fault.kind == "heater_stuck_off":
        du[inside] = -u[inside]
    elif fault.kind == "door_open":
        mult[inside] = fault.magnitude
    dev = np.zeros(n)
    d, de = 0.0, 0.0
    for k in range(1, n):
        step = minutes[k] - minutes[k - 1]
        forcing = (mult[k - 1] - 1.0) * B_PER_MIN * (trace.temp[k - 1] - trace.temp_out[k - 1])
        target = A_C_PER_MIN * du[k - 1]
        de_next = target + (de - target) * np.exp(-step / TAU_H_MIN)
        d = d + step * (0.5 * (de + de_next) - mult[k - 1] * B_PER_MIN * d - forcing)
        de = de_next
        dev[k] = d
    return dev


def inject(trace: Trace, fault: Fault) -> Trace:
    """The trace with the fault applied. Temperatures stay rounded to 0.01 °C like the sensor."""
    temp = trace.temp.copy()
    if fault.kind in ("heater_stuck_on", "heater_stuck_off", "door_open"):
        temp = np.round(temp + deviation(trace, fault), 2)
    elif fault.kind == "stuck_sensor":
        temp[fault.start : fault.end + 1] = temp[fault.start]
    elif fault.kind == "spike":
        temp[fault.start] = round(temp[fault.start] + fault.magnitude, 2)
    return trace.replace(temp=temp)


def affected_windows(fault: Fault, n_readings: int, window: int) -> np.ndarray:
    """Boolean per window (aligned with features rows): the window's last reading lies in
    [start - PRE_MARGIN, end + SETTLE_MARGIN]. A flag there counts as a detection."""
    ends = np.arange(window - 1, n_readings)
    return (ends >= fault.start - PRE_MARGIN) & (ends <= fault.end + SETTLE_MARGIN)


def random_starts(n_readings: int, length: int, n: int, window: int, seed: int) -> list[int]:
    """``n`` start indices so the window warm-up, the fault and the settle margin fit."""
    rng = np.random.default_rng(seed)
    lo = window + PRE_MARGIN  # the first full window must exist before the fault
    hi = n_readings - length - SETTLE_MARGIN - 1
    if hi <= lo:
        raise ValueError("trace too short for this fault")
    return sorted(int(x) for x in rng.integers(lo, hi + 1, size=n))
