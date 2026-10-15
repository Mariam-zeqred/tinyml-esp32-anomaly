import numpy as np
import pytest

from tinyml import faults as fl
from tinyml.load import load_trace
from tinyml.windows import WINDOW


@pytest.fixture(scope="module")
def trace():
    return load_trace()


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        fl.make_fault("dropout", 100)


def test_physical_faults_push_the_right_way(trace):
    on = fl.inject(trace, fl.make_fault("heater_stuck_on", 700)).temp - trace.temp
    off = fl.inject(trace, fl.make_fault("heater_stuck_off", 700)).temp - trace.temp
    door = fl.inject(trace, fl.make_fault("door_open", 700)).temp - trace.temp
    assert np.all(on[:700] == 0) and on.max() > 3 and on.min() >= 0
    assert np.all(off[:700] == 0) and off.min() < -3 and off.max() <= 0
    assert np.all(door[:700] == 0) and door.min() < -1 and door.max() <= 0
    # the deviation decays after the fault but is still there an hour later (tau ~ 90 min)
    assert on[700 + 40 + 240] > 0.5 * on.max()


def test_sensor_faults_are_local(trace):
    stuck = fl.inject(trace, fl.make_fault("stuck_sensor", 500))
    assert np.all(stuck.temp[500:560] == trace.temp[500])
    assert np.array_equal(stuck.temp[:500], trace.temp[:500])
    assert np.array_equal(stuck.temp[560:], trace.temp[560:])
    spike = fl.inject(trace, fl.make_fault("spike", 500))
    assert spike.temp[500] == pytest.approx(trace.temp[500] + 1.5)
    assert np.count_nonzero(spike.temp != trace.temp) == 1


def test_affected_windows_and_starts(trace):
    f = fl.make_fault("spike", 500)
    aff = fl.affected_windows(f, len(trace), WINDOW)
    ends = np.arange(WINDOW - 1, len(trace))
    assert aff[ends == 500 - fl.PRE_MARGIN].all()
    assert aff[ends == 500 + fl.SETTLE_MARGIN].all()
    assert not aff[ends == 500 + fl.SETTLE_MARGIN + 1].any()
    starts = fl.random_starts(len(trace), 80, 20, WINDOW, seed=0)
    assert len(starts) == 20 and min(starts) >= WINDOW
    assert max(starts) + 80 + fl.SETTLE_MARGIN < len(trace)
    assert starts == fl.random_starts(len(trace), 80, 20, WINDOW, seed=0)
