import numpy as np

from tinyml import windows as w
from tinyml.load import load_trace


def test_feature_layout_and_level_invariance():
    temp = np.array([30.0, 30.1, 30.3, 30.2] + [30.2] * (w.WINDOW - 4))
    heater = np.array([1, 1, 0, 0] + [0] * (w.WINDOW - 4))
    x = w.features(temp, heater)
    assert x.shape == (1, w.N_FEATURES)
    np.testing.assert_allclose(x[0, :3], [0.05, 0.15, 0.1], atol=1e-6)
    assert list(x[0, w.WINDOW - 1 : w.WINDOW + 3]) == [1, 1, 0, 0]
    y = w.features(temp + 8.8, heater)  # same shape at another setpoint
    np.testing.assert_allclose(x, y, atol=1e-5)


def test_windows_align_with_readings():
    n = w.WINDOW + 5
    x = w.features(np.linspace(20, 21, n), np.zeros(n))
    assert len(x) == 6
    assert len(w.window_ends(n)) == 6
    assert w.window_ends(n)[0] == w.WINDOW - 1
    assert w.features(np.zeros(w.WINDOW - 1), np.zeros(w.WINDOW - 1)).shape == (0, w.N_FEATURES)


def test_split_has_no_shared_readings():
    n = 1367
    mask = w.split_mask(n)
    ends = w.window_ends(n)
    starts = ends - w.WINDOW + 1
    train_readings = set()
    for s, e in zip(starts[mask == 0], ends[mask == 0], strict=True):
        train_readings.update(range(s, e + 1))
    for s, e in zip(starts[mask == 1], ends[mask == 1], strict=True):
        assert not train_readings.intersection(range(s, e + 1))
    assert (mask == 0).sum() > 400 and (mask == 1).sum() > 400


def test_episodes():
    assert w.episodes([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert w.episodes([]) == []


def test_real_trace_loads():
    t = load_trace()
    assert len(t) == 1367
    assert np.all(np.diff(t.ts) > 0)
    assert set(np.unique(t.heater)) <= {0, 1}
    assert 26 < t.temp.min() < t.temp.max() < 40
