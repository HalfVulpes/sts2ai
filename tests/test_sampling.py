import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("sampling", Path(__file__).parents[1] / "train/sts2ai/sampling.py")
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


def test_uniform_control_cannot_adapt_to_losses():
    loss = np.zeros(12)
    loss[2] = 100
    np.testing.assert_allclose(sampling.situation_weights(loss, "uniform"), np.full(12, 1 / 12))


def test_failure_mass_targets_new_situation_but_keeps_support():
    loss = np.zeros(12)
    loss[2] = 100
    w = sampling.situation_weights(loss)
    assert np.argmax(w) == 2
    assert np.all(w > 0)
    assert w.sum() == pytest.approx(1)
    # static control retains the initialization despite new failures
    np.testing.assert_allclose(sampling.situation_weights(loss, "static"),
                               sampling.situation_weights(np.zeros(12)))


@pytest.mark.parametrize("loss", [np.zeros(11), [float("nan")] * 12, [-1] * 12])
def test_invalid_failure_counts_fail_loudly(loss):
    with pytest.raises(ValueError):
        sampling.situation_weights(loss)


def test_no_unavailable_state_and_same_restart_rate_for_both_arms():
    masks = np.full(20000, (1 << 0) | (1 << 3), dtype=np.int64)
    weights = np.arange(1, 13, dtype=float)
    out = sampling.available_situations(np.random.default_rng(2), weights, masks, .5)
    assert set(out.tolist()) <= {-1, 0, 3}
    assert .48 < np.mean(out >= 0) < .52
    assert np.mean(out[out >= 0] == 3) > .75
    empty = sampling.available_situations(np.random.default_rng(2), weights, np.zeros(20, dtype=int), 1)
    assert np.all(empty == -1)
