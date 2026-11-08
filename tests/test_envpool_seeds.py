from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "train"))
from sts2ai.envpool import CMD_RESET, EnvPool


def test_explicit_game_seeds_reach_the_worker_protocol_unchanged():
    pool = object.__new__(EnvPool)
    pool.rng = np.random.default_rng(999)
    sent = []
    pool._send = lambda *args: sent.append(args)
    seeds = np.array([1567772636522628672, 3187160234521354990], dtype=np.int64)
    pool.reset(np.array([0, 1], dtype=np.int32), game_seeds=seeds)
    assert sent[0][1] == CMD_RESET
    np.testing.assert_array_equal(sent[0][3], seeds)


@pytest.mark.parametrize("seeds", [[1], [-1, 2], [1.5, 2], [2**63, 2**63]])
def test_invalid_seeds_are_not_coerced_or_sent(seeds):
    pool = object.__new__(EnvPool)
    pool._send = lambda *args: pytest.fail("invalid seed was sent")
    with pytest.raises(ValueError):
        pool.reset(np.array([0, 1], dtype=np.int32), game_seeds=seeds)


def test_reserved_eval_seeds_are_never_generated_for_training():
    pool = object.__new__(EnvPool)
    pool.rng = np.random.default_rng(7)
    pool.seed_upper_bound = 8
    pool.reserved_seeds = np.array([2, 4, 6], dtype=np.int64)
    sent = []
    pool._send = lambda *args: sent.append(args)
    pool.reset(np.arange(2000, dtype=np.int32))
    assert set(sent[0][3].tolist()) <= {1, 3, 5, 7}
