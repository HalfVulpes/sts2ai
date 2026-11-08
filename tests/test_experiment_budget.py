import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("budget", Path(__file__).parents[1] / "train/sts2ai/experiment_budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def test_async_overshoot_does_not_change_optimizer_budget():
    assert budget.update_rows(65540, 65536) == budget.update_rows(65790, 65536) == 65536
    assert 763 * 65536 == 50003968


def test_short_rollout_is_not_silently_treated_as_full_budget():
    with pytest.raises(ValueError):
        budget.update_rows(65535, 65536)


def test_training_range_is_disjoint_from_reserved_test_range():
    assert budget.seed_upper_bound(61) == 2**61
    assert budget.seed_upper_bound(61) < 2**62
    with pytest.raises(ValueError):
        budget.seed_upper_bound(64)
