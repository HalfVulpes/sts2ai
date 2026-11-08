import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("seeds", Path(__file__).parents[1] / "train/sts2ai/seed_protocol.py")
seeds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seeds)


@pytest.mark.parametrize("value", [0, 1, 2**48-1, 2**48, 2**49, 2**50-1, 34**10-1])
def test_seed_partition_remains_injective_after_text_conversion(value):
    assert seeds.to_number(seeds.from_number(value)) == value
    assert not set(seeds.from_number(value)) & set("IO")


def test_shared_external_panel_roundtrips():
    assert seeds.from_number(seeds.to_number("CHECK00001")) == "CHECK00001"
    with pytest.raises(ValueError):
        seeds.from_number(34**10)
