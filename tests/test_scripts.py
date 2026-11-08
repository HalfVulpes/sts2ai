import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def value(cmd, flag):
    return cmd[len(cmd) - 1 - cmd[::-1].index(flag) + 1]  # last occurrence wins, as in argparse


def test_a10_push_preset_matches_the_paper_recipe(tmp_path):
    train = load("train")
    cmd = train.command("a10_push", "defect", tmp_path, init=tmp_path / "defect.pt")
    assert value(cmd, "--ascension") == "10" and value(cmd, "--lr") == "0.0001" and value(cmd, "--iters") == "3072"
    assert value(cmd, "--restart-prob") == "0.5" and value(cmd, "--focus-sampler") == "failure"
    assert value(cmd, "--fixed-update-rows") == "65536" and value(cmd, "--init-mode") == "weights"
    assert value(cmd, "--hp-label-fix") == "0" and value(cmd, "--seed") == "711071"
    with pytest.raises(SystemExit):
        train.command("a10_push", "defect", tmp_path)


def test_controlled_study_arms_differ_only_in_restart_allocation(tmp_path):
    train = load("train")
    arms = {a: train.command(f"a0_{a}", "regent", tmp_path, seed=41001) for a in ("fresh", "uniform", "failure")}
    assert value(arms["fresh"], "--restart-prob") == "0" and value(arms["uniform"], "--focus-sampler") == "uniform"
    assert all(value(c, "--iters") == "763" and value(c, "--lr") == "0.0002" for c in arms.values())
    small = train.command("a0_failure", "regent", tmp_path, updates=1, extra=["--n-envs", "8"])
    assert value(small, "--iters") == "1" and value(small, "--n-envs") == "8"
    assert all(value(c, "--hp-label-fix") == "0" for c in arms.values())  # trained before the won-run label fix


# Trainer flags of the release continuation (the control arms whose final EMA are the released Ironclad, Defect and
# Silent policies), as recorded in their checkpoints; paths, per-segment targets and resume flags left out.
RELEASE_RUN = {"--preset": "base", "--n-envs": "256", "--steps-per-iter": "65536", "--fixed-update-rows": "65536",
               "--minibatch": "4096", "--epochs": "2", "--token-budget": "262144", "--ascension": "10",
               "--asc-max": "0", "--coop-prob": "0", "--lr": "0.0001", "--gamma": "0.9998", "--lam": "0.99",
               "--env-seed": "-1", "--seed-protocol": "canonical", "--train-seed-bits": "48",
               "--train-seed-limit": "2064377754059776", "--fail-nonfinite": "1", "--save-initial": "1",
               "--save-every": "8", "--iters": "1800", "--restart-prob": "0.5", "--focus": "1",
               "--focus-sampler": "failure", "--practice-sampler": "uncertainty", "--mask-unavailable-restarts": "1",
               "--hp-label-fix": "0"}


def test_release_continuation_preset_matches_the_recorded_runs(tmp_path):
    train = load("train")
    for character, seed in (("ironclad", "811090"), ("defect", "811092"), ("silent", "811093")):
        cmd = train.command("a10_release", character, tmp_path, init=tmp_path / "base.pt",
                            reserved_seeds=tmp_path / "reserved_seeds.json")
        assert {flag: value(cmd, flag) for flag in RELEASE_RUN} == RELEASE_RUN
        assert value(cmd, "--seed") == seed and value(cmd, "--characters") == character
        assert value(cmd, "--init-mode") == "weights" and "--boss2-situation" not in cmd
    assert value(train.command("a10_release", "regent", tmp_path, init=tmp_path / "r.pt", seed=7), "--seed") == "7"


def test_route1_presets(tmp_path):
    train = load("train")
    longer = train.command("a10_longer", "regent", tmp_path, init=tmp_path / "regent.pt")
    boss2 = train.command("a10_boss2", "regent", tmp_path, init=tmp_path / "regent.pt")
    assert value(longer, "--iters") == value(boss2, "--iters") == "600"
    assert value(longer, "--seed") == value(boss2, "--seed") == "711091"
    assert value(longer, "--hp-label-fix") == "0" and "--boss2-situation" not in longer
    assert [value(boss2, f) for f in ("--boss2-situation", "--boss2-hp-stratified", "--fight-outcome",
                                      "--practice-repeats", "--hp-label-fix")] == ["1", "1", "1", "4", "1"]
    assert value(train.command("a10_push", "necrobinder", tmp_path, init=tmp_path / "n.pt"), "--seed") == "711072"


def test_every_preset_is_accepted_by_the_trainer(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "train"))
    from sts2ai import ppo
    train = load("train")
    seen = {}

    class FakeTrainer:
        def __init__(self, cfg, wcfg):
            seen["cfg"], seen["wcfg"] = cfg, wcfg

        def run(self):
            pass

        def save(self, path):
            pass

        def close(self):
            pass

    monkeypatch.setattr(ppo, "Trainer", FakeTrainer)
    for preset in train.PRESETS:
        cmd = train.command(preset, "regent", tmp_path, init=tmp_path / "regent.pt", device="cpu")
        monkeypatch.setattr(sys, "argv", ["ppo", *cmd[cmd.index("sts2ai.ppo") + 1:]])
        ppo.main()
        cfg, extra = seen["cfg"], seen["wcfg"].extra_args
        assert cfg.iters == train.PRESETS[preset]["updates"] and cfg.characters == "regent"
        assert cfg.hp_label_fix == (1 if preset == "a10_boss2" else 0)
        assert ("--boss2-situation" in extra) == (preset == "a10_boss2")


def test_wilson_interval_and_panels(tmp_path):
    evaluate = load("evaluate")
    lo, hi = evaluate.wilson(317, 2560)
    assert round(100 * lo, 2) == 11.16 and round(100 * hi, 2) == 13.72  # paper, Table tab:a10, Ironclad
    assert evaluate.wilson(0, 0) == (0.0, 0.0)
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"test": ["AAAA000001", {"seed": "BBBB000002"}]}))
    assert evaluate.load_seeds("", panel, "test") == ["AAAA000001", "BBBB000002"]
    assert evaluate.load_seeds("X1, X2", None, "") == ["X1", "X2"]
