"""Route 1 at the trainer level: checkpoint backward compatibility of the failure map, failure attribution to
a3_boss2, the won-run HP label in the rollout, and the worker command line. Needs torch (skipped otherwise)."""
from collections import Counter, defaultdict, deque
from pathlib import Path
import json
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "train"))
from sts2ai import curriculum, ppo, shm
from sts2ai.model import ModelConfig
from sts2ai.sampling import situation_weights


def header(**kw):
    h = np.zeros(shm.HEADER_INTS, np.int32)
    for k, v in kw.items():
        h[getattr(shm, "H_" + k.upper())] = v
    return h


def stub_trainer(n_sit: int, n_envs: int = 4, **cfg):
    cfg = ppo.PPOConfig(**{"focus": 1, "boss2_situation": int(n_sit == 13), "init_mode": "resume", "device": "cpu",
                           "n_envs": n_envs, **cfg})
    tr = object.__new__(ppo.Trainer)
    tr.cfg, tr.dev, tr.mcfg = cfg, torch.device("cpu"), ModelConfig()
    tr.model = torch.nn.Linear(2, 2)
    tr.opt = torch.optim.AdamW(tr.model.parameters())
    tr.learner = types.SimpleNamespace(ema=None, ema_state_dict=lambda: None)
    tr.n_sit = n_sit
    tr.bosses = curriculum.FinalBossTracker(n_envs)
    tr.update_rng, tr.fmap_rng, tr.asc_rng = (np.random.default_rng(s) for s in (1, 2, 3))
    tr.asc_level = tr.iter = tr.total_steps = tr.trained_steps = 0
    tr.initial_model_sha256 = None
    tr.prefix_actions_observed = tr.restart_requested_episodes = tr.restart_actual_episodes = 0
    tr.completed_episodes = tr.engine_errors_total = 0
    tr.fmap_loss = np.zeros(n_sit)
    tr.fmap_w = situation_weights(tr.fmap_loss, cfg.focus_sampler, cfg.focus_floor)
    tr._fmap_clear()
    return tr


def old_checkpoint(tmp_path, n_sit=12):
    """A checkpoint as written before route 1 (12 situations, no n_sit key)."""
    tr = stub_trainer(n_sit)
    tr.fmap_loss = np.arange(1, n_sit + 1, dtype=float)
    tr.fmap_w = situation_weights(tr.fmap_loss)
    tr.fmap_tries, tr.fmap_wins, tr.fmap_dec = np.full(n_sit, 4.0), np.full(n_sit, 1.0), np.arange(n_sit, dtype=float)
    path = tmp_path / f"old{n_sit}.pt"
    tr.save(str(path))
    ck = torch.load(path, weights_only=False)
    del ck["n_sit"]
    torch.save(ck, path)
    return path, tr


def test_old_12_situation_checkpoint_resumes_with_13(tmp_path, capsys):
    path, old = old_checkpoint(tmp_path)
    tr = stub_trainer(13)
    tr.load(str(path))
    log = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    resized = [l["failmap_resized"] for l in log if "failmap_resized" in l]
    assert resized and resized[0]["from"] == [12] and resized[0]["to"] == 13
    np.testing.assert_array_equal(tr.fmap_loss, list(old.fmap_loss) + [0.0])
    np.testing.assert_allclose(tr.fmap_w, situation_weights(tr.fmap_loss))
    assert tr.fmap_w.shape == (13,) and tr.fmap_w.sum() == pytest.approx(1)
    for key in ("fmap_tries", "fmap_wins", "fmap_dec"):
        np.testing.assert_array_equal(getattr(tr, key), list(getattr(old, key)) + [0.0])
    # and the resumed trainer can draw all 13 situations and save them
    assert tr.fmap_rng.choice(tr.n_sit, size=5, p=tr.fmap_w).max() < 13
    tr.save(str(tmp_path / "new.pt"))
    assert torch.load(tmp_path / "new.pt", weights_only=False)["n_sit"] == 13


def test_same_size_checkpoint_loads_unchanged_and_13_folds_back_into_12(tmp_path, capsys):
    path, old = old_checkpoint(tmp_path)
    tr = stub_trainer(12)
    tr.load(str(path))
    assert "failmap_resized" not in capsys.readouterr().out
    np.testing.assert_array_equal(tr.fmap_loss, old.fmap_loss)
    np.testing.assert_array_equal(tr.fmap_w, old.fmap_w)
    path13, old13 = old_checkpoint(tmp_path, 13)
    tr = stub_trainer(12)
    tr.load(str(path13))
    assert "failmap_resized" in capsys.readouterr().out
    assert tr.fmap_loss.shape == (12,) and tr.fmap_loss[11] == old13.fmap_loss[11] + old13.fmap_loss[12]


class FakePool:
    def __init__(self):
        self.resets = []

    def reset(self, ids, **kw):
        self.resets.append((np.asarray(ids).copy(), kw))


def finishing_trainer(n_sit, **cfg):
    tr = stub_trainer(n_sit, **cfg)
    tr.stats = defaultdict(lambda: deque(maxlen=1000))
    tr.deaths = Counter()
    tr.episode = np.zeros(tr.cfg.n_envs, np.int64)
    tr.pool = FakePool()
    return tr


def boss2_death(restart=0):
    return header(done=1, won=0, act=2, roomtype=3, bosskills=3, restart=restart, hp=0, maxhp=80)


@pytest.mark.parametrize("n_sit,slot", [(12, 11), (13, 12)])
def test_boss2_death_is_credited_to_a3_boss2(n_sit, slot):
    tr = finishing_trainer(n_sit)
    boss1_death = header(done=1, won=0, act=2, roomtype=3, bosskills=2, hp=0, maxhp=80)
    tr._finish_episodes(np.array([0, 1]), np.stack([boss2_death(), boss1_death]))
    fs = tr.cfg.focus_strategic
    expected = np.zeros(n_sit)
    expected[8] = 2 * fs
    expected[11] += 1 - fs
    expected[slot] += 1 - fs
    np.testing.assert_allclose(tr.fmap_round, expected)
    ids, kw = tr.pool.resets[0]
    assert kw["situation"].max() < n_sit


def test_practice_wins_use_the_fight_outcome_only_with_the_flag():
    for fight_outcome, expected in ((0, 0.0), (1, 1.0)):
        tr = finishing_trainer(13, fight_outcome=fight_outcome)
        # a boss-1 practice episode (a3_boss) that reaches boss 2 at 75% HP and then dies there
        for h in (header(incombat=1, act=2, roomtype=3, bosskills=2, combat_hp_start=70, maxhp=80, restart=12),
                  header(incombat=1, act=2, roomtype=3, bosskills=3, combat_hp_start=60, maxhp=80, restart=12)):
            tr.bosses.observe(np.array([0]), h[None])
        tr._finish_episodes(np.array([0]), boss2_death(restart=12)[None])
        assert tr.fmap_tries[11] == 1 and tr.fmap_wins[11] == expected


def test_won_run_hp_label_in_the_rollout():
    for fix, expected in ((1, 9 / 78), (0, 0.0)):
        tr = stub_trainer(12, n_envs=1, hp_label_fix=fix)
        tr.cbuf = ppo.Rollout(4)
        tr.episode = np.zeros(1, np.int64)
        tr.ev_keys, tr.ev_hp = [], []
        last = header(incombat=1, hp=9, maxhp=78, combatid=24, act=2, roomtype=3, bosskills=3, nact=2)
        # the last in-combat decision is stored as row 0, awaiting the terminal header
        tr.cbuf.ckey[0] = tr._ckey(np.array([0]), last[None])[0]
        tr.cbuf.size = 1
        tr.pend_row = np.array([0])
        tr.prev_head = last[None].copy()
        term = header(done=1, won=1, hp=0, maxhp=78, combatid=24, act=2, roomtype=6, bosskills=4)
        tr._close_pending(np.array([0]), term[None])
        tr._label_combat_hp()
        assert tr.cbuf.t_hp[0] == pytest.approx(expected)
        assert tr.cbuf.done[0] and tr.cbuf.r_wonind[0] == 1


def run_main(monkeypatch, *args):
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
    monkeypatch.setattr(sys, "argv", ["ppo", "--sim-dll", "x.dll", "--gamebin", "gb", "--restart-prob", "0.5", *args])
    ppo.main()
    return seen


def test_worker_command_line_is_unchanged_by_default_and_carries_the_flags(monkeypatch):
    base = run_main(monkeypatch)["wcfg"].extra_args
    assert base == ["--restart-prob", "0.5", "--practice-sampler", "uncertainty", "--seed-protocol", "legacy"]
    on = run_main(monkeypatch, "--boss2-situation", "1", "--boss2-hp-stratified", "1", "--fight-outcome", "1",
                  "--practice-repeats", "4")["wcfg"].extra_args
    assert on == base + ["--boss2-situation", "1", "--boss2-hp-stratified", "1", "--fight-outcome", "1",
                         "--practice-repeats", "4"]


@pytest.mark.parametrize("args", [["--boss2-hp-stratified", "1"], ["--practice-repeats", "0"], ["--fight-outcome", "2"]])
def test_invalid_route1_flags_are_rejected(monkeypatch, args):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *args)
