"""Route 1 (double-boss curriculum): situation indexing, sampling, fight outcomes and the won-run HP label.
numpy only; the trainer-level tests that need torch are in test_route1_trainer.py."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "train"))
from sts2ai import curriculum, sampling, shm
from sts2ai.envpool import EnvPool

H = shm


def header(**kw):
    h = np.zeros(shm.HEADER_INTS, np.int32)
    for k, v in kw.items():
        h[getattr(shm, "H_" + k.upper())] = v
    return h


def heads(*rows):
    return np.stack(rows)


# ---------------------------------------------------------------- situations
def test_twelve_situations_are_unchanged_and_boss2_is_the_thirteenth():
    assert shm.N_SIT == 12 and shm.N_SIT_BOSS2 == 13 and shm.SIT_BOSS2 == 12
    assert [shm.RESTART_KINDS[k] for k in (1, 4, 9, 12)] == ["a1_start", "a1_boss", "a3_start", "a3_boss"]
    assert shm.RESTART_KINDS[13] == "a3_boss2"
    assert curriculum.n_situations(False) == 12 and curriculum.n_situations(True) == 13
    assert (curriculum.A3_BOSS_KIND, curriculum.A3_BOSS2_KIND) == (12, 13)


def test_situation_index_splits_only_the_second_act3_boss():
    act = np.array([0, 1, 2, 2, 2, 2, 3, 2])
    kind = np.array([3, 3, 3, 3, 2, 0, 3, 1])
    bk = np.array([0, 1, 2, 3, 3, 3, 3, 3])
    # 12 situations: exactly the old act*4+kind (act clipped to 0..2)
    np.testing.assert_array_equal(curriculum.situation_index(act, kind, bk, 12), [3, 7, 11, 11, 10, 8, 11, 9])
    # 13: a boss fight in act 3 after three boss kills (the second final boss) is a3_boss2; elites/starts are not
    np.testing.assert_array_equal(curriculum.situation_index(act, kind, bk, 13), [3, 7, 11, 12, 10, 8, 12, 9])


def test_final_boss_index_from_headers():
    rows = heads(header(incombat=1, act=2, roomtype=3, bosskills=2), header(incombat=1, act=2, roomtype=3, bosskills=3),
                 header(incombat=0, act=2, roomtype=3, bosskills=3), header(incombat=1, act=1, roomtype=3, bosskills=1),
                 header(incombat=1, act=2, roomtype=2, bosskills=3))
    np.testing.assert_array_equal(curriculum.final_boss(rows), [0, 1, -1, -1, -1])


def test_hp_quarters_match_the_worker_integer_rule():
    for maxhp in (1, 7, 70, 78, 80, 99):
        for hp in range(0, maxhp + 1):
            assert curriculum.hp_quarter(hp / maxhp) == min(3, hp * 4 // maxhp), (hp, maxhp)


def test_reset_argument_encodes_a3_boss2_and_the_worker_can_decode_it():
    pool = object.__new__(EnvPool)
    pool.rng = np.random.default_rng(1)
    pool.seed_upper_bound = 2**40
    pool.reserved_seeds = np.array([], dtype=np.int64)
    sent = []
    pool._send = lambda *args: sent.append(args)
    pool.reset(np.array([0, 1, 2], np.int32), ascension=np.array([10, 10, 10]), situation=np.array([-1, 11, 12]))
    a = sent[0][2]
    # Worker.cs: sitField = a >> 12 (1 fresh, s+2 situation s); low 12 bits = (asc+1)*8 + character (7 random)
    np.testing.assert_array_equal(a >> 12, [1, 13, 14])
    np.testing.assert_array_equal(a & 0xFFF, [11 * 8 + 7] * 3)
    assert a.dtype == np.int32 and np.all(a > 0)


# ---------------------------------------------------------------- sampling
def test_thirteen_situation_weights():
    np.testing.assert_allclose(sampling.situation_weights(np.zeros(13), "uniform"), np.full(13, 1 / 13))
    w = sampling.situation_weights(np.zeros(13))
    assert w.shape == (13,) and w.sum() == pytest.approx(1) and np.all(w > 0)
    assert np.argsort(w)[-4:].tolist() == [7, 12, 11, 8]   # static start: a3_start > a3_boss > a3_boss2 = a2_boss
    loss = np.zeros(13)
    loss[12] = 5
    w = sampling.situation_weights(loss)
    assert np.argmax(w) == 12
    # 12-situation behaviour is unchanged
    w12 = sampling.situation_weights(np.zeros(12))
    expected = np.zeros(12)
    expected[[8, 11, 7]] = [0.35, 0.45, 0.2]
    np.testing.assert_allclose(w12, (expected + 0.02) / (expected + 0.02).sum())


def test_available_situations_with_the_boss2_bit():
    weights = np.ones(13)
    masks = np.full(20000, (1 << 12) | (1 << 11), dtype=np.int64)
    out = sampling.available_situations(np.random.default_rng(3), weights, masks, 1.0)
    assert set(out.tolist()) == {11, 12}
    assert .45 < np.mean(out == 12) < .55
    with pytest.raises(ValueError):   # bit 12 is not a situation when there are 12
        sampling.available_situations(np.random.default_rng(3), np.ones(12), np.array([1 << 12]), 1.0)
    with pytest.raises(ValueError):
        sampling.available_situations(np.random.default_rng(3), np.ones(13), np.array([1 << 13]), 1.0)
    with pytest.raises(ValueError):
        sampling.situation_weights(np.zeros(14))


# ---------------------------------------------------------------- checkpoint arrays
def test_resize_failure_map_arrays_between_12_and_13():
    old = np.arange(12, dtype=float)
    new = curriculum.resize_situations(old, 13)
    np.testing.assert_array_equal(new, list(range(12)) + [0])
    np.testing.assert_array_equal(curriculum.resize_situations(new, 13), new)
    back = curriculum.resize_situations(np.arange(13, dtype=float), 12)
    assert back[11] == 11 + 12 and back.sum() == np.arange(13).sum()
    with pytest.raises(ValueError):
        curriculum.resize_situations(np.zeros(11), 13)


# ---------------------------------------------------------------- end-of-combat HP label
def test_won_run_last_fight_label_uses_last_in_combat_hp():
    # the terminal header of a won A10 run, as the host reports it (verified on seed 0CTHB0GVM9): HP 0
    prev = header(incombat=1, hp=9, maxhp=78, combatid=24, act=2, roomtype=3, bosskills=3)
    term = header(done=1, won=1, hp=0, maxhp=78, combatid=24, act=2, roomtype=6, bosskills=4)
    fin, hpf = curriculum.combat_end_hp(heads(prev), heads(term), fix_won=False)
    assert fin[0] and hpf[0] == 0.0                       # the bug: labelled like a loss
    fin, hpf = curriculum.combat_end_hp(heads(prev), heads(term), fix_won=True)
    assert fin[0] and hpf[0] == pytest.approx(9 / 78)    # fixed


def test_other_combat_end_labels_are_unchanged_by_the_fix():
    p_fight = header(incombat=1, hp=50, maxhp=80, combatid=3)
    cases = [
        (header(incombat=0, hp=56, maxhp=80, combatid=3), True, 56 / 80),            # won fight, next decision
        (header(incombat=1, hp=40, maxhp=80, combatid=4), True, 40 / 80),            # boss 1 -> boss 2 directly
        (header(done=1, won=0, hp=0, maxhp=80, combatid=3), True, 0.0),              # death
        (header(incombat=1, hp=45, maxhp=80, combatid=3), False, None),              # same fight goes on
    ]
    for cur, fin_expected, label in cases:
        for fix in (False, True):
            fin, hpf = curriculum.combat_end_hp(heads(p_fight), heads(cur), fix_won=fix)
            assert fin[0] == fin_expected
            if label is not None:
                assert hpf[0] == pytest.approx(label)
    # outside combat nothing is labelled
    fin, _ = curriculum.combat_end_hp(heads(header(incombat=0, hp=50, maxhp=80)), heads(header(done=1, won=1)), True)
    assert not fin[0]


# ---------------------------------------------------------------- final-boss bookkeeping and fight outcomes
def boss_decision(bk, hp_start, maxhp=80, restart=0):
    return header(incombat=1, act=2, roomtype=3, bosskills=bk, combat_hp_start=hp_start, hp=hp_start, maxhp=maxhp,
                  restart=restart)


def terminal(won, bk, restart=0, error=0):
    return header(done=1, won=won, bosskills=bk, act=2, roomtype=6 if won else 3, restart=restart, error=error)


def play(tracker, env, decisions):
    for h in decisions:
        tracker.observe(np.array([env]), heads(h))


def test_tracker_fight_outcomes_and_practice_success():
    t = curriculum.FinalBossTracker(8)
    # env 0: fresh run, boss 1 at 72/80, boss 2 at 30/80, dies in boss 2
    play(t, 0, [boss_decision(2, 72), boss_decision(2, 72), boss_decision(3, 30)])
    # env 1: practice from a boss-1 point (a3_boss, kind 12): reaches boss 2 with 48/80 = 60%, then dies
    play(t, 1, [boss_decision(2, 70, restart=12), boss_decision(3, 48, restart=12)])
    # env 2: practice from a boss-1 point: reaches boss 2 with 32/80 = 40%, then dies
    play(t, 2, [boss_decision(2, 70, restart=12), boss_decision(3, 32, restart=12)])
    # env 3: practice from a boss-2 point (a3_boss2, kind 13), wins boss 2 (and the run)
    play(t, 3, [boss_decision(3, 20, restart=13)])
    # env 4: practice from a boss-2 point, dies
    play(t, 4, [boss_decision(3, 64, restart=13)])
    # env 5: boss-2 point practised in the shared a3_boss situation (12 situations), dies
    play(t, 5, [boss_decision(3, 64, restart=12)])
    # env 6: practice from an act-2 boss point (kind 8): run outcome only
    play(t, 6, [header(incombat=1, act=1, roomtype=3, bosskills=1, restart=8)])
    # env 7: practice from a boss-1 point, dies in boss 1
    play(t, 7, [boss_decision(2, 60, restart=12)])
    ends = heads(terminal(0, 3), terminal(0, 3, 12), terminal(0, 3, 12), terminal(1, 4, 13), terminal(0, 3, 13),
                 terminal(0, 3, 12), terminal(0, 1, 8), terminal(0, 2, 12))
    envs = np.arange(8)
    won = ends[:, H.H_WON] != 0
    # without --fight-outcome the practice outcome is the run
    t2 = curriculum.FinalBossTracker(8)
    t2.entry_hp[:], t2.entry_bk[:] = t.entry_hp, t.entry_bk
    np.testing.assert_array_equal(t2.finish(envs, ends, fight_outcome=False), won)
    success = t.finish(envs, ends, fight_outcome=True)
    np.testing.assert_array_equal(success, [False, True, False, True, False, False, False, False])
    s = t.summary()
    assert s["boss1"]["fresh_n"] == 1 and s["boss1"]["fresh_win%"] == 100.0 and s["boss1"]["fresh_entry_hp%"] == 90.0
    assert s["boss2"]["fresh_n"] == 1 and s["boss2"]["fresh_win%"] == 0.0 and s["boss2"]["fresh_entry_hp%"] == 37.5
    # practice started at boss 1: envs 1, 2, 7; boss 1 won in 1 and 2
    assert s["boss1"]["practice_n"] == 3 and s["boss1"]["practice_win%"] == pytest.approx(66.7)
    # practice started at boss 2: envs 3, 4, 5 at 25%, 80%, 80% -> quarters [0, 1, 0, 2]; one won
    assert s["boss2"]["practice_n"] == 3 and s["boss2"]["practice_win%"] == pytest.approx(33.3)
    assert s["boss2"]["practice_entry_hp_quarters"] == [0, 1, 0, 2]
    # finished envs are reset for their next episode
    assert np.all(np.isnan(t.entry_hp)) and not t.entry_bk.any()
    t.clear_round()
    assert t.summary()["boss2"]["practice_n"] == 0


def test_tracker_ignores_errored_episodes_and_a9_has_no_second_boss():
    t = curriculum.FinalBossTracker(2)
    play(t, 0, [boss_decision(2, 70, restart=12)])
    play(t, 1, [boss_decision(2, 70, restart=12)])
    # env 0: A9-style win of the single final boss; env 1: engine error
    ends = heads(terminal(1, 3, 12), terminal(0, 2, 12, error=1))
    success = t.finish(np.arange(2), ends, fight_outcome=True)
    assert success.tolist() == [True, False]
    s = t.summary()
    assert s["boss1"]["practice_n"] == 1 and s["boss2"]["practice_n"] == 0 and s["boss2"]["fresh_n"] == 0
