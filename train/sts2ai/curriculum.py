"""Restart-curriculum helpers for the double-boss (A10) changes; numpy only, so tests run without torch.

Situations (Worker.cs): sit = act*4 + kind (kind 0 act start, 1 monster, 2 elite, 3 boss), 12 in all. With
--boss2-situation the second final boss of a double-boss act (A10 act 3) is situation 12, "a3_boss2" (13 in all).
The trainer tells the two final bosses apart by the boss kills in the header: in act 3 the first boss is fought
with 2 boss kills, the second with 3 (the worker uses the map's second-boss node; both rules agree).
"""
from __future__ import annotations

import numpy as np

from . import shm

H = shm
N_SIT, N_SIT_BOSS2, SIT_BOSS2 = shm.N_SIT, shm.N_SIT_BOSS2, shm.SIT_BOSS2
A3_BOSS_KIND, A3_BOSS2_KIND = 2 * 4 + 3 + 1, SIT_BOSS2 + 1   # header H_RESTART values (situation + 1)
BOSS2_HP_TARGET = 0.5   # fight-level outcome of a boss-1 point: reach boss 2 with at least this share of max HP


def n_situations(boss2_situation: bool) -> int:
    return N_SIT_BOSS2 if boss2_situation else N_SIT


def situation_index(act, kind, boss_kills, n_sit: int) -> np.ndarray:
    """Situation of a fight (kind 1..3) or act start (kind 0); a boss fight in act 3 after three boss kills is
    a3_boss2 when there are 13 situations."""
    act = np.clip(np.asarray(act), 0, 2)
    kind = np.asarray(kind)
    sit = act * 4 + kind
    if n_sit > SIT_BOSS2:
        sit = np.where((act == 2) & (kind == 3) & (np.asarray(boss_kills) >= 3), SIT_BOSS2, sit)
    return sit


def final_boss(heads: np.ndarray) -> np.ndarray:
    """Per header row: -1 outside the final bosses, 0 in the (first) act-3 boss fight, 1 in the second one."""
    in_boss = (heads[:, H.H_INCOMBAT] != 0) & (np.clip(heads[:, H.H_ACT], 0, 2) == 2) & (heads[:, H.H_ROOMTYPE] == 3)
    return np.where(in_boss, (heads[:, H.H_BOSSKILLS] >= 3).astype(np.int64), -1)


def hp_quarter(frac) -> np.ndarray:
    """Entry-HP quarter as in Worker.cs SavePoint.HpBucket: [0, .25), [.25, .5), [.5, .75), [.75, 1]."""
    return np.clip(np.floor(np.asarray(frac, np.float64) * 4), 0, 3).astype(np.int64)


def resize_situations(values, n_sit: int) -> np.ndarray:
    """Per-situation failure-map array from a checkpoint written with 12 or 13 situations, as n_sit entries.
    12 -> 13: a3_boss2 starts at 0 (the old a3_boss entry, which mixed both final bosses, stays a3_boss).
    13 -> 12: a3_boss2 is folded back into a3_boss."""
    a = np.asarray(values, np.float64).reshape(-1)
    if len(a) == n_sit:
        return a.copy()
    if len(a) == N_SIT and n_sit == N_SIT_BOSS2:
        return np.append(a, 0.0)
    if len(a) == N_SIT_BOSS2 and n_sit == N_SIT:
        b = a[:N_SIT].copy()
        b[SIT_BOSS2 - 1] += a[SIT_BOSS2]
        return b
    raise ValueError(f"cannot map {len(a)} failure-map situations to {n_sit}")


def combat_end_hp(prev: np.ndarray, cur: np.ndarray, fix_won: bool = True):
    """End-of-combat HP label of the transitions prev -> cur (headers [B, 32]).

    Returns (fin, hpf): fin marks transitions that end the combat of `prev`, hpf the HP fraction that labels
    that combat (HP at the next decision / max HP; 0 for a lost run). The terminal header of a won run reports
    HP 0 (the game's victory sequence), so with fix_won a won run's last fight is labelled with the HP at its
    last in-combat decision instead (before the fix it was labelled 0, like a loss)."""
    ended = (cur[:, H.H_DONE] != 0) | (cur[:, H.H_ERROR] != 0)
    fin = (prev[:, H.H_INCOMBAT] != 0) & ((cur[:, H.H_INCOMBAT] == 0) | (cur[:, H.H_COMBATID] != prev[:, H.H_COMBATID]) | ended)
    won = cur[:, H.H_WON] != 0
    hp = cur[:, H.H_HP]
    if fix_won:
        hp = np.where(ended & won, prev[:, H.H_HP], hp)
    hpf = np.where(~ended | won, hp / np.maximum(1, prev[:, H.H_MAXHP]), 0.0)
    return fin, hpf


class FinalBossTracker:
    """Per-environment record of the final-boss fights of the current episode, from the decision headers:
    HP fraction and boss kills at each boss's first decision. At the episode's end it gives each fight's outcome
    (boss b is won when the boss kills grew after its entry, or the run was won) and the fight-level practice
    outcome, and accumulates the round's boss-1 / boss-2 statistics for the failure-map log."""

    def __init__(self, n_envs: int):
        self.entry_hp = np.full((n_envs, 2), np.nan)
        self.entry_bk = np.zeros((n_envs, 2), np.int64)
        self.clear_round()

    def clear_round(self):
        # [boss, (fresh run reaching it, practice episode started at it), (n, fights won, entry HP sum)]
        self.round = np.zeros((2, 2, 3))
        self.practice_quarters = np.zeros((2, 4), np.int64)   # practice entry-HP quarters per boss point

    def observe(self, envs: np.ndarray, heads: np.ndarray):
        """Live decisions: remember each final boss's entry state the first time it is seen."""
        b = final_boss(heads)
        m = b >= 0
        if not m.any():
            return
        e, b, h = envs[m], b[m], heads[m]
        new = np.isnan(self.entry_hp[e, b])
        e, b, h = e[new], b[new], h[new]
        self.entry_hp[e, b] = h[:, H.H_COMBAT_HP_START] / np.maximum(1, h[:, H.H_MAXHP])
        self.entry_bk[e, b] = h[:, H.H_BOSSKILLS]

    def finish(self, envs: np.ndarray, heads: np.ndarray, fight_outcome: bool) -> np.ndarray:
        """Terminal headers of these envs' episodes. Returns the practice success per episode: the run won, or
        (fight_outcome) for a3_boss / a3_boss2 practice the fight-level outcome of the point's boss (as Worker.cs).
        Records the round statistics (episodes without an engine error) and resets the envs."""
        won = heads[:, H.H_WON] != 0
        ok = heads[:, H.H_ERROR] == 0
        bk = heads[:, H.H_BOSSKILLS]
        restart = heads[:, H.H_RESTART]
        hp, ebk = self.entry_hp[envs], self.entry_bk[envs]
        seen = ~np.isnan(hp)
        fight_won = seen & ((bk[:, None] > ebk) | won[:, None])
        # final-boss practice: the point is boss 1 if the episode played boss 1, else boss 2
        final_pt = np.isin(restart, (A3_BOSS_KIND, A3_BOSS2_KIND)) & seen.any(1)
        point = np.where(seen[:, 0], 0, 1)
        success = won.copy()
        if fight_outcome:
            s_boss1 = won | (seen[:, 1] & (np.nan_to_num(hp[:, 1], nan=-1.0) >= BOSS2_HP_TARGET))
            success = np.where(final_pt, np.where(point == 0, s_boss1, fight_won[:, 1]), won)
        for b in (0, 1):
            fresh = ok & (restart == 0) & seen[:, b]
            prac = ok & final_pt & (point == b)
            for g, m in ((0, fresh), (1, prac)):
                self.round[b, g] += (m.sum(), fight_won[m, b].sum(), hp[m, b].sum())
            np.add.at(self.practice_quarters[b], hp_quarter(hp[prac, b]), 1)
        self.entry_hp[envs] = np.nan
        self.entry_bk[envs] = 0
        return success

    def summary(self) -> dict:
        """boss1/boss2: fresh runs reaching the boss and practice episodes started at it: n, fight win%, mean
        entry HP% (of max), and the practice entry-HP quarters."""
        out = {}
        for b, name in ((0, "boss1"), (1, "boss2")):
            d = {}
            for g, tag in ((0, "fresh"), (1, "practice")):
                n, w, s = self.round[b, g]
                d[f"{tag}_n"] = int(n)
                d[f"{tag}_win%"] = round(100 * w / max(n, 1), 1)
                d[f"{tag}_entry_hp%"] = round(100 * s / max(n, 1), 1)
            d["practice_entry_hp_quarters"] = self.practice_quarters[b].tolist()
            out[name] = d
        return out
