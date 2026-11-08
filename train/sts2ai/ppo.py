"""PPO trainer for full Slay the Spire 2 runs on the headless real-game simulator.

Long-horizon credit assignment (the "wins fights, loses runs" problem) is addressed by:
  * one network for all decisions, trained on full-run returns (gamma ~ 0.9995 per decision);
  * reward = progress (floors / boss kills) + run win bonus, plus potential-based HP shaping
    that only redistributes credit in time (does not change the optimum);
  * advantages normalized separately for combat vs strategic decisions, and a separate entropy
    bonus for strategic decisions, so the ~15% strategic choices are not drowned out;
  * auxiliary heads predicting P(win), final floor and end-of-combat HP, which shape the
    shared representation toward run-level consequences. P(win) and final floor are trained as
    value functions (TD(lambda) targets), so every step gets a correct target even though a run
    spans many iterations;
  * run-history "global memory" tokens in the observation.

Collection is batched: the native pool (native/fastpool.c) returns a batch of ready workers, and
all bookkeeping below is one numpy operation per batch (no per-environment Python loops).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import curriculum, fastpool, shm, spec
from .envpool import EnvPool, WorkerConfig
from .model import PRESETS, ModelConfig, Sts2Net, count_params, expert_ids_from_vocab
from .sampling import situation_weights, available_situations
from .experiment_budget import update_rows, seed_upper_bound
from .seed_protocol import LIMIT as CANONICAL_SEED_LIMIT


@dataclass
class PPOConfig:
    preset: str = "base"
    n_envs: int = 128
    steps_per_iter: int = 65536
    gamma: float = 0.9995
    lam: float = 0.95
    lam_aux: float = 0.95        # lambda for the P(win) / final-floor value targets (gamma 1)
    lam_strategic: float = -1.0  # GAE lambda for strategic decisions' advantages (-1: same as lam)
    clip: float = 0.2
    vclip: float = 0.2
    epochs: int = 2
    minibatch: int = 4096
    lr: float = 2e-4
    weight_decay: float = 0.01
    ent_combat: float = 0.003
    ent_strategic: float = 0.01
    strat_pg_coef: float = 1.0   # policy-loss weight of strategic decisions (rare: card picks, shop, path, ...)
    explore_strategic: float = 0.0  # epsilon: share of strategic decisions taken uniformly at random
                                    # (behaviour log-prob of the mixture is stored, so PPO's ratio stays correct)
    vf_coef: float = 0.5
    aux_coef: float = 0.2
    max_grad_norm: float = 1.0
    # reward
    r_floor: float = 0.01
    r_boss: float = 0.1
    r_elite: float = 0.02
    r_win: float = 1.0
    r_death: float = -0.1
    hp_potential: float = 0.1
    # collection batching: act once this many workers are ready, or batch_wait_us after the first
    batch_min: int = 0           # 0 -> n_envs // 4
    batch_wait_us: int = 1500
    token_budget: int = 0        # update micro-batch size in padded tokens; 0 -> 32M / d_model
    compile: int = 0             # torch.compile the learner forward (dynamic shapes)
    ema_decay: float = 0.998     # Polyak average of the weights per optimizer step (saved as "ema"; 0 = off)
    coop_prob: float = 0.0       # share of fresh episodes played as co-op (all seats by the policy, team reward;
    coop_max_players: int = 3    # 2..max players); logged apart under "coop"
    asc_max: int = 0             # ascension curriculum up to this level (0 = off; episodes use --ascension)
    asc_promote: float = 0.65    # advance a level once fresh solo episodes at the current level win this often
    asc_window: int = 400        # ... over this many episodes
    asc_a0: float = 0.25         # share of episodes kept at ascension 0 (the rest: current level 60%, lower levels 40%)
    asc_dwell: int = 20_000_000  # minimum steps at a level before promotion (the synchronized start of all envs
                                 # makes early windows unrepresentative: quick deaths first, then a burst of wins)
    numa: str = ""               # worker NUMA placement "cpu:mem,mem;..." (see envpool.WorkerConfig.numa)
    restart_prob: float = 0.0    # share of episodes that start from a recorded act-3 start / boss fight (practice;
                                 # excluded from the logged win rate, logged per kind under "restart")
    # failure-map loop (needs restart_prob): every round the trainer maps where fresh runs are lost (situation =
    # act x {act start, monster, elite, boss}) and starts practice episodes there in proportion; logged as "failmap"
    focus: int = 0
    focus_sampler: str = "failure"  # failure, uniform, static; all use the same 12 situations
    practice_sampler: str = "uncertainty"  # within-situation: uncertainty or uniform
    focus_round: int = 10        # iterations per round
    focus_decay: float = 0.5     # weight of the earlier rounds in the map
    focus_strategic: float = 0.35  # share of a lost fight credited to its act start (the deck-building decisions)
    focus_floor: float = 0.02    # added to every situation's share before normalising (nothing is dropped)
    # double-boss curriculum (route 1, second-boss practice; sts2ai/curriculum.py, sim/Engine/Worker.cs); all off by default
    boss2_situation: int = 0     # 13 situations: the second final boss of a double-boss act (A10) is a3_boss2
    boss2_hp_stratified: int = 0  # a3_boss2 points drawn by entry-HP quarter (needs boss2_situation)
    fight_outcome: int = 0       # final-boss practice scored by its fight: a boss-1 point by reaching boss 2 with
                                 # HP >= 50% of max (or winning), a boss-2 point by winning boss 2; not the run
    practice_repeats: int = 1    # a drawn save point is replayed up to this many times before it can be evicted
    hp_label_fix: int = 1        # a won run's last fight is labelled with its last in-combat HP (the terminal header
                                 # reports HP 0 on a win); 0 restores the old label (0), only to reproduce older runs
    learner_devices: str = ""    # e.g. "cuda:1,cuda:2,cuda:3": one learner process per GPU (first = --device)
    # misc
    ascension: int = 0
    characters: str = "Ironclad"
    seed: int = 1
    device: str = "cuda"          # learner
    actor_device: str = ""        # rollout inference; a different GPU overlaps collection with training
    out: str = "runs/exp"
    iters: int = 100000
    save_every: int = 20
    init: str = ""
    init_mode: str = "resume"   # weights starts a new experiment, without optimizer/curriculum state
    init_weights: str = "model"  # model or ema (weights mode only)
    max_new_steps: int = 0       # bounded pilot budget; rounded up to a complete rollout update
    fixed_update_rows: int = 0  # train exactly this many rows; separately count collected overshoot
    save_initial: int = 0
    fail_nonfinite: int = 0
    train_seed_bits: int = 62   # legacy default; canonical screens set a finite limit and exclusions
    env_seed: int = -1          # optional new episode stream for a standardized segment restart
    seed_protocol: str = "legacy"
    mask_unavailable_restarts: int = 0
    train_seed_limit: int = 0
    reserved_seeds_file: str = ""
    expected_initial_sha256: str = ""


H = shm  # header index constants
CHARACTERS = ["Ironclad", "Silent", "Defect", "Necrobinder", "Regent"]  # order of RunDriver.CharacterNames


def potential(h: np.ndarray, cfg: PPOConfig) -> np.ndarray:
    """HP shaping potential per row of headers h [B, 32]; 0 after a real episode end."""
    ended = (h[:, H.H_DONE] != 0) & (h[:, H.H_ERROR] == 0)
    p = cfg.hp_potential * h[:, H.H_HP] / np.maximum(1, h[:, H.H_MAXHP])
    return np.where(ended, 0.0, p).astype(np.float32)


def step_reward(prev: np.ndarray, cur: np.ndarray, cfg: PPOConfig) -> np.ndarray:
    d = lambda k: np.maximum(0, cur[:, k] - prev[:, k])
    r = cfg.r_floor * d(H.H_TOTALFLOOR) + cfg.r_boss * d(H.H_BOSSKILLS) + cfg.r_elite * d(H.H_ELITEKILLS)
    ended = (cur[:, H.H_DONE] != 0) & (cur[:, H.H_ERROR] == 0)  # a simulator error truncates, it is not a death
    r = r + np.where(ended, np.where(cur[:, H.H_WON] != 0, cfg.r_win, cfg.r_death), 0.0)
    r = r + cfg.gamma * potential(cur, cfg) - potential(prev, cfg)
    return r.astype(np.float32)


class Rollout:
    """Preallocated rollout storage (max-size padded arrays); rows appended a batch at a time.

    With `shm_name` the arrays live in one POSIX shared-memory block, so learner processes on other
    GPUs can read the rollout (and the targets computed before each update) without copies."""

    @staticmethod
    def _fields(n: int):
        T, A = spec.MAX_TOKENS, spec.MAX_ACTIONS
        f32, nan = np.float32, np.nan
        return [
            ("tc", (n, T, spec.N_TOK_CAT), np.int16, 0), ("tn", (n, T, spec.F_TOK), np.float16, 0),
            ("ac", (n, A, spec.N_ACT_CAT), np.int16, 0), ("ar", (n, A, 2), np.int16, -1),
            ("an", (n, A, spec.F_ACT), np.float16, 0),
            ("ntok", (n,), np.int32, 0), ("nact", (n,), np.int32, 0), ("env", (n,), np.int32, 0),
            ("action", (n,), np.int64, 0), ("ckey", (n,), np.int64, 0), ("dtype", (n,), np.int8, 0),
            ("logp", (n,), f32, 0), ("value", (n,), f32, 0), ("vwin", (n,), f32, 0), ("vfloor_rem", (n,), f32, 0),
            ("floor_now", (n,), f32, 0), ("reward", (n,), f32, 0), ("r_wonind", (n,), f32, 0), ("r_floorinc", (n,), f32, 0),
            ("done", (n,), np.bool_, 0), ("trunc", (n,), np.bool_, 0),
            ("next_value", (n,), f32, nan), ("next_vwin", (n,), f32, nan), ("next_vfloor_rem", (n,), f32, nan),
            ("t_hp", (n,), f32, nan),
            # targets, computed by the main process before each update
            ("adv", (n,), f32, 0), ("ret", (n,), f32, 0), ("t_win", (n,), f32, 0), ("t_floor", (n,), f32, 0),
            ("strategic", (n,), np.bool_, 0),
            ("boss", (n,), np.int8, -1),  # final-boss fight of the row: -1 none, 0 boss 1, 1 boss 2 (for logging)
        ]

    def __init__(self, n: int, shm_name: str | None = None, create: bool = True):
        self.n = n
        fields = self._fields(n)
        offs, total = [], 0
        for _, shape, dt, _ in fields:
            offs.append(total)
            total += (int(np.prod(shape)) * np.dtype(dt).itemsize + 63) // 64 * 64
        self._shm = None
        if shm_name:
            from multiprocessing import shared_memory
            # attached copies (learner helpers) must not track the block: the creating process owns
            # and unlinks it, and spawned children share its resource tracker
            kw = {"track": False} if not create and sys.version_info >= (3, 13) else {}
            self._shm = shared_memory.SharedMemory(name=shm_name, create=create, size=total, **kw)
            mem = self._shm.buf
        else:
            mem = np.empty(total, np.uint8)
        for (name, shape, dt, fill), off in zip(fields, offs):
            arr = np.ndarray(shape, dt, buffer=mem, offset=off)
            if create:
                arr.fill(fill)
            setattr(self, name, arr)
        self.size = 0

    def close(self, unlink: bool):
        if self._shm is not None:
            for name, *_ in self._fields(0):
                self.__dict__.pop(name, None)
            try:
                self._shm.close()
                if unlink:
                    self._shm.unlink()
            except Exception:
                pass
            self._shm = None

    def reset(self):
        self.size = 0

    def add(self, obs: dict, heads: np.ndarray, envs: np.ndarray, ckey: np.ndarray, action, logp, value, vwin, vfloor) -> np.ndarray:
        B = len(envs)
        r0, r1 = self.size, self.size + B
        assert r1 <= self.n
        T, A = obs["token_cat"].shape[1], obs["action_cat"].shape[1]
        sl = slice(r0, r1)
        # Only the batch-padded region is written: anything past a row's own ntok/nact is masked by
        # batch() using the stored counts, so stale data there is never read. Features are already
        # bounded to +-64 by the C# encoder (ObsBuilder.Sanitize), so the float16 cast is safe.
        self.tc[sl, :T] = obs["token_cat"]
        self.tn[sl, :T] = obs["token_num"]
        self.ac[sl, :A] = obs["action_cat"]
        self.ar[sl, :A] = obs["action_ref"]
        self.an[sl, :A] = obs["action_num"]
        self.ntok[sl], self.nact[sl] = heads[:, H.H_NTOK], heads[:, H.H_NACT]
        self.boss[sl] = curriculum.final_boss(heads)
        self.env[sl], self.ckey[sl], self.dtype[sl] = envs, ckey, heads[:, H.H_DECISION]
        self.action[sl], self.logp[sl], self.value[sl] = action, logp, value
        floor = heads[:, H.H_TOTALFLOOR] / 50.0
        self.floor_now[sl], self.vwin[sl], self.vfloor_rem[sl] = floor, vwin, vfloor - floor
        self.reward[sl] = self.r_wonind[sl] = self.r_floorinc[sl] = 0
        self.done[sl] = self.trunc[sl] = False
        self.next_value[sl] = self.next_vwin[sl] = self.next_vfloor_rem[sl] = self.t_hp[sl] = np.nan
        self.size = r1
        return np.arange(r0, r1)


class DeviceRollout:
    """The used part of a Rollout copied to the learner GPU once per update, so minibatches are
    gathered on the device instead of fancy-indexed on the host and copied every time."""

    def __init__(self, b: Rollout, n: int, device, extra: dict):
        T = int(b.ntok[:n].max())
        A = int(b.nact[:n].max())
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.tc, self.tn = g(b.tc[:n, :T]), g(b.tn[:n, :T])
        self.ac, self.ar, self.an = g(b.ac[:n, :A]), g(b.ar[:n, :A]), g(b.an[:n, :A])
        self.ntok, self.nact = g(b.ntok[:n]).long(), g(b.nact[:n]).long()
        self.ntok_np, self.nact_np = b.ntok[:n], b.nact[:n]
        self.v = {k: g(v[:n]) for k, v in extra.items()}
        self.device = device

    def batch(self, idx: np.ndarray) -> dict:
        i = torch.from_numpy(idx).to(self.device)
        T = int(self.ntok_np[idx].max())
        A = int(self.nact_np[idx].max())
        tok_ok = torch.arange(T, device=self.device)[None, :] < self.ntok[i][:, None]
        act_ok = torch.arange(A, device=self.device)[None, :] < self.nact[i][:, None]
        # rows shorter than the batch maximum: mark their tail as padding (stale buffer data beyond it)
        return dict(
            token_cat=self.tc[i, :T].long() * tok_ok[..., None],
            token_num=self.tn[i, :T].float() * tok_ok[..., None],
            action_cat=self.ac[i, :A].long() * act_ok[..., None],
            action_ref=torch.where(act_ok[..., None], self.ar[i, :A].long(), -1),
            action_num=self.an[i, :A].float() * act_ok[..., None],
        ), i


CMD_UPDATE, CMD_QUIT = 1, 2


class Learner:
    """PPO update on one GPU; with world > 1 one Learner per GPU (rank 0 in the main process,
    the others in helper processes) and gradients all-reduced over NCCL."""

    def __init__(self, cfg: PPOConfig, mcfg: ModelConfig, dev, rank: int = 0, world: int = 1):
        self.cfg, self.mcfg, self.dev, self.rank, self.world = cfg, mcfg, dev, rank, world
        self.model = Sts2Net(mcfg).to(dev)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99), eps=1e-6)
        self.train_fwd = torch.compile(self.model, dynamic=True) if cfg.compile else self.model
        self.updates = 0
        self.params = [p for p in self.model.parameters()]
        self.ema = None
        if world > 1:
            # gradients live in one flat buffer (+1 slot for the "skip this step" flag): one all-reduce
            total = sum(p.numel() for p in self.params)
            self.flat = torch.zeros(total + 1, device=dev)
            o = 0
            for p in self.params:
                p.grad = self.flat[o:o + p.numel()].view_as(p)
                o += p.numel()
            self.cmd_group = dist.new_group(backend="gloo")  # CPU channel for commands

    def _update_ema(self):
        """Exponential moving average of the weights (rank 0 only). PPO's policy quality oscillates from
        checkpoint to checkpoint; the averaged weights are the steadier model to deploy."""
        d = self.cfg.ema_decay
        if d <= 0 or self.rank != 0:
            return
        with torch.no_grad():
            if self.ema is None:
                self.ema = [p.detach().clone() for p in self.params]
            else:
                torch._foreach_mul_(self.ema, d)
                torch._foreach_add_(self.ema, [p.detach() for p in self.params], alpha=1 - d)

    def ema_state_dict(self):
        if self.ema is None:
            return None
        names = [n for n, _ in self.model.named_parameters()]
        sd = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        for n, e in zip(names, self.ema):
            sd[n] = e.detach().clone()
        return sd

    def _zero_grad(self):
        if self.world > 1:
            self.flat.zero_()
        else:
            self.opt.zero_grad(set_to_none=True)

    def _reduce_grads(self, bad: bool) -> bool:
        """Sum gradients over ranks; returns True if any rank hit a non-finite loss."""
        if self.world == 1:
            if bad:
                self.opt.zero_grad(set_to_none=True)
            return bad
        self.flat[-1] = float(bad)
        dist.all_reduce(self.flat)
        return bool(self.flat[-1].item() > 0)

    def _sync_state(self, full: bool):
        """Make all ranks bit-identical to rank 0: parameters, and (full) optimizer state."""
        if self.world == 1:
            return
        with torch.no_grad():
            for p in self.params:
                dist.broadcast(p.data, 0)
            if full:
                for p in self.params:
                    st = self.opt.state[p]
                    if "exp_avg" not in st:  # create Adam state so it can be overwritten from rank 0
                        st["step"] = torch.zeros((), dtype=torch.float32)
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    step = st["step"].to(self.dev, torch.float32).reshape(1)
                    dist.broadcast(step, 0)
                    st["step"] = step.reshape(()).to(st["step"].device)
                    dist.broadcast(st["exp_avg"], 0)
                    dist.broadcast(st["exp_avg_sq"], 0)

    def command(self, cmd: int, buf_idx: int = 0, n: int = 0, seed: int = 0):
        """Rank 0: tell the helper ranks what to do next."""
        dist.broadcast(torch.tensor([cmd, buf_idx, n, seed], dtype=torch.int64), 0, group=self.cmd_group)

    def wait_command(self):
        t = torch.zeros(4, dtype=torch.int64)
        dist.broadcast(t, 0, group=self.cmd_group)
        return [int(x) for x in t]

    def update(self, buf: Rollout, n: int, seed: int):
        """PPO epochs over rollout rows [0, n). All ranks use the same permutation (seed); each rank
        trains on its stride of every minibatch and the summed gradients are all-reduced."""
        cfg = self.cfg
        self._sync_state(full=self.updates == 0 or self.updates % 50 == 0)
        self.updates += 1
        dbuf = DeviceRollout(buf, n, self.dev, dict(
            action=buf.action, logp=buf.logp, value=buf.value, adv=buf.adv, ret=buf.ret, t_win=buf.t_win,
            t_floor=buf.t_floor, t_hp=buf.t_hp, strategic=buf.strategic, dtype=buf.dtype, boss=buf.boss))
        self.model.train()
        logs = defaultdict(list)
        rng = np.random.default_rng(seed)
        for _ in range(cfg.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                mb = mb[np.argsort(buf.ntok[mb], kind="stable")]  # less padding
                mine = mb[self.rank::self.world]                  # this rank's share
                self._zero_grad()
                bad = False
                acc = defaultdict(float)
                # micro-batches under a token budget (memory grows with deck size), gradients accumulated
                for idx in self._split_tokens(buf, mine):
                    w = len(idx) / len(mb)
                    bad |= not self._loss_backward(dbuf, idx, w, acc)
                share = len(mine) / max(1, len(mb))  # weights w sum to this rank's share of the minibatch
                for k, v in acc.items():
                    logs[k].append(v / share)
                bad = self._reduce_grads(bad)
                if bad:
                    if cfg.fail_nonfinite:
                        raise FloatingPointError("nonfinite minibatch; stopping the experiment")
                    logs["skipped_nonfinite"].append(torch.ones(()))
                    continue
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(gn):
                    if cfg.fail_nonfinite:
                        raise FloatingPointError("nonfinite gradient; stopping the experiment")
                    logs["skipped_nonfinite"].append(torch.ones(()))
                    continue
                self.opt.step()
                self._update_ema()
                logs["grad_norm"].append(gn.detach())
        # one device sync for all logged scalars
        out = {k: float(torch.stack([torch.as_tensor(x).float().cpu() for x in v]).mean()) for k, v in logs.items()}
        # entropy by decision kind: sums and counts per minibatch -> pooled means
        for k in ENTROPY_SPLITS:
            total, count = out.pop(f"_ent_{k}_sum", 0.0), out.pop(f"_ent_{k}_n", 0.0)
            if count > 0:
                out[f"entropy_{k}"] = total / count
        return out

    def _split_tokens(self, buf: Rollout, idx: np.ndarray):
        """Split rows into chunks whose padded size rows * max(tokens + actions) stays under the token
        budget (activation memory grows with deck size; the budget scales inversely with model width)."""
        budget = self.cfg.token_budget or (32 << 20) // self.mcfg.d_model
        cost = (buf.ntok[idx] + buf.nact[idx] + self.mcfg.n_memory).tolist()
        if not cost:
            return
        start, mx = 0, 0
        for i, c in enumerate(cost):
            mx2 = max(mx, c)
            if i > start and (i + 1 - start) * mx2 > budget:
                yield idx[start:i]
                start, mx2 = i, c
            mx = mx2
        yield idx[start:]

    def _loss_backward(self, dbuf: DeviceRollout, idx, w, acc) -> bool:
        """Forward + backward on one micro-batch, loss scaled by its share w of the minibatch;
        adds w * each logged scalar to acc (so acc ends up holding minibatch means)."""
        cfg = self.cfg
        b, i = dbuf.batch(idx)
        t = lambda k: dbuf.v[k][i]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            out = self.train_fwd(**b)
        logits = out["logits"].float()
        logp_all = torch.log_softmax(logits, -1)
        act = t("action")
        logp = logp_all.gather(1, act[:, None]).squeeze(1)
        p = logp_all.exp()
        ent = -(p * torch.where(torch.isfinite(logp_all), logp_all, torch.zeros_like(logp_all))).sum(-1)
        ratio = (logp - t("logp")).exp()
        A = t("adv")
        strat = t("strategic")
        pg_i = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A)
        pg = (pg_i * torch.where(strat, cfg.strat_pg_coef, 1.0)).mean()
        v = out["v_return"].float()
        v_old = t("value")
        R = t("ret")
        v_clipped = v_old + (v - v_old).clamp(-cfg.vclip, cfg.vclip)
        vloss = 0.5 * torch.max((v - R) ** 2, (v_clipped - R) ** 2).mean()
        ent_coef = torch.where(strat, cfg.ent_strategic, cfg.ent_combat)
        ent_loss = -(ent_coef * ent).mean()
        l_win = F.binary_cross_entropy_with_logits(out["v_win"].float(), t("t_win"))
        l_floor = F.mse_loss(out["v_floor"].float(), t("t_floor"))
        hp_t = t("t_hp")
        m = (~torch.isnan(hp_t)).float()  # labeled steps only; no host sync
        l_hp = ((out["v_hp"].float() - torch.nan_to_num(hp_t)) ** 2 * m).sum() / m.sum().clamp(min=1)
        aux = l_win + l_floor + l_hp
        loss = pg + cfg.vf_coef * vloss + ent_loss + cfg.aux_coef * aux
        if not torch.isfinite(loss):
            return False
        (loss * w).backward()
        with torch.no_grad():
            for k, val in (("pg", pg), ("vloss", vloss), ("entropy", ent.mean()), ("aux_v_win", l_win),
                           ("aux_v_floor", l_floor), ("aux_v_hp", l_hp),
                           ("clipfrac", ((ratio - 1).abs() > cfg.clip).float().mean()),
                           ("approx_kl", ((ratio - 1) - (logp - t("logp"))).mean())):
                acc[k] = acc[k] + w * val.detach().float()
            ent_d, fb = ent.detach().float(), t("boss")
            for k, m in (("combat", t("dtype") == spec.DecisionType.COMBAT), ("strategic", strat),
                         ("boss1", fb == 0), ("boss2", fb == 1)):
                acc[f"_ent_{k}_sum"] = acc[f"_ent_{k}_sum"] + (ent_d * m).sum()
                acc[f"_ent_{k}_n"] = acc[f"_ent_{k}_n"] + m.sum().float()
        return True


ENTROPY_SPLITS = ("combat", "strategic", "boss1", "boss2")  # logged as entropy_<kind> (policy during the update)


class Trainer:
    def __init__(self, cfg: PPOConfig, wcfg: WorkerConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        os.makedirs(cfg.out, exist_ok=True)
        vocab = json.load(open(os.path.join(wcfg.gamebin, "vocab.json")))
        mcfg = PRESETS[cfg.preset]
        mcfg = ModelConfig(**{**asdict(mcfg), "vocab_size": int(vocab["size"])})
        if mcfg.moe_layers and not mcfg.expert_ids:
            mcfg = ModelConfig(**{**asdict(mcfg), "expert_ids": expert_ids_from_vocab(vocab["ids"])})
        self.mcfg = mcfg
        self.overlap = bool(cfg.actor_device) and torch.device(cfg.actor_device) != self.dev
        self.actor_dev = torch.device(cfg.actor_device) if self.overlap else self.dev
        N = cfg.n_envs
        rows = cfg.steps_per_iter + 2 * N
        nbuf = 2 if self.overlap else 1
        ldevs = [d for d in cfg.learner_devices.split(",") if d] or [cfg.device]
        assert torch.device(ldevs[0]) == self.dev, "the first learner device must be --device"
        self.world = len(ldevs)
        self.helpers = []
        shm_names = [f"sts2ai_{os.getpid()}_rollout{k}" for k in range(nbuf)] if self.world > 1 else [None] * nbuf
        # two buffers when overlapping: the actor fills one while the learner trains on the other
        self.bufs = [Rollout(rows, name) for name in shm_names]
        self.cbuf = self.bufs[0]  # buffer being collected into
        if self.world > 1:
            port = _free_port()
            ctx = torch.multiprocessing.get_context("spawn")
            for r in range(1, self.world):
                p = ctx.Process(target=_learner_helper, name=f"learner{r}", daemon=True,
                                args=(r, self.world, port, asdict(cfg), asdict(mcfg), ldevs[r], shm_names, rows))
                p.start()
                self.helpers.append(p)
            torch.cuda.set_device(self.dev)
            dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=self.world)
        self.learner = Learner(cfg, mcfg, self.dev, 0, self.world)
        self.model, self.opt = self.learner.model, self.learner.opt
        self.iter = 0
        self.total_steps = 0
        self.trained_steps = 0
        self.train_rows_last = 0
        self.initial_model_sha256 = None
        self.prefix_actions_observed = 0
        self.restart_requested_episodes = self.restart_actual_episodes = self.completed_episodes = 0
        self.engine_errors_total = 0
        self.update_rng = np.random.default_rng(cfg.seed + 19)
        # ascension curriculum state (before load: a checkpoint restores the level)
        self.asc_level = cfg.ascension
        self.asc_rng = np.random.default_rng(cfg.seed + 7)
        self.asc_since = -1  # total_steps when the current level started (-1: set at the first check)
        # failure map (before load: a checkpoint restores it); 12 situations, 13 with a3_boss2
        self.n_sit = curriculum.n_situations(cfg.boss2_situation)
        self.bosses = curriculum.FinalBossTracker(cfg.n_envs)  # final-boss entry HP and fight outcomes per env
        self.fmap_loss = np.zeros(self.n_sit)    # decayed lost fresh runs credited per situation
        self.fmap_w = situation_weights(self.fmap_loss, cfg.focus_sampler, cfg.focus_floor)
        self.fmap_rng = np.random.default_rng(cfg.seed + 11)
        self._fmap_clear()
        if cfg.init:
            self.load(cfg.init)
        self.start_steps = self.total_steps
        if self.initial_model_sha256 is None:
            digest = hashlib.sha256()
            for name, tensor in sorted(self.model.state_dict().items()):
                digest.update(name.encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            self.initial_model_sha256 = digest.hexdigest()
        if cfg.expected_initial_sha256 and cfg.expected_initial_sha256 != self.initial_model_sha256:
            raise ValueError("initial model hash does not match the preregistered paired initialization")
        print(json.dumps({"initial_model_sha256": self.initial_model_sha256, "seed": cfg.seed,
                          "init_mode": cfg.init_mode if cfg.init else "random",
                          "fixed_update_rows": cfg.fixed_update_rows, "train_seed_bits": cfg.train_seed_bits}), flush=True)
        if self.overlap:
            self.actor = Sts2Net(mcfg).to(self.actor_dev)
            self.actor.load_state_dict(self.model.state_dict())
            self.actor.eval()
        else:
            self.actor = self.model
        print(f"model {cfg.preset}: {count_params(self.model) / 1e6:.2f}M params, vocab {mcfg.vocab_size}, "
              f"learner {','.join(ldevs)}, actor {self.actor_dev}{' (overlapped)' if self.overlap else ''}", flush=True)
        self.batch_min = cfg.batch_min or max(1, N // 4)
        reserved = json.load(open(cfg.reserved_seeds_file)) if cfg.reserved_seeds_file else []
        self.pool = EnvPool(N, wcfg, seed=cfg.env_seed if cfg.env_seed >= 0 else cfg.seed,
                            seed_upper_bound=cfg.train_seed_limit or seed_upper_bound(cfg.train_seed_bits),
                            reserved_seeds=reserved)
        self.prefix_seen = np.zeros(N, dtype=np.int64)
        # per-env state
        self.pend_row = np.full(N, -1, np.int64)            # rollout row awaiting its outcome
        self.prev_head = np.zeros((N, shm.HEADER_INTS), np.int32)
        self.episode = np.zeros(N, np.int64)                # episodes started per env (for combat keys)
        self.ev_keys, self.ev_hp = [], []                    # combats that ended this iteration
        self.carry = []                                      # (envs, heads, obs) received but not yet acted on
        self.stopping = False                                # set by close(): the collector thread exits
        self.collector = None
        self.stats = defaultdict(lambda: deque(maxlen=1000))
        self.deaths = Counter()                              # (room type, encounter/event id) of lost runs
        self.vocab_names = vocab["ids"]
        self.timing = defaultdict(float)
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(cfg.out)
        if cfg.save_initial and not cfg.init:
            self.save(os.path.join(cfg.out, "initial.pt"))

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: str):
        tmp = path + ".tmp"
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "mcfg": asdict(self.mcfg),
                    "cfg": asdict(self.cfg), "iter": self.iter, "total_steps": self.total_steps, "asc_level": self.asc_level,
                    "trained_steps": self.trained_steps, "initial_model_sha256": self.initial_model_sha256,
                    "prefix_actions_observed": self.prefix_actions_observed,
                    "restart_requested_episodes": self.restart_requested_episodes,
                    "restart_actual_episodes": self.restart_actual_episodes, "completed_episodes": self.completed_episodes,
                    "engine_errors_total": self.engine_errors_total,
                    "fmap_loss": self.fmap_loss.tolist(), "fmap_w": self.fmap_w.tolist(),
                    "ema": self.learner.ema_state_dict(),
                    "update_rng": self.update_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "fmap_rng": self.fmap_rng.bit_generator.state, "asc_rng": self.asc_rng.bit_generator.state,
                    "fmap_round": self.fmap_round.tolist(), "fmap_tries": self.fmap_tries.tolist(),
                    "fmap_wins": self.fmap_wins.tolist(), "fmap_dec": self.fmap_dec.tolist(), "n_sit": self.n_sit,
                    "resume_contract": "optimizer/model continuation; worker pools and in-flight rollouts reset"}, tmp)
        os.replace(tmp, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.dev)
        if self.cfg.init_mode == "weights":
            key = self.cfg.init_weights
            if ck.get(key) is None:
                raise ValueError(f"checkpoint has no {key} weights")
            self.model.load_state_dict(ck[key], strict=True)
            # EMA must start at the imported weights, not the random initial network.
            self.learner.ema = [p.detach().clone() for p in self.model.parameters()]
            print(json.dumps({"weights_only_init": path, "weights": key,
                              "optimizer_reset": True, "curriculum_reset": True}), flush=True)
            return
        missing, unexpected = self.model.load_state_dict(ck["model"], strict=bool(self.cfg.expected_initial_sha256))
        if missing or unexpected:
            print("load: missing", missing, "unexpected", unexpected)
        try:
            self.opt.load_state_dict(ck["opt"])
        except Exception as e:
            if self.cfg.expected_initial_sha256:
                raise RuntimeError("frozen experiment requires exact optimizer-state resume") from e
            print("optimizer state not loaded:", e)
        # load_state_dict restores the checkpoint's lr / weight decay; use this run's (--lr applies on resume, and
        # learner ranks, which build their optimizer from the config, step with the same lr as rank 0)
        for g in self.opt.param_groups:
            g["lr"], g["weight_decay"] = self.cfg.lr, self.cfg.weight_decay
        self.iter = ck.get("iter", 0)
        self.total_steps = ck.get("total_steps", 0)
        self.trained_steps = ck.get("trained_steps", 0)
        self.initial_model_sha256 = ck.get("initial_model_sha256")
        for key in ("prefix_actions_observed", "restart_requested_episodes", "restart_actual_episodes", "completed_episodes", "engine_errors_total"):
            setattr(self, key, ck.get(key, 0))
        if ck.get("torch_rng") is not None:
            torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        for key, generator in (("fmap_rng", self.fmap_rng), ("asc_rng", self.asc_rng)):
            if ck.get(key) is not None:
                generator.bit_generator.state = ck[key]
        if ck.get("update_rng") is not None:
            self.update_rng.bit_generator.state = ck["update_rng"]
        if self.cfg.asc_max > 0 and "asc_level" in ck:
            self.asc_level = max(self.asc_level, int(ck["asc_level"]))
        self._restore_failmap(ck)
        if ck.get("ema") is not None:
            names = [n for n, _ in self.model.named_parameters()]
            self.learner.ema = [ck["ema"][n].to(self.dev).clone() for n in names]

    def _restore_failmap(self, ck: dict):
        """Failure-map state from a checkpoint. Checkpoints written with 12 situations load into 13 (a3_boss2
        starts empty) and back (a3_boss2 folded into a3_boss), with a log line; see curriculum.resize_situations."""
        keys = [k for k in ("fmap_loss", "fmap_w", "fmap_round", "fmap_tries", "fmap_wins", "fmap_dec") if k in ck]
        sizes = sorted({len(np.asarray(ck[k]).reshape(-1)) for k in keys})
        if sizes and sizes != [self.n_sit]:
            print(json.dumps({"failmap_resized": {
                "from": sizes, "to": self.n_sit, "keys": keys,
                "how": ("a3_boss2 starts at 0; the old a3_boss counts (both final bosses) stay in a3_boss; "
                        "focus weights recomputed" if self.n_sit > shm.N_SIT else "a3_boss2 folded into a3_boss")}}), flush=True)
        if self.cfg.focus and self.cfg.focus_sampler == "failure" and "fmap_w" in ck:
            self.fmap_loss = curriculum.resize_situations(ck["fmap_loss"], self.n_sit)
            w = np.asarray(ck["fmap_w"], np.float64)
            self.fmap_w = w if len(w) == self.n_sit else situation_weights(self.fmap_loss, self.cfg.focus_sampler,
                                                                          self.cfg.focus_floor)
        for key in ("fmap_round", "fmap_tries", "fmap_wins", "fmap_dec"):
            if key in ck:
                setattr(self, key, curriculum.resize_situations(ck[key], self.n_sit))

    # ------------------------------------------------------------------ rollout
    def _ckey(self, envs: np.ndarray, heads: np.ndarray) -> np.ndarray:
        """Unique id of the combat each step belongs to (-1 outside combat)."""
        key = ((envs.astype(np.int64) << 20) + self.episode[envs]) << 12 | (heads[:, H.H_COMBATID] & 0xFFF)
        return np.where(heads[:, H.H_INCOMBAT] != 0, key, -1)

    def _close_pending(self, ids: np.ndarray, heads: np.ndarray) -> np.ndarray:
        """Use fresh headers to finish the pending step of these envs. Returns their rollout rows (-1 if none)."""
        rows_all = self.pend_row[ids].copy()
        has = rows_all >= 0
        if has.any():
            e, h, rows = ids[has], heads[has], rows_all[has]
            p = self.prev_head[e]
            b = self.cbuf
            ended = (h[:, H.H_DONE] != 0) | (h[:, H.H_ERROR] != 0)
            err = h[:, H.H_ERROR] != 0
            b.reward[rows] = step_reward(p, h, self.cfg)
            b.done[rows] = ended
            b.trunc[rows] = err
            b.r_wonind[rows] = ((h[:, H.H_DONE] != 0) & (h[:, H.H_WON] != 0) & ~err).astype(np.float32)
            b.r_floorinc[rows] = np.maximum(0, h[:, H.H_TOTALFLOOR] - p[:, H.H_TOTALFLOOR]) / 50.0
            # combat finished -> end-of-combat hp fraction labels that combat's steps
            fin, hpf = curriculum.combat_end_hp(p, h, bool(self.cfg.hp_label_fix))
            if fin.any():
                self.ev_keys.append(self._ckey(e[fin], p[fin]))
                self.ev_hp.append(hpf[fin].astype(np.float32))
            self.pend_row[e] = -1
        return rows_all

    def _finish_episodes(self, ids: np.ndarray, heads: np.ndarray):
        # practice episodes (restarted mid-run) are logged apart so the run statistics stay comparable
        availability = heads[:, H.H_AVAILABLE_RESTARTS].copy()
        self.engine_errors_total += int(np.count_nonzero(heads[:, H.H_ERROR]))
        restart = heads[:, H.H_RESTART]
        self.completed_episodes += len(ids)
        self.restart_requested_episodes += int(np.count_nonzero(heads[:, H.H_REQUESTED_RESTART]))
        self.restart_actual_episodes += int(np.count_nonzero(restart))
        # practice outcome: the run won, or (--fight-outcome) the final-boss fight for a3_boss / a3_boss2 points
        success = self.bosses.finish(ids, heads, bool(self.cfg.fight_outcome))
        if self.cfg.focus:
            m = (restart > 0) & (restart <= self.n_sit) & (heads[:, H.H_ERROR] == 0)
            np.add.at(self.fmap_tries, restart[m] - 1, 1)
            np.add.at(self.fmap_wins, restart[m] - 1, success[m].astype(float))
        for k in np.unique(restart[restart > 0]):
            m = restart == k
            self.stats[f"restart_err_k{k}"].extend([1.0] * int((m & (heads[:, H.H_ERROR] != 0)).sum()))
            okk = heads[m & (heads[:, H.H_ERROR] == 0)]
            self.stats[f"restart_win_k{k}"].extend((okk[:, H.H_WON] != 0).astype(float).tolist())
        heads = heads[restart == 0]
        # co-op episodes (team runs) likewise
        coop = heads[:, H.H_COOP]
        for k in np.unique(coop[coop > 0]):
            okk = heads[(coop == k) & (heads[:, H.H_ERROR] == 0)]
            self.stats[f"coop_win_n{k}"].extend((okk[:, H.H_WON] != 0).astype(float).tolist())
            self.stats[f"coop_err_n{k}"].extend([1.0] * int(((coop == k) & (heads[:, H.H_ERROR] != 0)).sum()))
        heads = heads[coop == 0]
        err = heads[:, H.H_ERROR] != 0
        ok = heads[~err]
        self.stats["errors"].extend([1.0] * int(err.sum()))
        self.stats["win"].extend((ok[:, H.H_WON] != 0).astype(float).tolist())
        self.stats["floor"].extend(ok[:, H.H_TOTALFLOOR].astype(float).tolist())
        self.stats["act"].extend((ok[:, H.H_ACT] + 1).astype(float).tolist())
        self.stats["combats_won"].extend(ok[:, H.H_COMBATS_WON].astype(float).tolist())
        for c in np.unique(ok[:, H.H_CHARACTER]):
            m = ok[:, H.H_CHARACTER] == c
            self.stats[f"win_c{c}"].extend((ok[m, H.H_WON] != 0).astype(float).tolist())
            self.stats[f"floor_c{c}"].extend(ok[m, H.H_TOTALFLOOR].astype(float).tolist())
        for a in np.unique(ok[:, H.H_ASCENSION]):
            self.stats[f"asc_win_{a}"].extend((ok[ok[:, H.H_ASCENSION] == a, H.H_WON] != 0).astype(float).tolist())
        self._maybe_promote()
        lost = ok[ok[:, H.H_WON] == 0]
        for rt, rm in zip(lost[:, H.H_ROOMTYPE].tolist(), lost[:, H.H_ROOMMODEL].tolist()):
            self.deaths[(rt, rm)] += 1
        if self.cfg.focus and len(lost):
            # failure map: a run lost in a fight is credited to that fight and (focus_strategic) to its act start;
            # one lost outside a fight to the act start
            # (with a3_boss2, a death in the second final boss is credited to a3_boss2)
            act, rt = np.clip(lost[:, H.H_ACT], 0, 2), lost[:, H.H_ROOMTYPE]
            fight = np.isin(rt, (1, 2, 3))
            sit = curriculum.situation_index(act, rt, lost[:, H.H_BOSSKILLS], self.n_sit)
            np.add.at(self.fmap_round, act * 4, np.where(fight, self.cfg.focus_strategic, 1.0))
            np.add.at(self.fmap_round, sit[fight], 1 - self.cfg.focus_strategic)
        self.episode[ids] += 1
        n = len(ids)
        asc = self._sample_ascension(n) if self.cfg.asc_max > 0 else None
        if self.cfg.focus:
            # practice episodes start in situations drawn from the failure map (-1: fresh run)
            if self.cfg.mask_unavailable_restarts:
                sit = available_situations(self.fmap_rng, self.fmap_w, availability, self.cfg.restart_prob)
            else:
                sit = self.fmap_rng.choice(self.n_sit, size=n, p=self.fmap_w)
                sit = np.where(self.fmap_rng.random(n) < self.cfg.restart_prob, sit, -1)
            self.pool.reset(ids, ascension=asc if asc is not None else np.full(n, self.cfg.ascension), situation=sit)
        elif asc is not None:
            self.pool.reset(ids, ascension=asc)
        else:
            self.pool.reset(ids)

    def _fmap_clear(self):
        self.fmap_round = np.zeros(self.n_sit)   # this round's lost fresh runs, credited per situation
        self.fmap_tries = np.zeros(self.n_sit)   # practice episodes finished per situation
        self.fmap_wins = np.zeros(self.n_sit)    # ... successful (run won, or the fight with --fight-outcome)
        self.fmap_dec = np.zeros(self.n_sit)     # decisions collected per situation (all episodes)

    def _update_failmap(self):
        """One round of the failure-map loop: where fresh runs were lost -> where practice starts next round."""
        c = self.cfg
        lost = self.fmap_round.sum()
        if lost > 0:
            self.fmap_loss = c.focus_decay * self.fmap_loss + self.fmap_round
            self.fmap_w = situation_weights(self.fmap_loss, c.focus_sampler, c.focus_floor)
        loss = self.fmap_round / max(lost, 1e-9)
        dec = self.fmap_dec / max(self.fmap_dec.sum(), 1)
        print(json.dumps({"failmap": {
            shm.RESTART_KINDS[s + 1]: {"loss%": round(100 * loss[s], 1), "dec%": round(100 * dec[s], 1),
                                       "focus%": round(100 * self.fmap_w[s], 1), "practice_n": int(self.fmap_tries[s]),
                                       "practice_win%": round(100 * self.fmap_wins[s] / max(self.fmap_tries[s], 1), 1)}
            for s in range(self.n_sit)}, "lost_runs": int(round(lost)), "steps": self.total_steps,
            "practice_outcome": "fight" if c.fight_outcome else "run",
            "final_bosses": self.bosses.summary()}), flush=True)
        self.bosses.clear_round()
        self._fmap_clear()

    def _sample_ascension(self, n: int) -> np.ndarray:
        """Curriculum: asc_a0 of episodes at A0; the rest at the current level (60%) or a lower one (review)."""
        L, r = self.asc_level, self.asc_rng
        u = r.random(n)
        lower = r.integers(0, max(L, 1), n)
        return np.where(u < self.cfg.asc_a0, 0, np.where(u < self.cfg.asc_a0 + 0.6 * (1 - self.cfg.asc_a0), L, lower)).astype(np.int32)

    def _maybe_promote(self):
        cfg = self.cfg
        if cfg.asc_max <= 0 or self.asc_level >= cfg.asc_max:
            return
        if self.asc_since < 0:
            self.asc_since = self.total_steps
        if self.total_steps - self.asc_since < cfg.asc_dwell:
            return
        d = self.stats[f"asc_win_{self.asc_level}"]
        if len(d) >= cfg.asc_window and float(np.mean(list(d)[-cfg.asc_window:])) >= cfg.asc_promote:
            self.asc_level += 1
            self.asc_since = self.total_steps
            print(json.dumps({"ascension_promoted": self.asc_level, "steps": self.total_steps}), flush=True)

    def _receive(self, timeout_s: float = 120):
        """Wait for a batch of observations. Returns (envs needing an action, heads, obs, rows closed);
        envs is empty if nothing arrived within a short timeout_s (< 120)."""
        if self.stopping:
            raise SystemExit("trainer closing")
        t0 = time.perf_counter()
        ids = self.pool.wait(min_ready=self.batch_min, timeout_s=timeout_s, batch_wait_us=self.cfg.batch_wait_us)
        t1 = time.perf_counter()
        self.timing["wait"] += t1 - t0
        if not len(ids):
            if timeout_s < 120:
                return ids, None, None, ids
            raise RuntimeError("no worker responded for 120s")
        heads = self.pool.heads(ids)
        prefix = heads[:, H.H_PREFIX_ACTIONS].astype(np.int64)
        self.prefix_actions_observed += int(np.maximum(0, prefix - self.prefix_seen[ids]).sum())
        self.prefix_seen[ids] = prefix
        rows = self._close_pending(ids, heads)
        fin = (heads[:, H.H_DONE] != 0) | (heads[:, H.H_ERROR] != 0) | (heads[:, H.H_NACT] == 0)
        if fin.any():
            self._finish_episodes(ids[fin], heads[fin])
        live = ~fin
        envs, heads, rows = ids[live], heads[live], rows[live]
        if len(envs):
            self.bosses.observe(envs, heads)
        if self.cfg.focus and len(envs):
            rt = heads[:, H.H_ROOMTYPE]
            kind = np.where((heads[:, H.H_INCOMBAT] != 0) & np.isin(rt, (1, 2, 3)), rt, 0)
            sit = curriculum.situation_index(heads[:, H.H_ACT], kind, heads[:, H.H_BOSSKILLS], self.n_sit)
            self.fmap_dec += np.bincount(sit, minlength=self.n_sit)
        obs = self.pool.gather(envs, heads) if len(envs) else None
        self.timing["host"] += time.perf_counter() - t1
        return envs, heads, obs, rows

    def _forward(self, obs: dict):
        t0 = time.perf_counter()
        tens = {k: torch.from_numpy(v).to(self.actor_dev, non_blocking=True) for k, v in obs.items()}
        for k in ("token_cat", "action_cat", "action_ref"):
            tens[k] = tens[k].long()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.actor_dev.type == "cuda"):
            out = self.actor(**tens)
        self.timing["gpu"] += time.perf_counter() - t0
        return out

    def _launch(self, envs: np.ndarray, heads: np.ndarray, obs: dict):
        """Start the policy on a batch (GPU work is queued, results copied back asynchronously)."""
        out = self._forward(obs)
        t0 = time.perf_counter()
        logits = out["logits"].float()
        logp_all = torch.log_softmax(logits, -1)
        eps = self.cfg.explore_strategic
        if eps > 0:
            # behaviour policy mu = (1-eps) pi + eps uniform on strategic decisions: keeps rare strategic
            # options (card removal, skipping, elites) sampled so their consequences can be learned
            legal = torch.isfinite(logits)
            n_legal = legal.sum(-1, keepdim=True).clamp(min=1)
            strat = torch.from_numpy(np.isin(heads[:, H.H_DECISION], list(spec.DecisionType.STRATEGIC))).to(logits.device)
            e = torch.where(strat, eps, 0.0)[:, None]
            mix = (1 - e) * logp_all.exp() + e * legal.float() / n_legal
            logp_all = torch.log(mix.clamp_min(1e-30)).masked_fill(~legal, float("-inf"))
        # Gumbel-max sampling == Categorical(logits).sample()
        u = torch.rand_like(logits).clamp_(1e-10, 1.0)
        a = (logp_all - torch.log(-torch.log(u))).argmax(-1)
        logp = logp_all.gather(1, a[:, None]).squeeze(1)
        res = torch.stack([a.float(), logp, out["v_return"].float(), torch.sigmoid(out["v_win"].float()),
                           out["v_floor"].float()], 1)
        if self.actor_dev.type == "cuda":
            host = torch.empty(res.shape, dtype=res.dtype, pin_memory=True)
            host.copy_(res, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
        else:
            host, done = res, None
        self.timing["gpu"] += time.perf_counter() - t0
        return envs, heads, obs, host, done

    def _finish(self, launched) -> int:
        """Wait for a launched batch, store its steps and send the actions."""
        envs, heads, obs, host, done = launched
        t0 = time.perf_counter()
        if done is not None:
            done.synchronize()
        res = host.numpy()
        self.timing["gpu"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        act = res[:, 0].astype(np.int32)
        rows = self.cbuf.add(obs, heads, envs, self._ckey(envs, heads), act, res[:, 1], res[:, 2], res[:, 3], res[:, 4])
        self.pend_row[envs] = rows
        self.prev_head[envs] = heads
        self.pool.step(envs, act)
        self.timing["host"] += time.perf_counter() - t1
        return len(envs)

    def _act(self, envs: np.ndarray, heads: np.ndarray, obs: dict) -> int:
        return self._finish(self._launch(envs, heads, obs))

    @torch.no_grad()
    def collect(self, buf: Rollout):
        """Fill `buf` with steps_per_iter steps using the actor network. Returns (decisions/s, mean batch)."""
        cfg = self.cfg
        self.cbuf = buf
        buf.reset()
        self.ev_keys, self.ev_hp = [], []
        self.timing.clear()
        self.actor.eval()
        t0 = time.time()
        n_dec = 0
        batches = 0
        carry, self.carry = self.carry, []
        for envs, heads, obs in carry:
            n_dec += self._act(envs, heads, obs)
        # pipelined: while the GPU evaluates batch k, the host waits for and gathers batch k+1
        inflight = None
        while buf.size < cfg.steps_per_iter:
            # with a batch in flight, its envs are blocked until it finishes: wait only briefly, so the
            # loop never waits on workers that can only respond after the in-flight batch is sent
            envs, heads, obs, _ = self._receive(timeout_s=0.05 if inflight is not None else 120)
            launched = self._launch(envs, heads, obs) if len(envs) else None
            if inflight is not None:
                n_dec += self._finish(inflight)
            inflight = launched
            batches += launched is not None
        if inflight is not None:
            n_dec += self._finish(inflight)
        self._drain_pending()
        self._label_combat_hp()
        self.total_steps += buf.size
        dt = time.time() - t0
        return n_dec / dt, n_dec / max(1, batches)

    @torch.no_grad()
    def _drain_pending(self):
        """Wait for the observation following every stored step (needed for its reward and bootstrap
        values). Those observations are carried over and acted on next iteration."""
        b = self.cbuf
        while (self.pend_row >= 0).any():
            envs, heads, obs, rows = self._receive()
            if not len(envs):
                continue
            out = self._forward(obs)
            v = torch.stack([out["v_return"].float(), torch.sigmoid(out["v_win"].float()), out["v_floor"].float()], 1).cpu().numpy()
            m = rows >= 0
            r = rows[m]
            b.next_value[r] = v[m, 0]
            b.next_vwin[r] = v[m, 1]
            b.next_vfloor_rem[r] = v[m, 2] - heads[m, H.H_TOTALFLOOR] / 50.0
            self.carry.append((envs, heads, obs))

    def _label_combat_hp(self):
        b, n = self.cbuf, self.cbuf.size
        if not self.ev_keys:
            return
        ev = np.concatenate(self.ev_keys)
        hp = np.concatenate(self.ev_hp)
        order = np.argsort(ev)
        ev, hp = ev[order], hp[order]
        keys = b.ckey[:n]
        pos = np.clip(np.searchsorted(ev, keys), 0, len(ev) - 1)
        hit = (keys >= 0) & (ev[pos] == keys)
        b.t_hp[:n][hit] = hp[pos[hit]]

    # ------------------------------------------------------------------ targets
    def advantages(self, b: Rollout):
        cfg, n = self.cfg, b.size
        N = cfg.n_envs
        s = slice(0, n)
        adv, ret = fastpool.gae(b.reward[s], b.value[s], b.done[s], b.trunc[s], b.next_value[s], b.env[s], N, cfg.gamma, cfg.lam)
        # auxiliary value targets (gamma 1): P(win) and floors still to gain
        _, t_win = fastpool.gae(b.r_wonind[s], b.vwin[s], b.done[s], b.trunc[s], b.next_vwin[s], b.env[s], N, 1.0, cfg.lam_aux)
        _, t_rem = fastpool.gae(b.r_floorinc[s], b.vfloor_rem[s], b.done[s], b.trunc[s], b.next_vfloor_rem[s], b.env[s], N, 1.0, cfg.lam_aux)
        t_win = np.clip(t_win, 0.0, 1.0)
        t_floor = t_rem + b.floor_now[s]
        strategic = np.isin(b.dtype[s], list(spec.DecisionType.STRATEGIC))
        if cfg.lam_strategic >= 0 and cfg.lam_strategic != cfg.lam:
            # strategic choices (card picks, shop, path, ...) pay off hundreds of decisions later: give them
            # a longer-horizon advantage estimate; combat keeps the lower-variance one
            adv_s, _ = fastpool.gae(b.reward[s], b.value[s], b.done[s], b.trunc[s], b.next_value[s], b.env[s], N, cfg.gamma, cfg.lam_strategic)
            adv[strategic] = adv_s[strategic]
        # separate normalization for combat vs strategic decisions
        for m in (strategic, ~strategic):
            if m.sum() > 1:
                adv[m] = (adv[m] - adv[m].mean()) / (adv[m].std() + 1e-8)
        return adv, ret, t_win, t_floor, strategic

    # ------------------------------------------------------------------ update
    def update(self, buf: Rollout):
        """Compute targets into the rollout, then run the (possibly multi-GPU) learner on it."""
        adv, ret, t_win, t_floor, strategic = self.advantages(buf)
        n = buf.size
        buf.adv[:n], buf.ret[:n], buf.t_win[:n], buf.t_floor[:n], buf.strategic[:n] = adv, ret, t_win, t_floor, strategic
        n = update_rows(n, self.cfg.fixed_update_rows)
        seed = int(self.update_rng.integers(2**31))
        if self.learner.world > 1:
            self.learner.command(CMD_UPDATE, self.bufs.index(buf), n, seed)
        logs = self.learner.update(buf, n, seed)
        self.train_rows_last = n
        self.trained_steps += n
        return logs

    # ------------------------------------------------------------------ main loop
    def _sync_actor(self):
        if self.overlap:
            with torch.no_grad():
                for pa, pl in zip(self.actor.parameters(), self.model.parameters()):
                    pa.copy_(pl, non_blocking=True)
            torch.cuda.synchronize(self.actor_dev)

    def run(self):
        """Sequential: collect, update. Overlapped (actor on its own GPU): while the learner trains on
        rollout k, the actor collects rollout k+1 with the weights from before that update; PPO's
        importance ratio uses the behaviour log-probs stored at collection, so the one-update lag is
        accounted for."""
        cfg = self.cfg
        def keep_running():
            return self.iter < cfg.iters and (cfg.max_new_steps <= 0 or
                    self.total_steps - self.start_steps < cfg.max_new_steps)
        if not self.overlap:
            while keep_running():
                t0 = time.time()
                sps, avg_batch = self.collect(self.bufs[0])
                t1 = time.time()
                timing = dict(self.timing)
                logs = self.update(self.bufs[0])
                self._log(self.bufs[0], sps, avg_batch, t1 - t0, time.time() - t1, timing, logs)
            return
        import threading
        sps, avg_batch = self.collect(self.bufs[0])
        k = 0
        while keep_running():
            result = {}
            t0 = time.time()
            def collector(buf=self.bufs[1 - k]):
                t = time.time()
                result["sps"], result["batch"] = self.collect(buf)
                result["t"] = time.time() - t
            th = self.collector = threading.Thread(target=collector, name="collector", daemon=True)
            th.start()
            logs = self.update(self.bufs[k])
            t_upd = time.time() - t0
            th.join()
            if "sps" not in result:
                raise RuntimeError("collector thread failed")
            timing = dict(self.timing)
            self._sync_actor()
            self._log(self.bufs[k], result["sps"], result["batch"], result["t"], t_upd, timing, logs,
                      iter_s=time.time() - t0)
            k = 1 - k

    def _log(self, buf: Rollout, sps, avg_batch, collect_s, update_s, timing, logs, iter_s=None):
        cfg = self.cfg
        self.iter += 1
        n = buf.size
        strategic = np.isin(buf.dtype[:n], list(spec.DecisionType.STRATEGIC)).mean()
        mean = lambda k: float(np.mean(self.stats[k])) if self.stats[k] else 0.0
        iter_s = iter_s if iter_s is not None else collect_s + update_s
        summary = {
            "iter": self.iter, "steps": self.total_steps, "trained_steps": self.trained_steps,
            "update_rows": self.train_rows_last, "collected_rows": n,
            "prefix_actions_observed": self.prefix_actions_observed,
            "restart_requested_episodes": self.restart_requested_episodes,
            "restart_actual_episodes": self.restart_actual_episodes, "completed_episodes": self.completed_episodes,
            "engine_errors_total": self.engine_errors_total,
            "dec/s": round(sps), "train_samples/s": round(self.train_rows_last / iter_s),
            "batch": round(avg_batch), "iter_s": round(iter_s, 1), "collect_s": round(collect_s, 1), "update_s": round(update_s, 1),
            "wait_s": round(timing.get("wait", 0), 1), "gpu_s": round(timing.get("gpu", 0), 1), "host_s": round(timing.get("host", 0), 1),
            "win%": 100 * mean("win"), "floor": mean("floor"), "act": mean("act"), "combats": mean("combats_won"),
            "episodes": len(self.stats["win"]), "strategic%": 100 * strategic, "err": len(self.stats["errors"]),
            "hp_labeled%": 100 * float(np.mean(~np.isnan(buf.t_hp[:n]))),
            **({"restart": {name: [round(100 * mean(f"restart_win_k{k}"), 1), len(self.stats[f"restart_win_k{k}"]),
                                   len(self.stats[f"restart_err_k{k}"])]
                            for k, name in shm.RESTART_KINDS.items() if self.stats[f"restart_win_k{k}"]}}
               if cfg.restart_prob > 0 else {}),
            **({"asc": {"level": self.asc_level,
                        "win": {a: [round(100 * mean(f"asc_win_{a}"), 1), len(self.stats[f"asc_win_{a}"])]
                                for a in range(0, cfg.asc_max + 1) if self.stats[f"asc_win_{a}"]}}}
               if cfg.asc_max > 0 else {}),
            **({"coop": {f"{k}p": [round(100 * mean(f"coop_win_n{k}"), 1), len(self.stats[f"coop_win_n{k}"]),
                                    len(self.stats[f"coop_err_n{k}"])]
                         for k in range(2, 5) if self.stats[f"coop_win_n{k}"]}}
               if cfg.coop_prob > 0 else {}),
            **({"by_char": {CHARACTERS[c]: [round(100 * mean(f"win_c{c}"), 1), round(mean(f"floor_c{c}"), 1)]
                            for c in range(len(CHARACTERS)) if self.stats[f"win_c{c}"]}}
               if self.cfg.characters == "all" or "," in self.cfg.characters else {}),
            "learner_peak_gb": round(torch.cuda.max_memory_allocated(self.dev) / 2**30, 2) if self.dev.type == "cuda" else 0,
            **{k: round(v, 4) for k, v in logs.items()},
        }
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True)
        if self.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.dev)
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                self.tb.add_scalar(k.replace("%", "_pct").replace("/", "_per_"), v, self.total_steps)
        if self.iter % 10 == 0 and self.deaths:
            self._log_deaths()
        if cfg.focus and self.iter % cfg.focus_round == 0:
            self._update_failmap()
        if self.iter % cfg.save_every == 0:
            self.save(os.path.join(cfg.out, "latest.pt"))
        if self.iter % (10 * cfg.save_every) == 0:  # keep a sparse history (~80 MB each)
            self.save(os.path.join(cfg.out, f"ckpt_{self.iter:06d}.pt"))

    def _log_deaths(self):
        """Where lost runs ended over the last 10 iterations: room types and the top killers."""
        total = sum(self.deaths.values())
        by_type = Counter()
        for (rt, _), c in self.deaths.items():
            by_type[shm.ROOM_TYPES[rt] if 0 <= rt < len(shm.ROOM_TYPES) else str(rt)] += c
        name = lambda i: self.vocab_names[i - 2] if 2 <= i < len(self.vocab_names) + 2 else str(i)
        top = [(name(rm), round(100 * c / total, 1)) for (rt, rm), c in self.deaths.most_common(12)]
        extra = {}
        if not self.cfg.focus:  # with the failure map, the final-boss split is logged per round there
            extra["final_bosses"] = self.bosses.summary()
            self.bosses.clear_round()
        print(json.dumps({"deaths": total, "death_room%": {k: round(100 * v / total, 1) for k, v in by_type.most_common()},
                          "top_killers%": top, **extra}), flush=True)
        self.deaths.clear()

    def close(self):
        # shared rollout blocks first: cheap, and nothing leaks if a slow shutdown gets killed
        # (processes that still map them keep their mapping)
        for b in self.bufs:
            if b._shm is not None:
                try:
                    b._shm.unlink()
                except Exception:
                    pass
        # the collector thread uses the pool: stop it before the pool's native state is freed
        self.stopping = True
        if self.collector is not None and self.collector.is_alive():
            self.collector.join(timeout=60)
        if getattr(self, "pool", None) is not None:
            self.pool.close()
        if self.world > 1 and getattr(self, "learner", None) is not None:
            try:
                self.learner.command(CMD_QUIT)
            except Exception:
                pass
        for p in self.helpers:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
        if self.world > 1 and dist.is_initialized():
            dist.destroy_process_group()
        for b in self.bufs:
            b.close(unlink=True)


def _free_port() -> int:
    import socket
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


def _learner_helper(rank, world, port, cfg_d, mcfg_d, device, shm_names, rows):
    """Learner rank >= 1: attach the shared rollouts, then run the updates rank 0 asks for."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    cfg, mcfg = PPOConfig(**cfg_d), ModelConfig(**mcfg_d)
    learner = Learner(cfg, mcfg, dev, rank, world)
    bufs = [Rollout(rows, name, create=False) for name in shm_names]
    try:
        while True:
            cmd, k, n, seed = learner.wait_command()
            if cmd == CMD_QUIT:
                break
            if cmd == CMD_UPDATE:
                learner.update(bufs[k], n, seed)
    finally:
        for b in bufs:
            b.close(unlink=False)
        dist.destroy_process_group()


def main():
    # variable batch shapes fragment the caching allocator; expandable segments avoid that
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    for f, v in asdict(PPOConfig()).items():
        ap.add_argument("--" + f.replace("_", "-"), type=type(v) if not isinstance(v, bool) else int, default=v)
    ap.add_argument("--sim-dll", required=True)
    ap.add_argument("--gamebin", required=True)
    ap.add_argument("--dotnet", default=os.path.expanduser("~/.dotnet/dotnet"))
    a = ap.parse_args()
    cfg = PPOConfig(**{f: getattr(a, f) for f in asdict(PPOConfig())})
    if cfg.init_mode not in ("resume", "weights") or cfg.init_weights not in ("model", "ema"):
        ap.error("init-mode must be resume/weights; init-weights must be model/ema")
    if cfg.practice_sampler not in ("uniform", "uncertainty") or cfg.focus_sampler not in ("failure", "uniform", "static"):
        ap.error("unknown restart sampler")
    if cfg.max_new_steps < 0 or not 0 <= cfg.restart_prob <= 1:
        ap.error("invalid budget or restart probability")
    if any(v not in (0, 1) for v in (cfg.boss2_situation, cfg.boss2_hp_stratified, cfg.fight_outcome, cfg.hp_label_fix)):
        ap.error("boss2-situation, boss2-hp-stratified, fight-outcome and hp-label-fix take 0 or 1")
    if cfg.boss2_hp_stratified and not cfg.boss2_situation:
        ap.error("--boss2-hp-stratified needs --boss2-situation 1")
    if cfg.practice_repeats < 1:
        ap.error("--practice-repeats must be >= 1")
    if not 0 <= cfg.fixed_update_rows <= cfg.steps_per_iter or cfg.fixed_update_rows and cfg.fixed_update_rows % cfg.minibatch:
        ap.error("fixed update rows must fit the rollout and be a multiple of minibatch size")
    seed_upper_bound(cfg.train_seed_bits)
    limit = cfg.train_seed_limit or seed_upper_bound(cfg.train_seed_bits)
    if cfg.seed_protocol not in ("legacy", "canonical") or cfg.seed_protocol == "canonical" and limit > CANONICAL_SEED_LIMIT:
        ap.error("canonical numeric seed range must fit ten base-34 characters")
    wcfg = WorkerConfig(sim_dll=a.sim_dll, gamebin=a.gamebin, dotnet=a.dotnet, ascension=cfg.ascension, characters=cfg.characters,
                        extra_args=(["--restart-prob", str(cfg.restart_prob)] if cfg.restart_prob > 0 else [])
                                   + (["--coop-prob", str(cfg.coop_prob), "--coop-max-players", str(cfg.coop_max_players)]
                                      if cfg.coop_prob > 0 else [])
                                   + ["--practice-sampler", cfg.practice_sampler, "--seed-protocol", cfg.seed_protocol]
                                   + (["--boss2-situation", "1"] if cfg.boss2_situation else [])
                                   + (["--boss2-hp-stratified", "1"] if cfg.boss2_hp_stratified else [])
                                   + (["--fight-outcome", "1"] if cfg.fight_outcome else [])
                                   + (["--practice-repeats", str(cfg.practice_repeats)] if cfg.practice_repeats > 1 else []),
                        numa=cfg.numa)
    # SIGTERM (timeout, kill) -> normal exit so the finally below stops workers and removes /dev/shm slots
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tr = None
    try:
        try:
            tr = Trainer(cfg, wcfg)
        except BaseException:
            # a failed start can leave NCCL / learner helpers that keep the interpreter from exiting (seen as a hang
            # that run_forever never notices): exit hard; the sim workers exit when their stdin pipe closes
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        tr.run()
        tr.save(os.path.join(cfg.out, "final.pt"))
        tr.save(os.path.join(cfg.out, "latest.pt"))
    finally:
        if tr is not None:
            tr.close()


if __name__ == "__main__":
    main()
