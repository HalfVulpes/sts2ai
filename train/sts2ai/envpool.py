"""Pool of C# simulator worker processes driven through pipes + shared-memory observation slots.

Each worker runs the *real* game logic (sts2.dll) headless and plays one run at a time.
Protocol (little endian):
  python -> worker : int32 cmd, int32 a, int64 b      (16 bytes)
      cmd 1 RESET  a = character index (-1 random), b = seed
      cmd 2 STEP   a = action index
      cmd 3 QUIT
  worker -> python : 1 byte b"R" after the slot was rewritten (header seq incremented)

The per-step hot path (waiting on all pipes, copying observations into batch arrays, writing
commands) is native code: see native/fastpool.c. Python only makes one call per batch.
"""
from __future__ import annotations

import atexit
import ctypes
import os
import resource
import signal
import subprocess
from dataclasses import dataclass, field

import numpy as np

from . import fastpool, shm, spec

CMD_RESET, CMD_STEP, CMD_QUIT = 1, 2, 3


@dataclass
class WorkerConfig:
    sim_dll: str
    gamebin: str
    dotnet: str = os.path.expanduser("~/.dotnet/dotnet")
    shm_dir: str = "/dev/shm"
    ascension: int = 0
    characters: str = "Ironclad"          # comma list or "all"
    extra_args: list[str] = field(default_factory=list)
    # NUMA placement, e.g. "0:0,2/1:1,3" (groups separated by "/"; ";" also accepted): worker i runs on the CPUs of group (i % n)'s first node with its
    # memory interleaved over that group's nodes (for hosts with CPU-less memory nodes next to the CPU nodes, where the
    # default policy plus NUMA balancing can fill the CPU nodes and swap while the others stay empty)
    numa: str = ""


class Worker:
    def __init__(self, idx: int, cfg: WorkerConfig, tag: str):
        self.idx = idx
        self.path = os.path.join(cfg.shm_dir, f"sts2ai_{tag}_{idx}.slot")
        with open(self.path, "wb") as f:
            f.truncate(shm.SLOT_BYTES)
        env = dict(os.environ, STS2_GAMEBIN=cfg.gamebin, DOTNET_ROOT=os.path.dirname(cfg.dotnet),
                   DOTNET_TieredPGO="1", DOTNET_gcServer="0",
                   DOTNET_GCHeapHardLimit="0xA000000")  # 160 MB cap: live heap ~45 MB; RSS ~250 MB instead of ~350
        args = [cfg.dotnet, cfg.sim_dll, "worker", "--slot", self.path, "--ascension", str(cfg.ascension),
                "--characters", cfg.characters, "--quiet"] + cfg.extra_args
        if cfg.numa:
            groups = [g.split(":") for g in cfg.numa.replace(";", "/").split("/")]
            cpu, mem = groups[idx % len(groups)]
            args = ["numactl", f"--cpunodebind={cpu}", f"--interleave={mem}"] + args
        with open(self.path + ".log", "wb") as log:  # the child keeps its own copy of the descriptor
            self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, bufsize=0)

    def close(self):
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except Exception:
                pass
        for path in (self.path, self.path + ".log"):
            try:
                if path == self.path or os.path.getsize(path) == 0:
                    os.unlink(path)
            except OSError:
                pass


class EnvPool:
    """Asynchronous pool of simulator workers with a batched interface.

    wait() returns the ids of workers holding a fresh observation; heads()/gather() copy their
    headers / observations into dense arrays; step()/reset() send commands to many workers at once.
    """

    def __init__(self, n: int, cfg: WorkerConfig, seed: int = 0, tag: str | None = None,
                 game_seeds: np.ndarray | None = None, seed_upper_bound: int = 2**62,
                 reserved_seeds=None):
        # ~3 descriptors per worker (2 pipes + epoll): lift the soft limit to the hard one
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # a dead worker -> EPIPE error, not a killed trainer
        self.n = n
        if not 1 < seed_upper_bound <= 2**62:
            raise ValueError("invalid training seed range")
        self.seed_upper_bound = seed_upper_bound
        self.reserved_seeds = np.unique(np.asarray([] if reserved_seeds is None else reserved_seeds, dtype=np.int64))
        if np.count_nonzero((self.reserved_seeds >= 1) & (self.reserved_seeds < seed_upper_bound)) >= seed_upper_bound - 1:
            raise ValueError("reserved seeds exhaust the training seed range")
        self.cfg = cfg
        self.tag = tag or f"{os.getpid()}"
        self._pool = None
        self.workers = []
        atexit.register(self.close)  # also on SystemExit before the owner could close us
        for i in range(n):
            self.workers.append(Worker(i, cfg, self.tag))
        L = fastpool.lib()
        paths = (ctypes.c_char_p * n)(*[w.path.encode() for w in self.workers])
        in_fds = np.array([w.proc.stdin.fileno() for w in self.workers], np.int32)
        out_fds = np.array([w.proc.stdout.fileno() for w in self.workers], np.int32)
        layout = np.array([shm.SLOT_BYTES, shm.OFF_TOKCAT, shm.OFF_TOKNUM, shm.OFF_ACTCAT, shm.OFF_ACTREF, shm.OFF_ACTNUM,
                           spec.N_TOK_CAT, spec.F_TOK, spec.N_ACT_CAT, spec.F_ACT], np.int64)
        self._pool = L.pool_create(n, paths, in_fds, out_fds, layout)
        if not self._pool:
            raise RuntimeError("pool_create failed")
        self._ids = np.empty(n, np.int32)
        self.rng = np.random.default_rng(seed)
        self.reset(np.arange(n, dtype=np.int32), game_seeds=game_seeds)

    def _check(self, r: int):
        if r < 0:
            i = -r - 1
            raise RuntimeError(f"worker {i} died (see {self.workers[i].path}.log)" if i < self.n else f"pool error {r}")

    def wait(self, min_ready: int = 1, timeout_s: float = 120.0, batch_wait_us: int = 2000) -> np.ndarray:
        """Ids (int32, ascending) of workers with a fresh observation; empty on timeout."""
        r = fastpool.lib().pool_wait(self._pool, min_ready, int(timeout_s * 1000), batch_wait_us, self._ids)
        self._check(r)
        return self._ids[:r].copy()

    def heads(self, ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(ids, np.int32)
        out = np.empty((len(ids), shm.HEADER_INTS), np.int32)
        fastpool.lib().pool_heads(self._pool, ids, len(ids), out)
        return out

    def gather(self, ids: np.ndarray, heads: np.ndarray) -> dict:
        """Observations of `ids` padded to the batch maximum (T tokens, A actions)."""
        ids = np.ascontiguousarray(ids, np.int32)
        B = len(ids)
        T = max(1, int(heads[:, shm.H_NTOK].max())) if B else 1
        A = max(1, int(heads[:, shm.H_NACT].max())) if B else 1
        tc = np.empty((B, T, spec.N_TOK_CAT), np.int16)
        tn = np.empty((B, T, spec.F_TOK), np.float32)
        ac = np.empty((B, A, spec.N_ACT_CAT), np.int16)
        ar = np.empty((B, A, 2), np.int16)
        an = np.empty((B, A, spec.F_ACT), np.float32)
        fastpool.lib().pool_gather(self._pool, ids, B, T, A, tc, tn, ac, ar, an)
        return dict(token_cat=tc, token_num=tn, action_cat=ac, action_ref=ar, action_num=an)

    def _send(self, ids, cmd: int, a=None, b=None):
        ids = np.ascontiguousarray(ids, np.int32)
        if not len(ids):
            return
        a = None if a is None else np.ascontiguousarray(a, np.int32)
        b = None if b is None else np.ascontiguousarray(b, np.int64)
        r = fastpool.lib().pool_send(self._pool, ids, len(ids), cmd,
                                     None if a is None else a.ctypes.data, None if b is None else b.ctypes.data)
        self._check(r)

    def step(self, ids: np.ndarray, actions: np.ndarray):
        self._send(ids, CMD_STEP, actions)

    def reset(self, ids: np.ndarray, character: int = -1, ascension: np.ndarray | None = None,
              situation: np.ndarray | None = None, game_seeds: np.ndarray | None = None):
        """New episodes; `ascension` (per id) overrides the workers' --ascension (encoded as (asc+1)*8 + character).
        `situation` (per id, needs `ascension`): -1 = fresh run, s = practise from a recorded point of situation s
        (bits 12+ of the argument; without it the workers decide about practice themselves)."""
        ids = np.atleast_1d(np.asarray(ids, np.int32))
        if game_seeds is None:
            seeds = self.rng.integers(1, self.seed_upper_bound, size=len(ids), dtype=np.int64)
            while len(self.reserved_seeds):
                forbidden = np.isin(seeds, self.reserved_seeds)
                if not forbidden.any():
                    break
                seeds[forbidden] = self.rng.integers(1, self.seed_upper_bound, size=int(forbidden.sum()), dtype=np.int64)
        else:
            seeds = np.asarray(game_seeds)
            if (seeds.dtype.kind not in "iu" or seeds.shape != (len(ids),)
                    or np.any(seeds < 0) or np.any(seeds > np.iinfo(np.int64).max)):
                raise ValueError("game_seeds must be a nonnegative int64 seed for each worker")
            seeds = seeds.astype(np.int64)
        a = np.full(len(ids), character, np.int32)
        if ascension is not None:
            a = (np.asarray(ascension, np.int32) + 1) * 8 + np.where(a >= 0, a, 7)
            if situation is not None:
                a = a + ((np.asarray(situation, np.int32) + 2) << 12)
        self._send(ids, CMD_RESET, a.astype(np.int32), seeds)

    def close(self):
        """Stop all workers and remove their slots. Idempotent."""
        if self._pool:
            try:
                self._send(np.arange(self.n, dtype=np.int32), CMD_QUIT)
            except Exception:
                pass
        workers, self.workers = self.workers, []
        for w in workers:
            if self._pool is None:
                w.proc.kill()  # never connected to the native pool: no QUIT was sent
            w.close()
        if self._pool:
            fastpool.lib().pool_destroy(self._pool)
            self._pool = None
