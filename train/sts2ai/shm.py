"""Shared-memory observation slots written by the C# simulator workers (see shared/Encoding/ShmLayout.cs)."""
from __future__ import annotations

import mmap
import os

import numpy as np

from . import spec

HEADER_INTS = 32
OFF_HEADER = 0
OFF_TOKCAT = OFF_HEADER + HEADER_INTS * 4
OFF_TOKNUM = OFF_TOKCAT + spec.MAX_TOKENS * spec.N_TOK_CAT * 2
OFF_ACTCAT = OFF_TOKNUM + spec.MAX_TOKENS * spec.F_TOK * 4
OFF_ACTREF = OFF_ACTCAT + spec.MAX_ACTIONS * spec.N_ACT_CAT * 2
OFF_ACTNUM = OFF_ACTREF + spec.MAX_ACTIONS * 2 * 2
SLOT_BYTES = OFF_ACTNUM + spec.MAX_ACTIONS * spec.F_ACT * 4

# header indices (keep in sync with ShmLayout.cs)
H_SEQ, H_DECISION, H_NTOK, H_NACT, H_DONE, H_WON, H_ACT, H_ACTFLOOR, H_TOTALFLOOR, H_HP, H_MAXHP, H_GOLD, \
    H_INCOMBAT, H_COMBATID, H_DECKSIZE, H_ERROR, H_CHARACTER, H_ASCENSION, H_BOSSKILLS, H_ELITEKILLS, \
    H_COMBAT_HP_START, H_DROPPED_TOK, H_DROPPED_ACT, H_STEP_IN_RUN, H_COMBATS_WON, H_ROOMTYPE, H_ROOMMODEL, \
    H_RESTART, H_COOP = range(29)
H_PREFIX_ACTIONS, H_REQUESTED_RESTART, H_AVAILABLE_RESTARTS = 29, 30, 31
# practice situations (Worker.cs): sit = act*4 + kind; header H_RESTART = sit + 1 (0 = fresh run).
# With --boss2-situation the second final boss of a double-boss act (A10 act 3) is situation 12, a3_boss2.
SIT_KINDS = ["start", "monster", "elite", "boss"]
N_SIT = 12
N_SIT_BOSS2, SIT_BOSS2 = 13, 12
RESTART_KINDS = {**{s + 1: f"a{s // 4 + 1}_{SIT_KINDS[s % 4]}" for s in range(N_SIT)}, SIT_BOSS2 + 1: "a3_boss2"}
ROOM_TYPES = ["none", "monster", "elite", "boss", "treasure", "shop", "event", "rest", "map"]


class Slot:
    """Numpy views onto one environment's observation slot."""

    def __init__(self, path: str, create: bool = True):
        self.path = path
        if create:
            with open(path, "wb") as f:
                f.truncate(SLOT_BYTES)
        with open(path, "r+b") as f:  # the mapping stays valid after the file is closed
            self.mm = mmap.mmap(f.fileno(), SLOT_BYTES)
        buf = self.mm
        self.header = np.frombuffer(buf, dtype=np.int32, count=HEADER_INTS, offset=OFF_HEADER)
        self.tok_cat = np.frombuffer(buf, dtype=np.int16, count=spec.MAX_TOKENS * spec.N_TOK_CAT,
                                     offset=OFF_TOKCAT).reshape(spec.MAX_TOKENS, spec.N_TOK_CAT)
        self.tok_num = np.frombuffer(buf, dtype=np.float32, count=spec.MAX_TOKENS * spec.F_TOK,
                                     offset=OFF_TOKNUM).reshape(spec.MAX_TOKENS, spec.F_TOK)
        self.act_cat = np.frombuffer(buf, dtype=np.int16, count=spec.MAX_ACTIONS * spec.N_ACT_CAT,
                                     offset=OFF_ACTCAT).reshape(spec.MAX_ACTIONS, spec.N_ACT_CAT)
        self.act_ref = np.frombuffer(buf, dtype=np.int16, count=spec.MAX_ACTIONS * 2,
                                     offset=OFF_ACTREF).reshape(spec.MAX_ACTIONS, 2)
        self.act_num = np.frombuffer(buf, dtype=np.float32, count=spec.MAX_ACTIONS * spec.F_ACT,
                                     offset=OFF_ACTNUM).reshape(spec.MAX_ACTIONS, spec.F_ACT)

    def close(self):
        # numpy views must be dropped before the mmap can close
        for k in ("header", "tok_cat", "tok_num", "act_cat", "act_ref", "act_num"):
            self.__dict__.pop(k, None)
        try:
            self.mm.close()
        except BufferError:
            pass
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
