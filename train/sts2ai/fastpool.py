"""ctypes binding for native/fastpool.c (built on first use with the system C compiler)."""
from __future__ import annotations

import ctypes as C
import os
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "native", "fastpool.c")
_LIB_PATH = os.path.join(_HERE, "..", "native", "fastpool.so")
_lib = None


def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH) or os.path.getmtime(_LIB_PATH) < os.path.getmtime(_SRC):
            tmp = f"{_LIB_PATH}.{os.getpid()}.tmp"
            subprocess.check_call(["cc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, _SRC, "-lm"])
            os.replace(tmp, _LIB_PATH)
        L = C.CDLL(_LIB_PATH)
        P = C.c_void_p
        i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        i16p = np.ctypeslib.ndpointer(np.int16, flags="C_CONTIGUOUS")
        i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
        f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
        L.pool_create.restype = P
        L.pool_create.argtypes = [C.c_int, C.POINTER(C.c_char_p), i32p, i32p, i64p]
        L.pool_destroy.argtypes = [P]
        L.pool_wait.restype = C.c_int
        L.pool_wait.argtypes = [P, C.c_int, C.c_int, C.c_long, i32p]
        L.pool_heads.argtypes = [P, i32p, C.c_int, i32p]
        L.pool_gather.argtypes = [P, i32p, C.c_int, C.c_int, C.c_int, i16p, f32p, i16p, i16p, f32p]
        L.pool_send.restype = C.c_int
        L.pool_send.argtypes = [P, i32p, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        L.gae.argtypes = [C.c_long, f32p, f32p, u8p, u8p, f32p, i32p, C.c_int, C.c_float, C.c_float, f32p, f32p]
        _lib = L
    return _lib


def gae(reward, value, done, trunc, next_value, env, n_env: int, gamma: float, lam: float):
    """Per-env reverse scan; rows of one env must be in time order. Returns (advantages, lambda-returns)."""
    n = len(reward)
    adv = np.empty(n, np.float32)
    ret = np.empty(n, np.float32)
    f = lambda a: np.ascontiguousarray(a, np.float32)
    lib().gae(n, f(reward), f(value), np.ascontiguousarray(done, np.uint8), np.ascontiguousarray(trunc, np.uint8),
              f(next_value), np.ascontiguousarray(env, np.int32), n_env, gamma, lam, adv, ret)
    return adv, ret
