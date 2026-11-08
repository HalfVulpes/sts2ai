"""Situation distributions for controlled restart experiments (12 situations, or 13 with a3_boss2)."""
import numpy as np

SIT_COUNTS = (12, 13)


def situation_weights(loss, mode="failure", floor=0.02):
    loss = np.asarray(loss, dtype=np.float64)
    if loss.ndim != 1 or len(loss) not in SIT_COUNTS or not np.all(np.isfinite(loss)) or np.any(loss < 0):
        raise ValueError("loss must contain 12 (or 13) finite nonnegative values")
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and nonnegative")
    n = len(loss)
    if mode == "uniform":
        return np.full(n, 1 / n)
    if mode not in ("failure", "static"):
        raise ValueError(f"unknown situation sampler: {mode}")
    if mode == "static" or loss.sum() == 0:
        w = np.zeros(n)
        if n == 12:
            w[[8, 11, 7]] = [0.35, 0.45, 0.2]
        else:
            # a3_start, a3_boss, a3_boss2, a2_boss: the legacy act-3 boss share (0.45) split between the bosses
            w[[8, 11, 12, 7]] = [0.35, 0.25, 0.2, 0.2]
    else:
        w = loss / loss.sum()
    w = w + floor
    return w / w.sum()


def available_situations(rng, weights, masks, restart_probability):
    """Draw only available restart situations, preserving the requested mixture rate.

    An empty pool set falls back to a fresh run. The same availability rule is
    used by uniform and failure-weighted controls.
    """
    masks = np.asarray(masks, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    if weights.ndim != 1 or n not in SIT_COUNTS or np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("invalid situation weights")
    if masks.ndim != 1 or np.any(masks < 0) or np.any(masks >= 1 << n):
        raise ValueError("invalid availability mask")
    if not 0 <= restart_probability <= 1:
        raise ValueError("invalid restart probability")
    available = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)
    probabilities = available * weights[None, :]
    cdf = probabilities.cumsum(axis=1)
    total = cdf[:, -1]
    u = rng.random(len(masks)) * total
    selected = np.minimum((u[:, None] >= cdf).sum(axis=1), n - 1)
    restart = (rng.random(len(masks)) < restart_probability) & (total > 0)
    return np.where(restart, selected, -1).astype(np.int32)
