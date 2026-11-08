"""Canonical ten-character game seeds, for cross-engine reproducibility.

The legacy driver used a different alphabet/act-selection RNG. Keep old results
labeled legacy; use this protocol for new preregistered experiments.
"""
ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
LIMIT = len(ALPHABET) ** 10


def from_number(value: int) -> str:
    if not 0 <= value < LIMIT:
        raise ValueError("canonical seed number must fit ten base-34 characters")
    result = []
    for _ in range(10):
        value, digit = divmod(value, 34)
        result.append(ALPHABET[digit])
    return "".join(result)


def to_number(seed: str) -> int:
    if len(seed) != 10 or any(c not in ALPHABET for c in seed):
        raise ValueError("expected ten canonical game-seed characters (no I or O)")
    return sum(ALPHABET.index(c) * 34**i for i, c in enumerate(seed))
