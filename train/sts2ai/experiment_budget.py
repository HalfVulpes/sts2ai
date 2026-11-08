"""Budget helpers kept independent of PyTorch for public source tests."""


def update_rows(collected: int, fixed: int) -> int:
    if collected <= 0 or fixed < 0:
        raise ValueError("invalid rollout budget")
    if fixed and collected < fixed:
        raise ValueError("rollout shorter than the predeclared update budget")
    return fixed or collected


def seed_upper_bound(bits: int) -> int:
    if not 1 <= bits <= 62:
        raise ValueError("training seed bits must be between 1 and 62")
    return 1 << bits
