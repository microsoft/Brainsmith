"""Mathematical utilities for brainsmith."""


def divisors(n: int) -> set[int]:
    """Return all divisors of n.

    Uses sqrt optimization: only check up to sqrt(n), add both i and n/i.

    Args:
        n: Positive integer

    Returns:
        Set of all divisors

    Raises:
        ValueError: If n is not positive

    Example:
        >>> divisors(12)
        {1, 2, 3, 4, 6, 12}
        >>> divisors(768)
        {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768}

    Performance: O(√n) instead of O(n)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    result = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            result.add(i)
            result.add(n // i)
    return result


__all__ = ["divisors"]
