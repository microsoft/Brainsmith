import numpy as np

def shuffle_perfect_loopnest_coeffs(
        shape:tuple[int],
        perm:tuple[int]
    ) -> tuple[int]:
    """
    Given an input shape and permutation matrix calculate the
    coefficients for the perfect loop nest for HLS generation.
    """
    adjusted_shape = list(shape) + [1]
    input_coeffs = [np.prod(adjusted_shape[i+1:]) for i in range(len(shape))]
    out_coeffs = [input_coeffs[i] for i in perm]
    return tuple(out_coeffs)

def innerloop_moves(
        shape:tuple[int],
        perm:tuple[int]
    )->bool:
    """
    Returns true if the inner dimension moves
    otherwise returns false
    """
    innermost_original = len(shape) - 1
    new_position = perm.index(innermost_original)
    if new_position == len(perm) - 1:
        return False
    else:
        return True



