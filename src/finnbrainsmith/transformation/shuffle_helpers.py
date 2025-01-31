############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

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

def prime_factorial(n:int)->list[int]:
    """
    Calculates the set of prime factors of n.
    This is used in the creation of the bank
    scheduling for 2D transposes.
    """
    if n < 4:
        return [n]
    arr = []
    while n > 1:
        for i in range(2, int(2+n//2)):
            if i == (1 + n // 2):
                arr.append(n)
                n = n // n
            if n % i == 0:
                arr.append(i)
                n = n // i
                break
    return arr 

def wr_rot_factor(factor:int, i:int)->int:
    if (i % factor) != 0:
        return 0
    else:
        return int(i/factor)

def calculate_wr_rot_period(simd:int, i:int)->int:
    """ For cases where we have a direct transpose we can use this
    function to analytically calculate the shuffle pattern 
    
    i : is the input innernmost dimension
    """
    factors = prime_factorial(simd)
    if (i%simd) != 0:
        partial_wr_rot_factor = partial(wr_rot_factor, i=i) 
        return max(map(partial_wr_rot_factor, factors))
    else:
        return int(i/simd)

def simplify_transpose(self, shape, perm):
   """Detect if a multi-dimensional transpose can be reduced to a 2D transpose 
   and return the simplified transpose.

   It attempts to squeeze singular dimensions, find groups that move together, etc..
   If it is unable to simplify the shape it returns the original shape. 

   As an input take the original shape and permutation matrix
   return the new simplifed shape and permutation matrix
   
   """
   if len(shape) != len(perm):
       raise ValueError("Shape and permutation must have the same length")

   new_shape = []
   mapping = {}  # Old index → New index after squeezing
   
   new_perm = []
   new_index = 0

   for old_index, dim in enumerate(shape): # Squeeze the dims
       if dim > 1:
           mapping[old_index] = new_index
           new_shape.append(dim)
           new_index += 1

   # Adjust the permutation to match the new shape indices
   for old_index in perm:
       if old_index in mapping:
           new_perm.append(mapping[old_index])

   # Check if perm is now a valid permutation of new_shape indices
   if sorted(new_perm) != list(range(len(new_perm))):
       raise ValueError("Invalid permutation indices after adjustment")

   # Find contiguous groups before and after the permutation
   def find_groups(shape, perm):
       groups = []
       temp_group = [shape[0]]

       for i in range(1, len(shape)):
           if perm[i] == perm[i - 1] + 1:  # Check if indices stayed together
               temp_group.append(shape[i])
           else:
               groups.append(temp_group)
               temp_group = [shape[i]]
       groups.append(temp_group)
       return groups

   original_groups = find_groups(new_shape, perm)
   transposed_groups = find_groups([new_shape[i] for i in new_perm], new_perm)

   # If exactly two groups swap places, reduce to a 2D transpose
   if len(original_groups) == 2 and len(transposed_groups) == 2:
       simplified_original = (np.prod(original_groups[0]), np.prod(original_groups[1]))
       simplified_transposed = (simplified_original[1], simplified_original[0])
       return simplified_original, (1,0)
   else:
       return shape, perm


