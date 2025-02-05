############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np
import copy

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

def simplify_transpose(shape, perm):
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


class ParallelInnerShuffle:

    def __init__(self, shape:tuple[int,...], perm:tuple[int,...], simd:int, relax:bool=False)->None:
        """ 
            ParallelInnerShuffle is used to represent and model a shuffle operation
            where the inner dimension moves as part of the shuffle.

            If it is able to, it will analytically solve what the write bank period
            and read bank periods should be.
        """
        self.shape = shape
        self.perm = perm
        self.simd = simd
        self._relax = relax

        # Check for constraints? 
        #         Should I throw an error here? Or fallback to a SIMD=1?
        self._checks()

        # Internal representation of unbanked memory layout
        # Each memory location is given a unique signature that we can
        # use for tracking how bank allocation has happened.
        self.in_mem_layout, self.in_addr_map = self._populate_unbanked_layout()

        # How SIMD data items are allocated to banks (can be rotated)
        self.wr_alloc = list(range(0,self.simd))
        self.rd_alloc = [0] + [self.simd - i for i in range(1, self.simd)]

        # Internal model of hardware banks
        self.banks = []
        for i in range(0, self.simd):
            self.banks.append([])

        self._solve_rotation_periods()

        # Verify that the transformation and banking was correct
        if not self.validate:
            raise RuntimeError(f"Error! The bank scheduling for {self.name} is not correct shape:{self.shape} wr_rot_period={self.wr_rot_period} rd_rot_period={self.rd_rot_period}")
 

    def _solve_rotation_periods(self)->None:
        """ 
            Solves the rotation period of the inner dim transpose
            sets wr_rot_period and rd_rot_period either analytically (if possible)
            or via brute force search.
        """
        # Attempt to analytically solve the bank scheduling problem
        if len(self.shape) == 2 and len(self.perm) == 2:
            self.wr_rot_period = calculate_wr_rot_period(self.simd, self.shape[-1])
            self.rd_rot_period = int(self.shape[0]/self.simd)
            self._attempt_allocation() # Should pass with no issues here 
        else:
            # Brute force solve
            self.wr_rot_period = self._brute_force_wr_rot_period()
            self.rd_rot_period = self._brute_force_rd_rot_period()

    def _populate_unbanked_layout(self) -> tuple[dict[int,tuple[int,...]], dict[tuple[int,...], int]]:
        """ 
                Creates a data layout that is not considering banking for an arbitrary number of dimensions.
                Each element has a unique signature that can be used to track it as different banking schemes are used.
                returns two mappings:
                        * an indices tuple to linear address :  int -> tuple[int, ...]
                        * a linear address to indices tuple  :  tuple[int, ...] -> int

                This is used at various points for either checking that a schedule is correct, or bruteforcing one
        """
        in_mem_layout = {}
        in_addr_map = {}
        total_elements = 1
        for dim in self.shape:
            total_elements *= dim

        for addr in range(total_elements):
            indices = []
            remainder = addr
            for dim in reversed(self.shape):
                indices.append(remainder % dim)
                remainder //= dim
            indices.reverse()
            in_mem_layout[addr] = tuple(indices)
            in_addr_map[tuple(indices)] = addr
        return (in_mem_layout, in_addr_map)

    def _brute_force_wr_rot_period(self, give_up=500)->int:
        """ 
            Find a feasible rotation pointer for this setup.
            Will attempt and allocation, ensure there are no colflicts, and then ensure that we can have a constant rd_ropt_period.
            There is plenty of room for optimisation here and improving the search speed.
        """
        for i in range(give_up):
            self.wr_rot_period = i
            self._attempt_allocation()
            if not self._detect_bank_conflict() and (len(set(self._brute_force_rd_rot_period()))==1):
                return i
        return None

    def _brute_force_rd_rot_period(self)->int:
        """ 
            From the access pattern exhaustively determine the 
            read period from the banks and access pattern.

            Again this is currently quite a slow operation with space to optimise
        """
        first = True
        curr_count = 0
        counts = []
        for pattern in self.read_bank_schedule:
            if not first:
                curr_count += 1
                if prev_pattern != pattern:
                    counts.append(curr_count)
                    curr_count = 0
            first = False
            prev_pattern = pattern
        return counts

    def _detect_bank_conflict(self)->bool:
        """ 
            Walks through the read pattern and determines if there are any bank conflicts 
        """
        conflict = False
        for rd in self.addr(self.readorder):
            if not self._disjoint_banks(rd):
                conflict = True
        return conflict

    def _disjoint_banks(self, items)->bool:
        """
           For a given set of items return true if they are in disjoint banks
        """
        for bank_idx, bank in enumerate(self.banks):
            hits=[]
            for i in items:
                if i in bank:
                    hits.append(i)
            if len(hits) != 1:
                return False
        return True

    def _clear_banks(self):
        """ 
            Clears all the data from the banks 
        """
        self.banks = []
        self._plot_data = []
        for i in range(0,self.simd):
            self.banks.append([])

    def _attempt_allocation(self):
        """ 
            Attempts to allocate based on the bank write rotation period 
            self.wr_rot_period 
        """
        self._clear_banks()
        rot_count = 0
        for wr in self.addr(self.writeorder):
            for b in range(self.simd):
                self.banks[self.wr_alloc[b]].append(wr[b])
            rot_count += 1
            if rot_count == self.wr_rot_period:
                self.rotate_wr_alloc()
                rot_count = 0

    @property
    def validate(self)->bool:
        """ 
            Validates that the transpose was correct. 
            This will use python models of the banks, write rotation
            and read rotation to compare against a numpy implementation.
        """
        seen=0 # For more helpful error reporting
        ref = self._create_numpy_readpattern
        #for rd_simd in self.read:
        for rd_simd in self.addr(self.readorder):
            for rd in rd_simd:
                item = ref.pop(0)
                if item != rd:
                    raise RuntimeError(f"""
                    Error: mismatch between the numpy golden reference and the parallel shuffle.
                    expected {item}, but got, {rd} on read index {seen}
                    """)
                seen += 1
        return True

    @property
    def _create_numpy_readpattern(self)->list[int]:
        """ 
          Create the golden read pattern using the numpy.transpose op.
          This is used for validation. 
        """
        total_elements = np.prod(self.shape)
        flat_array = np.arange(total_elements)
        input_array = flat_array.reshape(self.shape)

        reshaped_array = np.transpose(input_array, self.perm)
        assert (reshaped_array.shape == self.postshape), f"Error: the calculated postshape {self.postshape} does not match the numpy model {reshaped_array.shape}"

        # Iterate through the transposed array
        read_pattern = []
        for index, value in np.ndenumerate(reshaped_array):
            read_pattern.append(value)
        return read_pattern



    @property
    def read(self)->list[int]:
        """ 
          Behaviourally read using the read pointer from the banks
          and generate a flattened list of the output in order.
          This is mimicking the same behaviour as the hardware unit. 
        """
        overall_output = []
        count=0
        for simd_row in self.addr(self.readorder):
            t = [int(x/self.simd) for x in simd_row] # The bank addresses
            out_line = []
            for b in range(self.simd):
                out_line.append(self.banks[self.rd_alloc[b]][t[b]])
            count += 1
            if count == self.rd_rot_period:
                self.rotate_rd_alloc()
                count=0
            overall_output.append(out_line)
        return overall_output

    def rotate_wr_alloc(self)->None:
        self.wr_alloc = self.wr_alloc[-1:] + self.wr_alloc[:-1]

    def rotate_rd_alloc(self)->None:
        self.rd_alloc = self.rd_alloc[-1:] + self.rd_alloc[:-1]

    def addr(self, order:list[tuple[int,...]])->list[int]:
        """ 
            For a given read/write order in terms of indices tuple
            return the address in local memory model (index) 
        """
        res = []
        for s in order:
            res_s = []
            for i in s:
                res_s.append(self.in_addr_map[i])
            res.append(res_s)
        return res

    @property
    def writeorder(self)->list[tuple[int,...]]:
        """ 
            Recursivley walk the input shape and generate the write order, taking into account SIMD 
        """
        result = []
        simd = []

        def nested_loop(dim:int, indices:list[int]):
            if dim == len(self.shape):
                simd.append(tuple(indices))
                if len(simd) == self.simd:
                    result.append(copy.deepcopy(simd))
                    simd.clear()
                return
            for i in range(self.shape[dim]):
                nested_loop(dim + 1, indices + [i])

        nested_loop(0, [])
        return result

    @property
    def readorder(self)->list[tuple[int,...]]:
        """ 
           Recursivly walk the postshape and generate the readorder, taking into account SIMD
        """
        result = []
        simd = []

        def nested_loop(dim:int, indices:list[int]):
            if dim == len(self.postshape):
                reordered_indices = tuple(indices[self.perm.index(i)] for i in range(len(self.shape)))
                simd.append(tuple(reordered_indices))
                if len(simd) == self.simd:
                    result.append(copy.deepcopy(simd))
                    simd.clear()
                return
            for i in range(self.postshape[dim]):
                nested_loop(dim + 1, indices + [i])

        nested_loop(0, [])
        return result

    @property
    def read_bank_schedule(self)->list[list[int]]:
        """ 
           Returns the order in which the banks are read from every iteration 
        """
        bank_schedule = []
        for rd_simd in self.addr(self.readorder):
            bs_simd = []
            for rd in rd_simd:
                b,i = self._find_bank_location(rd)
                bs_simd.append(b)
            bank_schedule.append(bs_simd)
        return bank_schedule

    def _find_bank_location(self, item:int)->tuple[int,int]:
        """ 
            Given a unique index to an item find it's bank and the index into that bank 
        """
        for bidx, b in enumerate(self.banks):
            for lidx, l in enumerate(b):
                if item==l:
                    return (bidx, lidx)
        raise RuntimeError(f"Unable to find item {item} in the banks")

    @property
    def postshape(self)->tuple[int,...]:
        """ Returns what the shape of the output will be after the operation """
        return tuple([self.shape[x] for x in self.perm])

    def _checks(self)->None:
        """ Checks the sanity of the inputs, raises exceptions if they are incorrect """
        assert len(self.perm) == len(self.shape), f"Error: input has {len(self.shape)} dimensions but {len(self.perm)} permutations are specified"
        assert all(x <= (len(self.shape)-1) for x in self.perm), f"Error: one permutation is greater than the dimensions specified, {self.perm=}"
        assert len(self.perm) == len(set(self.perm)), f"Error: One or more permutation points at the same dimension {self.perm=}"

        if not self._relax:
            assert self.postshape[-1] % self.simd == 0, f"Error: Currently there is the constraint that for the shuffle the final inner dimension % simd == 0, in this case final shape is {self.postshape} with inner dimension {self.postshape[-1]} which {self.postshape[-1]} % {self.simd} = {self.postshape[-1]%self.simd}"


















