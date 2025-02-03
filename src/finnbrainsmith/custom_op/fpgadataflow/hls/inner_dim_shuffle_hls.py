############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np
import os
import copy

from finnbrainsmith.custom_op.fpgadataflow import brainsmith_templates
from finnbrainsmith.custom_op.fpgadataflow.brainsmith_hlsbackend import BS_HLSBackend
from finnbrainsmith.custom_op.fpgadataflow.shuffle import Shuffle 
from finnbrainsmith.transformation.shuffle_helpers import simplify_transpose, calculate_wr_rot_period 
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.basic import CppBuilder

class InnerDimShuffle_hls(Shuffle, BS_HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

        # Assert that it does have an inner dim moving
        if self.get_nodeattr("inner_moves") != 1:
            raise RuntimeError(f"InnerDimShuffle created for a Shuffle where inner dim is not moved")
	
        self._relax = False # Used to constrain the input to space where feasible schedule can be found via brute force

        self.simd = self.get_nodeattr("SIMD")

        # Attempt to simplify the transpose
        self.shape = self.get_nodeattr("in_reshaped")
        self.perm = self.get_nodeattr("perm")
        self.shape, self.perm = simplify_transpose(self.shape, self.perm)

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
        self.set_nodeattr("wr_rot_period", self.wr_rot_period)
        self.set_nodeattr("rd_rot_period", self.rd_rot_period)

    def get_nodeattr_types(self):
        attr = Shuffle.get_nodeattr_types(self) | BS_HLSBackend.get_nodeattr_types(self)
        attr["wr_rot_period"] = ("i", True, None) 
        attr["rd_rot_period"] = ("i", True, None)
        return attr 

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
            self._attempt_allocation(i)
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
        assert len(set(counts)) == 1, f"We do not have a static read rotation period rd_rot_period={counts} {self.shape=} {self.simd=}"
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
        for rd_simd in self.read:
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


    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "input_gen.hpp"',
            '#include <ap_int.h>',
            '#include <hls_vector.h>',
            '#include <hls_stream.h>',
        ]

    def defines(self, var):
        simd = self.get_nodeattr("SIMD")
        dtype = self.get_input_datatype()
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD = {simd}; 
            using  TE = {dtype.get_hls_datatype_str()};
            using  TV = hls::vector<TE, SIMD>;
            """
        ]

    def get_exp_cycles(self):
        out_shape = self.get_nodeattr("out_shape")
        simd = self.get_nodeattr("SIMD")
        return int(np.prod(out_shape)/simd) 

    def docompute(self):
        simd = self.get_nodeattr("SIMD")
        out_shape = self.get_nodeattr("out_shape")
        out_shape[-1] = int(out_shape[-1]/simd)
        loop_coeffs = [1 if x == 1 else int(x/simd) for x in self.get_nodeattr("loop_coeffs")]  
        interleaved = [int(item) for pair in zip(out_shape, loop_coeffs) for item in pair] 
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            hls::stream<TV>  src0;
	    hls::stream<TV>  dst0;
            #pragma HLS stream variable=src0 depth=2
            #pragma HLS stream variable=dst0 depth=2

            move(in0_{self.hls_sname()}, src0);
	    input_gen<-1,{np.prod(out_shape)},{','.join(map(str,interleaved))}>(src0, dst0);
	    move(dst0, out_{self.hls_sname()});
            """
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name} (
                hls::stream<TV> &in0_{self.hls_sname()},
	        hls::stream<TV> &out_{self.hls_sname()}
            )
            """
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            f"""
            #pragma HLS interface AXIS port=in0_{self.hls_sname()}
            #pragma HLS interface AXIS port=out_{self.hls_sname()}
	    #pragma HLS aggregate variable=in0_{self.hls_sname()} compact=bit
	    #pragma HLS aggregate variable=out_{self.hls_sname()} compact=bit

            #pragma HLS interface ap_ctrl_none port=return
            #pragma HLS dataflow disable_start_propagation
            """
        ]


    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        folded_ishape = self.get_folded_input_shape()
        export_dt = self.get_input_datatype()

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        inp = context[node.input[0]]
        inp = inp.reshape(folded_ishape)
        np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)
        

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # Load output npy file
            super().npy_to_dynamic_output(context)
        elif mode =="rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                f"{code_gen_dir}/input_0.npy", export_dt, nbits 
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)

            io_dict = {
                "inputs" : {"in0" : rtlsim_inp},
                "outputs" : {"out" : []}
            }
            self.rtlsim_multi_io(sim, io_dict)

            out = io_dict["outputs"]["out"]
            target_bits = export_dt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = f"{code_gen_dir}/output.npy"
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(out, out_npy_path, export_dt, out_shape, packed_bits, target_bits)

            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32,).reshape(*oshape)
            context[node.output[0]] = output

        else:
            raise Exception(f"Unsupported execution mode: {mode}")


    def compile_singlenode_code(self):
        """
        Builds the bash script for compilation using the CppBuilder from
        finn.util.basic and executes the script to produce the executable
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        builder = CppBuilder()
        # to enable additional debug features please uncommand the next line
        # builder.append_includes("-DDEBUG")
        builder.append_includes("-I$FINN_ROOT/src/finn/qnn-data/cpp")
        builder.append_includes("-I$FINN_ROOT/deps/cnpy/")
        builder.append_includes("-I$FINN_ROOT/deps/finn-hlslib")
        builder.append_includes("-I$FINN_ROOT/deps/finnbrainsmith/hlslib_extensions")
        #builder.append_includes("-I{}/include".format(os.environ["HLS_PATH"]))
        builder.append_includes("-I{}/include".format(os.environ["VITIS_PATH"]))
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("$FINN_ROOT/deps/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)


    def code_generation_cppsim(self, model):
        """Generates c++ code for simulation (cppsim)."""
        self.code_gen_dict["$READNPYDATA$"] = [""]
        self.code_gen_dict["$DATAOUTSTREAM$"] = [""]
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [""]
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.generate_params(model, path)
        self.global_includes()
        self.defines("cppsim")
        self.pragmas()
        oshape = self.get_folded_output_shape()
        oshape_str = str(oshape).replace("(", "{").replace(")", "}")

        simd = self.get_nodeattr("SIMD")
        out_shape = self.get_nodeattr("out_shape")
        out_shape[-1] = int(out_shape[-1]/simd)
        loop_coeffs = [1 if x == 1 else int(x/simd) for x in self.get_nodeattr("loop_coeffs")]  
        interleaved = [int(item) for pair in zip(out_shape,loop_coeffs) for item in pair]

        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            static hls::stream<TV>  in0_V;
            static hls::stream<TV>  out_V;

            npy2vectorstream<TE, float, SIMD>("{path}/input_0.npy", in0_V);
            int stream_size = in0_V.size();

            while(out_V.size() != stream_size) {{
                input_gen<-1,{np.prod(out_shape)},{','.join(map(str,interleaved))}>(in0_V, out_V);
            }}

            vectorstream2npy<TE, float, SIMD>(out_V,{oshape_str}, "{path}/output.npy");
            """
        ]
        self.save_as_npy()

        template = brainsmith_templates.docompute_template

        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim") + f"/execute_{node.op_type}.cpp"
        with open(code_gen_dir, "w") as f:
            for key in self.code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(self.code_gen_dict[key])
                template = template.replace(key, code_gen_line)
            f.write(template)

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

