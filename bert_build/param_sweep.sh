#!/bin/bash  
  
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

for fps in 1000; do  
  for heads in 12; do  
    for hidden_size in 384; do  
        for bitwidth in 8; do  
        	for seqlen in 128; do  
                	python endtoend.py -o ${heads}_${hidden_size}_${bitwidth}_${fps}_${seqlen}.onnx -s step_minimize_bit_width -n $heads -z $hidden_size -f $fps -b $bitwidth -q ${seqlen} 
                	mv intermediate_models ${heads}_${hidden_size}_${bitwidth}_${fps}_$seqlen  
		done
            done  
        done  
    done  
done  
