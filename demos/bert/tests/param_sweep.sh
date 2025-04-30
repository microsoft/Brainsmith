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
  for heads in 12 24; do  
    for hidden_size in 384 192; do  
        for bitwidth in 8 4; do  
        	for seqlen in 128 64 32; do  
                	python end2end_bert.py -o h${heads}_hs${hidden_size}_b${bitwidth}_t${fps}_s${seqlen} -s step_minimize_bit_width -n $heads -z $hidden_size -f $fps -b $bitwidth -q ${seqlen} 
		done
            done  
        done  
    done  
done  
