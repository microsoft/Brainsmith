#!/bin/bash  
  
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################


for fps in 1000 2000 3000; do  
  for heads in 12 24 36; do  
    for hidden_size in 384 192 96; do  
        #for bitwidth in 8 4; do  
                python endtoend.py -o ${heads}_${hidden_size}_${bitwidth}_${fps}.onnx -s step_hw_ipgen -n $heads -z $hidden_size -f $fps 
                mv intermediate_models ${heads}_${hidden_size}_${bitwidth}_${fps}  
            done  
        #done  
    done  
done  
