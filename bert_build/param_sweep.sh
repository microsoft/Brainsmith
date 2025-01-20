#!/bin/bash  
  
for heads in 12 24 36; do  
    for hidden_size in 384 192 96; do  
        for bitwidth in 8 4; do  
            for fps in 1000 2000 3000; do  
                python endtoend.py -o ${heads}_${hidden_size}_${bitwidth}_${fps}.onnx -s step_hw_ipgen -l 1  
                mv intermediate_models ${heads}_${hidden_size}_${bitwidth}_${fps}  
            done  
        done  
    done  
done  
