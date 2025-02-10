############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

# Warning these premade recipies can be quite fragile. If there are changes in the compiler the configuration can become stale and 
# incorrect meaning that the fifo depth step needs to be rerun to regenerate the configuration.

.PHONY: single_layer three_layer max_folding_three_layers

single_layer: l_1_n_12_z_384_i_1536.onnx
three_layer : l_3_n_12_z_384_i_1536.onnx
folding_three_layers: l3_simd24_pe16.onnx
max_folding_three_layers: l3_simd48_pe32.onnx
small_folding_three_layers: l3_simd12_pe8.onnx

l_1_n_12_z_384_i_1536.onnx: ./config/l_1_n_12_z_384_i_1536.json
	python endtoend.py -o l_1_n_12_z_384_i_1536.onnx -n 12 -l 1 -z 384 -i 1536 -x False -p ./config/l_1_n_12_z_384_i_1536.json
	mv ./intermediate_models ./l_1_n_12_z_384_i_1536

l_3_n_12_z_384_i_1536.onnx: ./config/l_3_n_12_z_384_i_1536.json
	python endtoend.py -o l_3_n_12_z_384_i_1536.onnx -n 12 -l 3 -z 384 -i 1536 -x False -p ./config/l_3_n_12_z_384_i_1536.json
	mv ./intermediate_models ./l_3_n_12_z_384_i_1536
	
l3_simd24_pe16.onnx: 
	python scripts/gen_initial_folding.py --simd 24 --pe 16 --num_layers 3 -o l3_simd24_pe16.json
	python endtoend.py -o l3_simd24_pe16.onnx -n 12 -l 3 -z 384 -i 1536 -x True -p ./l3_simd24_pe16.json
	mv ./intermediate_models ./l3_simd24_pe16

l3_simd48_pe32.onnx: 
	python scripts/gen_initial_folding.py --simd 24 --pe 16 --num_layers 3 -o l3_simd48_pe32.json
	python endtoend.py -o l3_simd48_pe32.onnx -n 12 -l 3 -z 384 -i 1536 -x True -p ./l3_simd48_pe32.json
	mv ./intermediate_models ./l3_simd48_pe32

l3_simd12_pe8.onnx: 
	python scripts/gen_initial_folding.py --simd 24 --pe 16 --num_layers 3 -o l3_simd12_pe8.json
	python endtoend.py -o l3_simd12_pe8.onnx -n 12 -l 3 -z 384 -i 1536 -x True -p ./l3_simd12_pe8.json
	mv ./intermediate_models ./l3_simd12_pe8
