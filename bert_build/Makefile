############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

.PHONY: single_layer three_layer max_folding_three_layers

single_layer: l_1_n_12_z_384_i_1536.onnx
three_layer : l_3_n_12_z_384_i_1536.onnx
max_folding_three_layers: n3_simd48_pe32.onnx

l_1_n_12_z_384_i_1536.onnx: ./config/l_1_n_12_z_384_i_1536.json
	python endtoend.py -o l_1_n_12_z_384_i_1536.onnx -n 12 -l 1 -z 384 -i 1536 -p ./config/l_1_n_12_z_384_i_1536.json
	mv ./intermediate_models ./l_1_n_12_z_384_i_1536

l_3_n_12_z_384_i_1536.onnx: ./config/l_3_n_12_z_384_i_1536.json
	python endtoend.py -o l_3_n_12_z_384_i_1536.onnx -n 12 -l 3 -z 384 -i 1536 -p ./config/l_3_n_12_z_384_i_1536.json
	mv ./intermediate_models ./l_3_n_12_z_384_i_1536
	
n3_simd48_pe32.onnx: ./config/n3_simd48_pe32.json
	python endtoend.py -o n3_simd48_pe32.onnx -n 12 -l 1 -z 384 -i 1536 -p ./config/n3_simd48_pe32.json
	mv ./intermediate_models ./n3_simd48_pe32
