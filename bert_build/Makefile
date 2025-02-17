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

folding_three_layers: l3_simd24_pe16.onnx
max_folding_three_layers: l3_simd48_pe32.onnx
small_folding_three_layers: l3_simd12_pe8.onnx

l3_simd24_pe16.onnx: 
	python scripts/gen_initial_folding.py --simd 24 --pe 16 --num_layers 3 -o l3_simd24_pe16.json
	python endtoend.py -o l3_simd24_pe16.onnx -n 12 -l 3 -z 384 -i 1536 -x True -p ./l3_simd24_pe16.json
	mv ./intermediate_models ./l3_simd24_pe16

l3_simd48_pe32.onnx: 
	python scripts/gen_initial_folding.py --simd 48 --pe 32 --num_layers 3 -o l3_simd48_pe32.json
	python endtoend.py -o l3_simd48_pe32.onnx -n 12 -l 3 -z 384 -i 1536 -x True -p ./l3_simd48_pe32.json
	mv ./intermediate_models ./l3_simd48_pe32

l3_simd12_pe8.onnx: 
	python scripts/gen_initial_folding.py --simd 12 --pe 8 --num_layers 3 -o l3_simd12_pe8.json
	python endtoend.py -o l3_simd12_pe8.onnx -n 12 -l 3 -z 384 -i 1536 -x True -p ./l3_simd12_pe8.json
	mv ./intermediate_models ./l3_simd12_pe8
