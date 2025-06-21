#!/usr/bin/env python3
"""Generate shape-compatible folding configuration for BERT."""

import json
import argparse

def generate_shape_compatible_folding(num_layers=1, output_file="shape_compatible_folding.json"):
    """Generate folding config with shape-compatible PE/SIMD values."""
    
    config = {"Defaults": {}}
    
    # Key insight: Use PE/SIMD values that create compatible tensor chunking
    # to avoid needing StreamingDataWidthConverter nodes
    
    for n in range(num_layers):
        # MVAUs - use standard folding
        for m in range(6):
            if m in [4, 5]:  # Larger FFN layers
                config[f"MVAU_rtl_{m + (6 * n)}"] = {
                    "PE": 16,
                    "SIMD": 24,
                    "ram_style": "auto",
                    "resType": "auto",
                    "mem_mode": "internal_decoupled",
                    "runtime_writeable_weights": 0
                }
            else:
                config[f"MVAU_rtl_{m + (6 * n)}"] = {
                    "PE": 8,
                    "SIMD": 12,
                    "ram_style": "auto",
                    "resType": "auto", 
                    "mem_mode": "internal_decoupled",
                    "runtime_writeable_weights": 0
                }
        
        # Thresholding - use PE that creates compatible chunking
        # For MVAU with SIMD=12, use Thresholding PE=12 or divisors
        for m in range(11):
            if m < 4:  # Thresholding feeding into MVAU with SIMD=12
                pe = 12  # Creates (1, 128, 32, 12) to match MVAU input
            elif m == 10:  # Thresholding for intermediate size
                pe = 24  # For 1536 features, matches MVAU SIMD=24
            else:
                pe = 4  # Conservative value for others
                
            config[f"Thresholding_rtl_{m + (11 * n)}"] = {
                "PE": pe,
                "runtime_writeable_weights": 0,
                "depth_trigger_uram": 0,
                "depth_trigger_bram": 0
            }
        
        # DuplicateStreams - match hidden size chunking
        for m in range(2):
            config[f"DuplicateStreams_hls_{m + (2 * n)}"] = {
                "PE": 12  # Match MVAU SIMD for compatibility
            }
        
        # Shuffles
        for m in range(4):
            config[f"Shuffle_hls_{m + (4 * n)}"] = {
                "SIMD": 4
            }
        
        # DynMVUs
        for m in range(2):
            config[f"DynMVU_rtl_{m + (2 * n)}"] = {
                "PE": 8,
                "SIMD": 4,  # 12/3 = 4
                "ram_style": "auto",
                "resType": "auto",
                "mem_mode": "external",
                "runtime_writeable_weights": 0
            }
        
        # ElementwiseAdd
        for m in range(2):
            config[f"ElementwiseAdd_hls_{m + (2 * n)}"] = {
                "PE": 12,  # Match common chunking
                "ram_style": "auto"
            }
        
        # ElementwiseMul
        for m in range(5):
            config[f"ElementwiseMul_hls_{m + (5 * n)}"] = {
                "PE": 12,  # Match common chunking
                "ram_style": "auto"
            }
        
        # Softmax
        config[f"HWSoftmax_hls_{n}"] = {
            "SIMD": 4
        }
        
        # LayerNorm
        for m in range(2):
            config[f"LayerNorm_hls_{m + (2 * n)}"] = {
                "SIMD": 12  # Match common chunking
            }
    
    # Write config
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated shape-compatible folding configuration: {output_file}")
    print("Key features:")
    print("  - Thresholding PE matches downstream MVAU SIMD")
    print("  - Creates compatible tensor chunking")
    print("  - Avoids need for StreamingDataWidthConverter")
    print("Folding values:")
    print("  - MVAU: SIMD=12, PE=8 (attention)")
    print("  - MVAU: SIMD=24, PE=16 (FFN)")
    print("  - Thresholding: PE=12 (attention), PE=24 (FFN)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate shape-compatible BERT folding')
    parser.add_argument('-n', '--num-layers', type=int, default=1, help='Number of BERT layers')
    parser.add_argument('-o', '--output', default='shape_compatible_folding.json', help='Output JSON file')
    args = parser.parse_args()
    
    generate_shape_compatible_folding(args.num_layers, args.output)