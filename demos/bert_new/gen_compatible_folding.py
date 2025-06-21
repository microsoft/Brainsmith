#!/usr/bin/env python3
"""Generate HLS-compatible folding configuration that matches old demo."""

import json
import argparse

def generate_compatible_folding(num_layers=1, output_file="compatible_folding.json"):
    """Generate folding config that matches working old demo configuration."""
    
    config = {"Defaults": {}}
    
    # Use exact values from working old demo config
    # These are proven to work and stay within HLS limits
    
    for n in range(num_layers):
        # MVAUs - matching old demo
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
        
        # Thresholding - use PE=1 like old demo
        for m in range(11):
            config[f"Thresholding_rtl_{m + (11 * n)}"] = {
                "PE": 1,
                "runtime_writeable_weights": 0,
                "depth_trigger_uram": 0,
                "depth_trigger_bram": 0
            }
        
        # DuplicateStreams
        for m in range(2):
            config[f"DuplicateStreams_hls_{m + (2 * n)}"] = {
                "PE": 1
            }
        
        # Shuffles
        for m in range(4):
            config[f"Shuffle_hls_{m + (4 * n)}"] = {
                "SIMD": 1
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
                "PE": 1,
                "ram_style": "auto"
            }
        
        # ElementwiseMul
        for m in range(5):
            config[f"ElementwiseMul_hls_{m + (5 * n)}"] = {
                "PE": 1,
                "ram_style": "auto"
            }
        
        # Softmax
        config[f"HWSoftmax_hls_{n}"] = {
            "SIMD": 1
        }
        
        # LayerNorm
        for m in range(2):
            config[f"LayerNorm_hls_{m + (2 * n)}"] = {
                "SIMD": 1
            }
    
    # Write config
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated HLS-compatible folding configuration: {output_file}")
    print("Using conservative values that match working old demo:")
    print("  MVAU: SIMD=12, PE=8 (standard layers)")
    print("  MVAU: SIMD=24, PE=16 (FFN layers)")  
    print("  All others: PE/SIMD=1")
    print("These values are proven to work and stay within HLS AP_INT limits.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate HLS-compatible BERT folding config')
    parser.add_argument('-n', '--num-layers', type=int, default=1, help='Number of BERT layers')
    parser.add_argument('-o', '--output', default='compatible_folding.json', help='Output JSON file')
    args = parser.parse_args()
    
    generate_compatible_folding(args.num_layers, args.output)