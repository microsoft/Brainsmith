#!/usr/bin/env python3
"""Generate a safe folding configuration for BERT that avoids shape mismatches."""

import json
import sys

def generate_safe_folding(num_layers=1, output_file="safe_folding.json"):
    """Generate conservative folding config that avoids zero-dimension issues."""
    
    config = {"Defaults": {}}
    
    # Critical: Set minimums that avoid zero dimensions
    MIN_PE = 4  # Minimum PE to avoid dimension 0 issues
    MIN_SIMD = 4  # Minimum SIMD
    
    # Conservative values for 1-layer BERT
    MVAU_SIMD = 12
    MVAU_PE = 8
    OTHER_PE = 4  # For thresholding, duplicatestreams, etc.
    
    for n in range(num_layers):
        # MVAUs - 6 per layer
        for m in range(6):
            if m in [4, 5]:  # Larger layers
                config[f"MVAU_rtl_{m + (6 * n)}"] = {
                    "PE": MVAU_PE * 2,
                    "SIMD": MVAU_SIMD * 2,
                    "ram_style": "auto",
                    "resType": "auto",
                    "mem_mode": "internal_decoupled",
                    "runtime_writeable_weights": 0
                }
            else:
                config[f"MVAU_rtl_{m + (6 * n)}"] = {
                    "PE": MVAU_PE,
                    "SIMD": MVAU_SIMD,
                    "ram_style": "auto",
                    "resType": "auto", 
                    "mem_mode": "internal_decoupled",
                    "runtime_writeable_weights": 0
                }
        
        # Thresholding - CRITICAL: Must have PE >= MIN_PE
        for m in range(11):  # Adjusted for actual count
            config[f"Thresholding_rtl_{m + (11 * n)}"] = {
                "PE": max(OTHER_PE, MIN_PE),  # Ensure minimum PE
                "runtime_writeable_weights": 0,
                "depth_trigger_uram": 0,
                "depth_trigger_bram": 0
            }
        
        # DuplicateStreams
        for m in range(2):  # Adjusted for actual count
            config[f"DuplicateStreams_hls_{m + (2 * n)}"] = {
                "PE": max(OTHER_PE, MIN_PE)
            }
        
        # Shuffles
        for m in range(4):
            config[f"Shuffle_hls_{m + (4 * n)}"] = {
                "SIMD": OTHER_PE
            }
        
        # DynMVUs
        for m in range(2):
            config[f"DynMVU_rtl_{m + (2 * n)}"] = {
                "PE": MVAU_PE,
                "SIMD": MVAU_SIMD // 3 if MVAU_SIMD % 3 == 0 else MVAU_SIMD // 4,
                "ram_style": "auto",
                "resType": "auto",
                "mem_mode": "external",
                "runtime_writeable_weights": 0
            }
        
        # ElementwiseAdd
        for m in range(2):
            config[f"ElementwiseAdd_hls_{m + (2 * n)}"] = {
                "PE": max(OTHER_PE, MIN_PE),
                "ram_style": "auto"
            }
        
        # ElementwiseMul
        for m in range(5):
            config[f"ElementwiseMul_hls_{m + (5 * n)}"] = {
                "PE": OTHER_PE,
                "ram_style": "auto"
            }
        
        # Softmax
        config[f"HWSoftmax_hls_{n}"] = {
            "SIMD": OTHER_PE
        }
        
        # LayerNorm
        for m in range(2):
            config[f"LayerNorm_hls_{m + (2 * n)}"] = {
                "SIMD": max(OTHER_PE, MIN_PE)
            }
    
    # Don't override specific nodes - let them use default conservative values
    # The AP_INT_MAX_W limit means we need to keep PE values reasonable
    
    # Write config
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated safe folding configuration: {output_file}")
    print(f"Configuration ensures minimum PE={MIN_PE} to avoid zero dimensions")
    print(f"MVAU: SIMD={MVAU_SIMD}, PE={MVAU_PE}")
    print(f"Others: PE={OTHER_PE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate safe BERT folding config')
    parser.add_argument('-n', '--num-layers', type=int, default=1, help='Number of BERT layers')
    parser.add_argument('-o', '--output', default='safe_folding.json', help='Output JSON file')
    args = parser.parse_args()
    
    generate_safe_folding(args.num_layers, args.output)