#!/usr/bin/env python3
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""
Generate folding configurations for BERT model DSE.

This script generates JSON folding configurations that specify parallelism
parameters (PE/SIMD) for different hardware layers in the BERT model.
It maintains exact compatibility with the old gen_initial_folding.py format.
"""

import argparse
import json
from pathlib import Path


def mvau(simd: int, pe: int, runtime_writeable: int) -> dict:
    """Generate MVAU (Matrix-Vector Activation Unit) configuration."""
    return {
        "PE": pe,
        "SIMD": simd,
        "ram_style": "auto",
        "resType": "auto",
        "mem_mode": "internal_decoupled",
        "runtime_writeable_weights": runtime_writeable
    }


def dupstreams(pe: int) -> dict:
    """Generate DuplicateStreams configuration."""
    return {
        "PE": pe
    }


def shuffle(simd: int) -> dict:
    """Generate Shuffle configuration."""
    return {
        "SIMD": simd
    }


def thresholding(pe: int, runtime_writeable: int) -> dict:
    """Generate Thresholding configuration."""
    return {
        "PE": pe,
        "runtime_writeable_weights": runtime_writeable,
        "depth_trigger_uram": 0,
        "depth_trigger_bram": 0
    }


def dynmvu(pe: int, simd: int) -> dict:
    """Generate DynMVU (Dynamic Matrix-Vector Unit) configuration."""
    return {
        "PE": pe,
        "SIMD": simd,
        "ram_style": "auto",
        "resType": "auto",
        "mem_mode": "external",
        "runtime_writeable_weights": 0
    }


def eltwiseadd(pe: int) -> dict:
    """Generate ElementwiseAdd configuration."""
    return {
        "PE": pe,
        "ram_style": "auto"
    }


def eltwisemul(pe: int) -> dict:
    """Generate ElementwiseMul configuration."""
    return {
        "PE": pe,
        "ram_style": "auto"
    }


def softmax(simd: int) -> dict:
    """Generate HWSoftmax configuration."""
    return {
        'SIMD': simd
    }


def layernorm(simd: int) -> dict:
    """Generate LayerNorm configuration."""
    return {
        'SIMD': simd
    }


def generate_config(args) -> dict:
    """Generate complete folding configuration for BERT model."""
    config = {}
    
    # Add defaults section (empty in original)
    config["Defaults"] = {}
    
    # Generate configuration for each layer
    for n in range(args.num_layers):
        # Generate all MVAUs
        for m in range(0, 8):
            if m == 7 or m == 8:
                d = mvau(2 * args.simd, 2 * args.pe, args.runtime_writeable_weights)
            # dyn mvau
            elif m == 3 or m == 4:
                if args.simd % 3 == 0:
                    d = dynmvu(args.pe, int(args.simd/3))
                elif args.simd % 4 == 0:
                    d = dynmvu(args.pe, int(args.simd/4))
                else:
                    d = dynmvu(args.pe, args.simd)
            else:
                d = mvau(args.simd, args.pe, args.runtime_writeable_weights)
            config[f"MVAU_rtl_{m + (8 * n)}"] = d
        
        # DuplicateStreams - 3 per layer
        for m in range(3):
            d = dupstreams(args.other)
            config[f"DuplicateStreams_hls_{m + (3 * n)}"] = d
        
        # Shuffles - 4 per layer
        for m in range(4):
            d = shuffle(args.other)
            config[f"Shuffle_hls_{m + (4 * n)}"] = d
        
        # Thresholding - 9 per layer
        for m in range(9):
            d = thresholding(args.other, 0)
            config[f"Thresholding_rtl_{m + (9 * n)}"] = d
        
        # ElementwiseAdds - 2 per layer
        for m in range(2):
            d = eltwiseadd(args.other)
            config[f"ElementwiseAdd_hls_{m + (2 * n)}"] = d
        
        # ElementwiseMuls - 5 per layer
        for m in range(5):
            d = eltwisemul(args.other)
            config[f"ElementwiseMul_hls_{m + (5 * n)}"] = d
        
        # SoftMax - 1 per layer
        for m in range(1):
            d = softmax(args.other)
            config[f"HWSoftmax_hls_{m + (n * 1)}"] = d
        
        # LayerNorms - 2 per layer
        for m in range(2):
            d = layernorm(args.other)
            config[f"LayerNorm_hls_{m + (n * 2)}"] = d
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Generate folding configurations for BERT model'
    )
    
    # Output configuration
    parser.add_argument('-o', '--output', 
                       help='Output JSON config file path',
                       default='config.json')
    
    # MVAU configuration
    parser.add_argument('-s', '--simd', type=int, default=48,
                       help='SIMD setting for MVAU layers')
    parser.add_argument('-p', '--pe', type=int, default=32,
                       help='PE setting for MVAU layers')
    
    # Other operators configuration
    parser.add_argument('-t', '--other', type=int, default=4,
                       help='SIMD/PE for other operators between MVAUs')
    
    # Model configuration
    parser.add_argument('-n', '--num_layers', type=int, default=3,
                       help='Number of BERT hidden layers')
    
    # Runtime configuration
    parser.add_argument('-w', '--runtime_writeable_weights', type=int, default=0,
                       help='Make MVAU weights runtime writeable (0 or 1)')
    
    # Unused in modern version but kept for compatibility
    parser.add_argument('-f', '--shuffleb', type=bool, default=False,
                       help='(Deprecated) ShuffleB parallelization flag')
    
    args = parser.parse_args()
    
    # Generate configuration
    config = generate_config(args)
    
    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as fp:
        json.dump(config, fp, indent=4)
    
    print(f"Folding configuration generated: {output_path}")
    print(f"  Layers: {args.num_layers}")
    print(f"  MVAU SIMD: {args.simd}, PE: {args.pe}")
    print(f"  Other operators SIMD/PE: {args.other}")
    print(f"  Total nodes configured: {len(config) - 1}")  # -1 for Defaults


if __name__ == "__main__":
    main()