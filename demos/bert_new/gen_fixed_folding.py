#!/usr/bin/env python3
"""Generate a corrected folding configuration for 1-layer BERT model."""

import json
import sys

def generate_fixed_folding_config():
    """Generate folding config that maintains compatible shapes between connected nodes."""
    config = {"Defaults": {}}
    
    # Based on the debug output, we need to ensure:
    # 1. Thresholding outputs match MVAU input requirements
    # 2. MVAU outputs match subsequent node requirements
    
    # First set of Thresholding -> MVAU connections
    # Thresholding needs PE that creates compatible output with MVAU SIMD=12
    config["Thresholding_rtl_0"] = {
        "PE": 12,  # Match MVAU SIMD
        "runtime_writeable_weights": 0,
        "depth_trigger_uram": 0,
        "depth_trigger_bram": 0
    }
    config["Thresholding_rtl_1"] = config["Thresholding_rtl_0"].copy()
    config["Thresholding_rtl_2"] = config["Thresholding_rtl_0"].copy()
    
    # MVAUs that receive from Thresholding
    config["MVAU_rtl_0"] = {
        "PE": 8,
        "SIMD": 12,
        "ram_style": "auto",
        "resType": "auto", 
        "mem_mode": "internal_decoupled",
        "runtime_writeable_weights": 0
    }
    config["MVAU_rtl_1"] = config["MVAU_rtl_0"].copy()
    config["MVAU_rtl_2"] = config["MVAU_rtl_0"].copy()
    
    # Shuffle nodes - need to match MVAU output shapes
    # MVAU outputs (1,128,48,8) but Shuffle expects different shape
    # Let's use SIMD that can handle the transformation
    config["Shuffle_hls_0"] = {"SIMD": 8}
    config["Shuffle_hls_1"] = {"SIMD": 8}
    config["Shuffle_hls_2"] = {"SIMD": 8}
    config["Shuffle_hls_3"] = {"SIMD": 8}
    
    # DuplicateStreams
    config["DuplicateStreams_hls_0"] = {"PE": 1}
    config["DuplicateStreams_hls_1"] = {"PE": 1}
    
    # More Thresholding nodes for attention
    config["Thresholding_rtl_3"] = {
        "PE": 8,  # Match DynMVU requirements
        "runtime_writeable_weights": 0,
        "depth_trigger_uram": 0,
        "depth_trigger_bram": 0
    }
    config["Thresholding_rtl_4"] = {"PE": 4, "runtime_writeable_weights": 0, "depth_trigger_uram": 0, "depth_trigger_bram": 0}
    config["Thresholding_rtl_5"] = {"PE": 8, "runtime_writeable_weights": 0, "depth_trigger_uram": 0, "depth_trigger_bram": 0}
    config["Thresholding_rtl_6"] = {"PE": 1, "runtime_writeable_weights": 0, "depth_trigger_uram": 0, "depth_trigger_bram": 0}
    config["Thresholding_rtl_7"] = {"PE": 4, "runtime_writeable_weights": 0, "depth_trigger_uram": 0, "depth_trigger_bram": 0}
    config["Thresholding_rtl_8"] = {"PE": 12, "runtime_writeable_weights": 0, "depth_trigger_uram": 0, "depth_trigger_bram": 0}
    config["Thresholding_rtl_9"] = {"PE": 24, "runtime_writeable_weights": 0, "depth_trigger_uram": 0, "depth_trigger_bram": 0}
    config["Thresholding_rtl_10"] = {"PE": 1, "runtime_writeable_weights": 0, "depth_trigger_uram": 0, "depth_trigger_bram": 0}
    
    # DynMVU nodes
    config["DynMVU_rtl_0"] = {
        "PE": 8,
        "SIMD": 4,
        "ram_style": "auto",
        "resType": "auto",
        "mem_mode": "external",
        "runtime_writeable_weights": 0
    }
    config["DynMVU_rtl_1"] = config["DynMVU_rtl_0"].copy()
    
    # Remaining MVAUs
    config["MVAU_rtl_3"] = {
        "PE": 8,
        "SIMD": 12,
        "ram_style": "auto",
        "resType": "auto",
        "mem_mode": "internal_decoupled",
        "runtime_writeable_weights": 0
    }
    config["MVAU_rtl_4"] = {
        "PE": 16,
        "SIMD": 24,
        "ram_style": "auto",
        "resType": "auto",
        "mem_mode": "internal_decoupled", 
        "runtime_writeable_weights": 0
    }
    config["MVAU_rtl_5"] = config["MVAU_rtl_4"].copy()
    
    # ElementwiseMul nodes
    for i in range(5):
        config[f"ElementwiseMul_hls_{i}"] = {"PE": 1, "ram_style": "auto"}
    
    # ElementwiseAdd nodes
    config["ElementwiseAdd_hls_0"] = {"PE": 1, "ram_style": "auto"}
    config["ElementwiseAdd_hls_1"] = {"PE": 1, "ram_style": "auto"}
    
    # Other nodes
    config["HWSoftmax_hls_0"] = {"SIMD": 1}
    config["LayerNorm_hls_0"] = {"SIMD": 1} 
    config["LayerNorm_hls_1"] = {"SIMD": 1}
    
    return config

if __name__ == "__main__":
    config = generate_fixed_folding_config()
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else "l1_fixed_folding.json"
    
    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Generated fixed folding configuration: {output_file}")
    print(f"Total nodes configured: {len(config) - 1}")  # -1 for Defaults