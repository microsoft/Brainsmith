#!/usr/bin/env python3
"""Test with minimal folding configuration to isolate the issue."""

import json

# Create a minimal folding config with PE=1, SIMD=1 for all nodes
# This should work but be very slow
minimal_config = {
    "Defaults": {}
}

# Set all nodes to minimal parallelization
node_types = [
    ("MVAU_rtl", 6, {"PE": 1, "SIMD": 1, "ram_style": "auto", "resType": "auto", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0}),
    ("Thresholding_rtl", 11, {"PE": 1, "runtime_writeable_weights": 0, "depth_trigger_uram": 0, "depth_trigger_bram": 0}),
    ("DynMVU_rtl", 2, {"PE": 1, "SIMD": 1, "ram_style": "auto", "resType": "auto", "mem_mode": "external", "runtime_writeable_weights": 0}),
    ("Shuffle_hls", 4, {"SIMD": 1}),
    ("DuplicateStreams_hls", 2, {"PE": 1}),
    ("ElementwiseAdd_hls", 2, {"PE": 1, "ram_style": "auto"}),
    ("ElementwiseMul_hls", 5, {"PE": 1, "ram_style": "auto"}),
    ("HWSoftmax_hls", 1, {"SIMD": 1}),
    ("LayerNorm_hls", 2, {"SIMD": 1})
]

for node_type, count, params in node_types:
    for i in range(count):
        minimal_config[f"{node_type}_{i}"] = params.copy()

# Save the minimal config
with open("minimal_folding.json", "w") as f:
    json.dump(minimal_config, f, indent=4)

print("Created minimal_folding.json with all nodes set to PE=1, SIMD=1")
print("This should work but will be very slow")