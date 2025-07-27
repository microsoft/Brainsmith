#!/usr/bin/env python3
"""Test simplified config extraction."""

import sys
import tempfile
import os
sys.path.insert(0, '/home/tafk/dev/brainsmith-4')

from brainsmith.core.blueprint_parser import BlueprintParser
from brainsmith.core.design_space import OutputStage

# Test different config paths
test_cases = [
    # Test 1: Config in global_config
    {
        "name": "Config in global_config",
        "yaml": """
version: "4.0"
global_config:
  output_stage: "generate_reports"
  max_combinations: 5000
  working_directory: "test_work"
finn_config:
  board: "Pynq-Z1"
  synth_clk_period_ns: 5.0
design_space:
  steps:
    - qonnx_to_finn
"""
    },
    # Test 2: Config at top level
    {
        "name": "Config at top level",
        "yaml": """
version: "4.0"
output_stage: "generate_reports"
max_combinations: 5000
working_directory: "test_work"
finn_config:
  board: "Pynq-Z1"
  synth_clk_period_ns: 5.0
design_space:
  steps:
    - qonnx_to_finn
"""
    },
    # Test 3: Legacy params (platform/target_clk)
    {
        "name": "Legacy params",
        "yaml": """
version: "4.0"
platform: "Pynq-Z1"
target_clk: "5ns"
design_space:
  steps:
    - qonnx_to_finn
"""
    },
]

with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "model.onnx")
    with open(model_path, "wb") as f:
        f.write(b"dummy")
    
    parser = BlueprintParser()
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        print("-" * 40)
        
        blueprint_path = os.path.join(tmpdir, "test.yaml")
        with open(blueprint_path, "w") as f:
            f.write(test['yaml'])
        
        try:
            # Parse blueprint
            design_space, _ = parser.parse(blueprint_path, model_path)
            config = design_space.config
            
            # Check results
            print(f"✓ Parsing successful")
            print(f"  output_stage: {config.output_stage.value}")
            print(f"  max_combinations: {config.max_combinations}")
            print(f"  working_directory: {config.working_directory}")
            print(f"  finn_params: {config.finn_params}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")

print("\nDay 5-6: Config extraction simplified!")