#!/usr/bin/env python3
"""Test ForgeConfig integration."""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from brainsmith.core.blueprint_parser import BlueprintParser
from brainsmith.core.design_space import ForgeConfig, OutputStage

def test_forge_config_parsing():
    """Test that ForgeConfig is properly parsed from blueprints."""
    
    # Create test blueprint with various configs
    blueprint_path = Path(__file__).parent / "test_config.yaml"
    blueprint_path.write_text("""
design_space:
  steps:
    - qonnx_to_finn
    - streamline
    
global_config:
  output_stage: synthesize_bitstream
  working_directory: custom_work
  save_intermediate_models: true
  fail_fast: true
  max_combinations: 5000
  timeout_minutes: 30
  
finn_config:
  board: U250
  synth_clk_period_ns: 5.0
  target_fps: 2000
  shell_flow_type: vivado_zynq
  generate_outputs:
    - "report"
    - "rtlsim_performance"
  custom_finn_param: "test_value"
""")
    
    try:
        parser = BlueprintParser()
        design_space, tree = parser.parse(str(blueprint_path), "dummy.onnx")
        
        # Check ForgeConfig fields
        config = design_space.config
        assert isinstance(config, ForgeConfig)
        assert config.output_stage == OutputStage.SYNTHESIZE_BITSTREAM
        assert config.working_directory == "custom_work"
        assert config.save_intermediate_models == True
        assert config.fail_fast == True
        assert config.max_combinations == 5000
        assert config.timeout_minutes == 30
        
        # Check FINN params
        assert config.finn_params["board"] == "U250"
        assert config.finn_params["synth_clk_period_ns"] == 5.0
        assert config.finn_params["target_fps"] == 2000
        assert config.finn_params["shell_flow_type"] == "vivado_zynq"
        assert config.finn_params["generate_outputs"] == ["report", "rtlsim_performance"]
        assert config.finn_params["custom_finn_param"] == "test_value"
        
        print("✅ ForgeConfig parsing test passed!")
        
        # Test top-level params (backward compatibility)
        blueprint_path.write_text("""
design_space:
  steps:
    - qonnx_to_finn
    
# Top-level params (old style)
output_stage: generate_reports
platform: Pynq-Z1
target_clk: 10ns
""")
        
        design_space, tree = parser.parse(str(blueprint_path), "dummy.onnx")
        config = design_space.config
        
        assert config.output_stage == OutputStage.GENERATE_REPORTS
        assert config.finn_params["board"] == "Pynq-Z1"
        assert config.finn_params["synth_clk_period_ns"] == 10.0
        
        print("✅ Top-level params test passed!")
        
    finally:
        if blueprint_path.exists():
            blueprint_path.unlink()


if __name__ == "__main__":
    test_forge_config_parsing()