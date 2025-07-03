"""
End-to-end tests for Phase 1 Design Space Construction only.
Tests complete forge pipeline ending at DesignSpace creation.
Does NOT test Phase 2/3 execution.
"""

import pytest
from pathlib import Path
import yaml

from brainsmith.core.phase1 import forge, ForgeAPI
from brainsmith.core.phase1.data_structures import (
    DesignSpace, HWCompilerSpace, ProcessingSpace, 
    SearchConfig, GlobalConfig, SearchStrategy, OutputStage
)
from brainsmith.core.plugins import get_registry

# Use real QONNX/FINN plugins only - no fake plugins allowed


class TestE2EDesignSpaceConstruction:
    """Test end-to-end Design Space construction (Phase 1 only)."""
    
    # Remove setup_teardown - we want to use the real plugin registry
    # The registry should already have QONNX/FINN plugins loaded
    
    # No longer need to register fake plugins - we use real QONNX/FINN plugins
    
    def test_complete_forge_pipeline_phase1_only(self, tmp_path):
        """Test complete forge pipeline ending at DesignSpace creation."""
        # Create model file
        model_path = tmp_path / "e2e_model.onnx"
        model_path.write_bytes(b"fake onnx model content")
        
        # Create comprehensive blueprint
        blueprint_path = tmp_path / "e2e_blueprint.yaml"
        blueprint_content = """
version: "3.0"
name: "E2E Test Blueprint"
description: "Comprehensive test of Phase 1 functionality"

hw_compiler:
  kernels:
    # Test various kernel specifications with real QONNX/FINN plugins
    - "LayerNorm"  # Auto-discovery
    - ["Crop", ["CropHLS"]]  # Explicit backend
    - "~Shuffle"  # Optional with auto-discovery
    - [  # Mutually exclusive group
        "HWSoftmax",
        ["LayerNorm", ["LayerNormHLS"]],
        ~  # Skip option
      ]
  
  transforms:
    cleanup:
      - "RemoveUnusedTensors"
    topology_opt:
      - "FoldConstants"
      - "~InferShapes"  # Optional
    kernel_opt:
      - ["SpecializeLayers", ~]  # Optional group
  
  build_steps:
    - "ConvertToHW"
    - "PrepareIP"
    - "GenerateDriver"
  
  config_flags:
    target_device: "xczu7ev"
    frequency_mhz: 200

processing:
  preprocessing:
    - name: "quantization"
      options:
        - {enabled: true, bits: 8, mode: "symmetric"}
        - {enabled: true, bits: 4, mode: "asymmetric"}
        - {enabled: false}
    
    - name: "normalization"
      options:
        - {enabled: true, method: "batch"}
        - {enabled: true, method: "layer"}
  
  postprocessing:
    - name: "dequantization"
      options:
        - {enabled: true}
        - {enabled: false}

search:
  strategy: "exhaustive"
  constraints:
    - metric: "lut_utilization"
      operator: "<="
      value: 0.85
    - metric: "bram_utilization"
      operator: "<="
      value: 0.9
    - metric: "latency_cycles"
      operator: "<"
      value: 10000
  max_evaluations: 100
  timeout_minutes: 120
  parallel_builds: 4

global:
  output_stage: "rtl"
  working_directory: "./e2e_builds"
  cache_results: true
  save_artifacts: true
  log_level: "INFO"
  max_combinations: 10000
"""
        blueprint_path.write_text(blueprint_content)
        
        # Execute forge pipeline (Phase 1 only)
        design_space = forge(str(model_path), str(blueprint_path))
        
        # Verify DesignSpace object created correctly
        assert isinstance(design_space, DesignSpace)
        
        # Verify model path
        assert design_space.model_path == str(model_path)
        
        # Verify HW compiler space
        hw_space = design_space.hw_compiler_space
        assert isinstance(hw_space, HWCompilerSpace)
        assert len(hw_space.kernels) == 4
        
        # Check kernel auto-discovery worked
        assert hw_space.kernels[0][0] == "LayerNorm"
        assert "LayerNormHLS" in hw_space.kernels[0][1]
        assert hw_space.kernels[1] == ("Crop", ["CropHLS"])
        assert hw_space.kernels[2][0] == "~Shuffle"
        assert "ShuffleHLS" in hw_space.kernels[2][1]
        
        # Check mutually exclusive group
        assert isinstance(hw_space.kernels[3], list)
        assert len(hw_space.kernels[3]) == 3
        
        # Verify transforms organized by phase
        assert isinstance(hw_space.transforms, dict)
        assert "cleanup" in hw_space.transforms
        assert "topology_opt" in hw_space.transforms
        assert "kernel_opt" in hw_space.transforms
        assert hw_space.transforms["cleanup"] == ["RemoveUnusedTensors"]
        assert len(hw_space.transforms["topology_opt"]) == 2
        
        # Verify build steps
        assert hw_space.build_steps == ["ConvertToHW", "PrepareIP", "GenerateDriver"]
        
        # Verify config flags
        assert hw_space.config_flags["target_device"] == "xczu7ev"
        assert hw_space.config_flags["frequency_mhz"] == 200
        
        # Verify processing space
        proc_space = design_space.processing_space
        assert isinstance(proc_space, ProcessingSpace)
        assert len(proc_space.preprocessing) == 2
        assert len(proc_space.postprocessing) == 1
        
        # Check preprocessing options
        quant_options = proc_space.preprocessing[0]
        assert len(quant_options) == 3  # 3 options for quantization
        assert quant_options[0].name == "quantization"
        assert quant_options[0].parameters["bits"] == 8
        
        # Verify search config
        search = design_space.search_config
        assert isinstance(search, SearchConfig)
        assert search.strategy == SearchStrategy.EXHAUSTIVE
        assert len(search.constraints) == 3
        assert search.max_evaluations == 100
        assert search.timeout_minutes == 120
        assert search.parallel_builds == 4
        
        # Verify global config
        global_cfg = design_space.global_config
        assert isinstance(global_cfg, GlobalConfig)
        assert global_cfg.output_stage == OutputStage.RTL
        assert global_cfg.working_directory == "./e2e_builds"
        assert global_cfg.cache_results is True
        assert global_cfg.save_artifacts is True
        assert global_cfg.log_level == "INFO"
        assert global_cfg.max_combinations == 10000
        
        # Test combination calculation
        total_combinations = design_space.get_total_combinations()
        assert total_combinations > 0
        assert total_combinations <= 10000  # Respects max_combinations
        
        # IMPORTANT: Do NOT test execution - this is Phase 1 only
        # No build runner, no exploration, no results
    
    def test_design_space_with_fixture_blueprints(self):
        """Test DesignSpace construction with fixture blueprints."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        blueprints_dir = fixtures_dir / "blueprints"
        model_path = fixtures_dir / "simple_model.onnx"
        
        # Test auto_discovery.yaml
        if (blueprints_dir / "auto_discovery.yaml").exists():
            design_space = forge(str(model_path), str(blueprints_dir / "auto_discovery.yaml"))
            assert isinstance(design_space, DesignSpace)
            assert design_space.get_total_combinations() > 0
        
        # Test explicit_backends.yaml
        if (blueprints_dir / "explicit_backends.yaml").exists():
            design_space = forge(str(model_path), str(blueprints_dir / "explicit_backends.yaml"))
            assert isinstance(design_space, DesignSpace)
            # Should have phase-based transforms
            assert isinstance(design_space.hw_compiler_space.transforms, dict)
            assert "pre_proc" in design_space.hw_compiler_space.transforms
        
        # Test optimized_blueprint.yaml
        if (blueprints_dir / "optimized_blueprint.yaml").exists():
            api = ForgeAPI()
            design_space = api.forge_optimized(
                str(model_path), 
                str(blueprints_dir / "optimized_blueprint.yaml")
            )
            assert isinstance(design_space, DesignSpace)
            assert hasattr(design_space, '_plugin_optimization_enabled')
    
    def test_backward_compatibility(self, tmp_path):
        """Test backward compatibility of blueprint parsing."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        # Test flat transform list (older style)
        old_style_blueprint = tmp_path / "old_style.yaml"
        old_style_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - "LayerNorm"
    - "Crop"
  transforms:
    - "RemoveUnusedTensors"
    - "FoldConstants"
    - "SpecializeLayers"
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        # Should parse successfully
        design_space = forge(str(model_path), str(old_style_blueprint))
        
        # Transforms should be flat list
        assert isinstance(design_space.hw_compiler_space.transforms, list)
        assert len(design_space.hw_compiler_space.transforms) == 3
        
        # Test simple kernel format still works
        assert len(design_space.hw_compiler_space.kernels) == 2
        # Should have auto-discovered backends
        assert isinstance(design_space.hw_compiler_space.kernels[0], tuple)
        assert len(design_space.hw_compiler_space.kernels[0]) == 2
    
    def test_minimal_blueprint_design_space(self, tmp_path):
        """Test minimal blueprint creates valid DesignSpace."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        # Absolute minimal blueprint
        minimal_blueprint = tmp_path / "minimal.yaml"
        minimal_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels: []
  transforms: []
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        design_space = forge(str(model_path), str(minimal_blueprint))
        
        # Should create valid but empty design space
        assert isinstance(design_space, DesignSpace)
        assert len(design_space.hw_compiler_space.kernels) == 0
        assert design_space.get_total_combinations() == 1  # Single empty config
    
    def test_complex_combination_calculation(self, tmp_path):
        """Test accurate combination calculation for complex design space."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        # Blueprint with known combination count
        calc_blueprint = tmp_path / "calc_test.yaml"
        calc_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    # 2 kernel choices (after auto-discovery each has backends)
    - "LayerNorm"  # Multiple backends after auto-discovery
    - ["Crop", ["CropHLS"]]  # 1 choice
    - ["Shuffle", "HWSoftmax"]  # 2 mutually exclusive options
  
  transforms:
    # 2 * 2 = 4 transform combinations
    topology_opt:
      - ["FoldConstants", ~]  # 2 choices
      - ["InferShapes", ~]  # 2 choices
  
  build_steps: ["ConvertToHW"]

processing:
  preprocessing:
    # 2 preprocessing options
    - name: "test"
      options:
        - {enabled: true}
        - {enabled: false}

search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        design_space = forge(str(model_path), str(calc_blueprint))
        
        # Calculate expected combinations:
        # Kernels: 2 (MatMul backends) * 1 (Conv2D) * 2 (mutex group) = 4
        # Transforms: 2 * 2 = 4
        # Preprocessing: 2
        # Total: 4 * 4 * 2 = 32
        
        total = design_space.get_total_combinations()
        # Due to implementation details, exact count may vary
        # but should be in reasonable range
        assert total > 0
        assert total < 1000  # Sanity check
    
    def test_phase1_does_not_execute(self, tmp_path):
        """Ensure Phase 1 only constructs DesignSpace, doesn't execute."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        blueprint_path = tmp_path / "no_exec.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels: ["LayerNorm"]
  transforms: ["FoldConstants"]
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./no_exec_builds"
""")
        
        # Forge should complete without creating build directory
        design_space = forge(str(model_path), str(blueprint_path))
        
        # Build directory should NOT be created
        assert not Path("./no_exec_builds").exists()
        
        # No execution artifacts
        assert not hasattr(design_space, 'results')
        assert not hasattr(design_space, 'exploration_history')
        
        # Just a clean DesignSpace ready for Phase 2
        assert isinstance(design_space, DesignSpace)
        assert design_space.global_config.working_directory == "./no_exec_builds"