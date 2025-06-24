"""
Test YAML serialization/deserialization for Blueprint V2 structures.
"""

import pytest
import yaml
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Any

from brainsmith.core.blueprint import (
    DesignSpaceDefinition, NodeDesignSpace, TransformDesignSpace,
    ComponentSpace, ExplorationRules, DSEStrategy, DSEStrategies,
    Objective, Constraint, OptimizationDirection,
    load_blueprint, _is_blueprint_v2
)


class TestYAMLSerialization:
    """Test YAML serialization and deserialization."""
    
    def test_complete_blueprint_round_trip(self):
        """Test complete blueprint serialization round trip."""
        # Create complete YAML blueprint
        blueprint_yaml = """
name: "test_bert_accelerator"
version: "2.0"
base_blueprint: null

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
      - "MultiHeadAttention"
    exploration:
      required:
        - "LayerNorm"
        - "Softmax"
      optional:
        - "MultiHeadAttention"
      
  hw_kernels:
    available:
      - "MatMul":
          - "matmul_hls"
          - "matmul_rtl"
      - "Conv2D":
          - "conv2d_hls"
      - "Add"
    exploration:
      required:
        - "MatMul"
      optional:
        - "Conv2D"
        - "Add"
      mutually_exclusive:
        - ["matmul_hls", "matmul_rtl"]

transforms:
  model_topology:
    available:
      - "cleanup"
      - "streamlining"
      - "aggressive_streamlining"
      - "conservative_streamlining"
      - "constant_folding"
    exploration:
      required:
        - "cleanup"
      optional:
        - "constant_folding"
      mutually_exclusive:
        - ["aggressive_streamlining", "conservative_streamlining"]
      dependencies:
        streamlining: ["cleanup"]
  
  hw_kernel:
    available:
      - "target_fps_parallelization"
      - "apply_folding_config"
      - "minimize_bit_width"
    exploration:
      required:
        - "target_fps_parallelization"
        - "apply_folding_config"
      optional:
        - "minimize_bit_width"
  
  hw_graph:
    available:
      - "set_fifo_depths"
      - "create_stitched_ip"
    exploration:
      required:
        - "set_fifo_depths"
        - "create_stitched_ip"

configuration_files:
  folding_override: "configs/bert_folding.json"
  platform_config: "configs/zynq_ultrascale.yaml"

dse_strategies:
  primary_strategy: "hierarchical_exploration"
  strategies:
    quick_scan:
      description: "Fast exploration of main design choices"
      max_evaluations: 20
      sampling: "latin_hypercube"
      focus_areas:
        - "kernel_selection"
    
    hierarchical_exploration:
      description: "Hierarchical exploration strategy"
      max_evaluations: 100
      sampling: "adaptive"
      focus_areas:
        - "all_combinations"

objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.0
    target_value: 3000.0
  - name: "resource_efficiency"
    direction: "maximize"
    weight: 0.7

constraints:
  - name: "max_lut_utilization"
    operator: "<="
    value: 0.85
    description: "Maximum LUT utilization"
  - name: "max_power"
    operator: "<="
    value: 25.0
    description: "Maximum power consumption in watts"
"""
        
        # Write to temporary file and load
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(blueprint_yaml)
            f.flush()
            
            # Load blueprint
            blueprint = load_blueprint(f.name)
            
            # Verify all sections loaded correctly
            assert blueprint.name == "test_bert_accelerator"
            assert blueprint.version == "2.0"
            
            # Verify nodes
            assert "LayerNorm" in blueprint.nodes.canonical_ops.get_component_names()
            assert "MatMul" in blueprint.nodes.hw_kernels.get_component_names()
            assert blueprint.nodes.canonical_ops.exploration.required == ["LayerNorm", "Softmax"]
            
            # Verify transforms
            assert "cleanup" in blueprint.transforms.model_topology.get_component_names()
            assert blueprint.transforms.model_topology.exploration.dependencies == {"streamlining": ["cleanup"]}
            
            # Verify configuration files
            assert blueprint.configuration_files["folding_override"] == "configs/bert_folding.json"
            
            # Verify DSE strategies
            assert blueprint.dse_strategies.primary_strategy == "hierarchical_exploration"
            assert "quick_scan" in blueprint.dse_strategies.strategies
            assert blueprint.dse_strategies.strategies["quick_scan"].max_evaluations == 20
            
            # Verify objectives
            assert len(blueprint.objectives) == 2
            throughput_obj = next(obj for obj in blueprint.objectives if obj.name == "throughput")
            assert throughput_obj.direction == OptimizationDirection.MAXIMIZE
            assert throughput_obj.target_value == 3000.0
            
            # Verify constraints
            assert len(blueprint.constraints) == 2
            lut_constraint = next(c for c in blueprint.constraints if c.name == "max_lut_utilization")
            assert lut_constraint.operator == "<="
            assert lut_constraint.value == 0.85
        
        # Clean up
        Path(f.name).unlink()
    
    def test_blueprint_inheritance(self):
        """Test blueprint inheritance functionality."""
        # Create base blueprint
        base_yaml = """
name: "transformer_base"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
    exploration:
      required:
        - "LayerNorm"

transforms:
  model_topology:
    available:
      - "cleanup"
      - "streamlining"
    exploration:
      required:
        - "cleanup"

objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.0
"""
        
        # Create derived blueprint
        derived_yaml = """
name: "bert_accelerator"
version: "2.0"
base_blueprint: "transformer_base"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
      - "MultiHeadAttention"  # Additional component
    exploration:
      required:
        - "LayerNorm"
        - "Softmax"  # Additional requirement
      optional:
        - "MultiHeadAttention"

  hw_kernels:  # New section
    available:
      - "MatMul":
          - "matmul_hls"
    exploration:
      required:
        - "MatMul"

configuration_files:
  folding_override: "configs/bert_folding.json"
"""
        
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write base blueprint
            base_file = temp_path / "transformer_base.yaml"
            with open(base_file, 'w') as f:
                f.write(base_yaml)
            
            # Write derived blueprint
            derived_file = temp_path / "bert_accelerator.yaml"
            with open(derived_file, 'w') as f:
                f.write(derived_yaml)
            
            # Load derived blueprint (should inherit from base)
            blueprint = load_blueprint(str(derived_file))
            
            # Verify inheritance worked
            assert blueprint.name == "bert_accelerator"  # Overridden
            assert "MultiHeadAttention" in blueprint.nodes.canonical_ops.get_component_names()  # Added
            assert "MatMul" in blueprint.nodes.hw_kernels.get_component_names()  # New section
            assert "cleanup" in blueprint.transforms.model_topology.get_component_names()  # Inherited
            assert len(blueprint.objectives) == 1  # Inherited
            assert blueprint.configuration_files["folding_override"] == "configs/bert_folding.json"  # Added
    
    def test_minimal_blueprint_loading(self):
        """Test loading minimal valid blueprint."""
        minimal_yaml = """
name: "minimal_test"
version: "2.0"

nodes:
  canonical_ops:
    available: []

transforms:
  model_topology:
    available: []
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(minimal_yaml)
            f.flush()
            
            blueprint = load_blueprint(f.name)
            
            assert blueprint.name == "minimal_test"
            assert blueprint.version == "2.0"
            assert len(blueprint.nodes.canonical_ops.available) == 0
            assert len(blueprint.transforms.model_topology.available) == 0
            
            # Should still validate
            is_valid, errors = blueprint.validate()
            assert is_valid, f"Minimal blueprint should be valid: {errors}"
        
        # Clean up
        Path(f.name).unlink()
    
    def test_blueprint_v2_detection_edge_cases(self):
        """Test edge cases for V2 blueprint detection."""
        # Test empty file
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()
            assert not _is_blueprint_v2(f.name)
        Path(f.name).unlink()
        
        # Test invalid YAML
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            assert not _is_blueprint_v2(f.name)
        Path(f.name).unlink()
        
        # Test non-dict YAML
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("- list\n- instead\n- of\n- dict")
            f.flush()
            assert not _is_blueprint_v2(f.name)
        Path(f.name).unlink()
        
        # Test has nodes but no transforms
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("nodes:\n  canonical_ops: []\nother_section: {}")
            f.flush()
            assert not _is_blueprint_v2(f.name)
        Path(f.name).unlink()
        
        # Test has transforms but no nodes
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("transforms:\n  model_topology: []\nother_section: {}")
            f.flush()
            assert not _is_blueprint_v2(f.name)
        Path(f.name).unlink()
        
        # Test valid V2 blueprint
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("nodes:\n  canonical_ops: []\ntransforms:\n  model_topology: []")
            f.flush()
            assert _is_blueprint_v2(f.name)
        Path(f.name).unlink()
    
    def test_error_handling_in_loading(self):
        """Test error handling during blueprint loading."""
        # Test file not found
        with pytest.raises(FileNotFoundError):
            load_blueprint("non_existent_file.yaml")
        
        # Test invalid YAML syntax
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [unclosed")
            f.flush()
            
            with pytest.raises(yaml.YAMLError):
                load_blueprint(f.name)
        Path(f.name).unlink()
        
        # Test non-dict blueprint
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("- this\n- is\n- a\n- list")
            f.flush()
            
            with pytest.raises(ValueError, match="Blueprint must be a YAML dictionary"):
                load_blueprint(f.name)
        Path(f.name).unlink()
        
        # Test validation errors
        invalid_blueprint = """
name: "invalid_test"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
    exploration:
      required:
        - "NonExistentComponent"  # References component not in available

transforms:
  model_topology:
    available: []
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_blueprint)
            f.flush()
            
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint(f.name)
        Path(f.name).unlink()
    
    def test_complex_component_options_serialization(self):
        """Test serialization of complex component options."""
        complex_yaml = """
name: "complex_components_test"
version: "2.0"

nodes:
  hw_kernels:
    available:
      - "SimpleKernel"  # String component
      - "MatMul":        # Dict component with options
          - "matmul_hls"
          - "matmul_rtl" 
          - "matmul_mixed"
      - "Conv2D":        # Dict component with single option
          - "conv2d_hls"
      - "AnotherSimple"  # Another string component
    exploration:
      required:
        - "MatMul"
      optional:
        - "SimpleKernel"
        - "Conv2D"
        - "AnotherSimple"
      mutually_exclusive:
        - ["matmul_hls", "matmul_rtl"]

transforms:
  model_topology:
    available: ["cleanup"]
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(complex_yaml)
            f.flush()
            
            blueprint = load_blueprint(f.name)
            
            # Verify component parsing
            hw_kernels = blueprint.nodes.hw_kernels
            component_names = hw_kernels.get_component_names()
            
            assert "SimpleKernel" in component_names
            assert "MatMul" in component_names
            assert "Conv2D" in component_names
            assert "AnotherSimple" in component_names
            
            # Verify options
            matmul_options = hw_kernels.get_component_options("MatMul")
            assert matmul_options == ["matmul_hls", "matmul_rtl", "matmul_mixed"]
            
            conv_options = hw_kernels.get_component_options("Conv2D")
            assert conv_options == ["conv2d_hls"]
            
            simple_options = hw_kernels.get_component_options("SimpleKernel")
            assert simple_options == ["SimpleKernel"]  # String components return themselves
            
            # Verify exploration rules
            assert hw_kernels.exploration.required == ["MatMul"]
            assert set(hw_kernels.exploration.optional) == {"SimpleKernel", "Conv2D", "AnotherSimple"}
            assert hw_kernels.exploration.mutually_exclusive == [["matmul_hls", "matmul_rtl"]]
        
        # Clean up
        Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__])