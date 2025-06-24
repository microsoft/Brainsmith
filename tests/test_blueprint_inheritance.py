"""
Comprehensive tests for Blueprint V2 inheritance system.
"""

import pytest
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any

from brainsmith.core.blueprint import load_blueprint, DesignSpaceDefinition
from brainsmith.core.blueprint_inheritance import (
    merge_blueprints, resolve_blueprint_path, validate_inheritance_chain,
    BlueprintInheritanceError, _merge_available_components, _merge_exploration_rules
)


class TestBlueprintInheritance:
    """Test complete blueprint inheritance functionality."""
    
    def test_simple_inheritance(self):
        """Test basic inheritance with simple override."""
        base_yaml = """
name: "base_accelerator"
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
        
        derived_yaml = """
name: "bert_accelerator"
version: "2.0"
base_blueprint: "base_accelerator"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
      - "MultiHeadAttention"  # Extended
    exploration:
      required:
        - "LayerNorm"
        - "Softmax"  # Added requirement
      optional:
        - "MultiHeadAttention"

objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.5  # Override weight
  - name: "power"  # New objective
    direction: "minimize"
    weight: 0.8
"""
        
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write base blueprint
            base_file = temp_path / "base_accelerator.yaml"
            with open(base_file, 'w') as f:
                f.write(base_yaml)
            
            # Write derived blueprint
            derived_file = temp_path / "bert_accelerator.yaml"
            with open(derived_file, 'w') as f:
                f.write(derived_yaml)
            
            # Load derived blueprint
            blueprint = load_blueprint(str(derived_file))
            
            # Verify inheritance results
            assert blueprint.name == "bert_accelerator"  # Overridden
            
            # Verify canonical ops were extended
            canonical_ops = blueprint.nodes.canonical_ops.get_component_names()
            assert "LayerNorm" in canonical_ops
            assert "Softmax" in canonical_ops  
            assert "MultiHeadAttention" in canonical_ops
            
            # Verify exploration rules were merged
            assert "LayerNorm" in blueprint.nodes.canonical_ops.exploration.required
            assert "Softmax" in blueprint.nodes.canonical_ops.exploration.required
            assert "MultiHeadAttention" in blueprint.nodes.canonical_ops.exploration.optional
            
            # Verify transforms were inherited
            topology_components = blueprint.transforms.model_topology.get_component_names()
            assert "cleanup" in topology_components
            assert "streamlining" in topology_components
            assert "cleanup" in blueprint.transforms.model_topology.exploration.required
            
            # Verify objectives were merged properly
            assert len(blueprint.objectives) == 2
            throughput_obj = next(obj for obj in blueprint.objectives if obj.name == "throughput")
            power_obj = next(obj for obj in blueprint.objectives if obj.name == "power")
            
            assert throughput_obj.weight == 1.5  # Overridden
            assert power_obj.weight == 0.8  # New
    
    def test_multi_level_inheritance(self):
        """Test inheritance chain: grandparent -> parent -> child."""
        grandparent_yaml = """
name: "base_transformer"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
    exploration:
      required:
        - "LayerNorm"

transforms:
  model_topology:
    available:
      - "cleanup"
    exploration:
      required:
        - "cleanup"
"""
        
        parent_yaml = """
name: "transformer_base"  
version: "2.0"
base_blueprint: "base_transformer"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"  # Extended
    exploration:
      required:
        - "LayerNorm"
        - "Softmax"  # Added requirement

  hw_kernels:  # New section
    available:
      - "MatMul":
          - "matmul_hls"
    exploration:
      required:
        - "MatMul"
"""
        
        child_yaml = """
name: "bert_accelerator"
version: "2.0"
base_blueprint: "transformer_base"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
      - "MultiHeadAttention"  # Further extended
    exploration:
      optional:
        - "MultiHeadAttention"  # New optional

  hw_kernels:
    available:
      - "MatMul":
          - "matmul_hls"
          - "matmul_rtl"  # Extended options
      - "Conv2D":  # New component
          - "conv2d_hls"
    exploration:
      required:
        - "MatMul"
      optional:
        - "Conv2D"

configuration_files:
  folding_override: "configs/bert_folding.json"
"""
        
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write all blueprints
            for name, content in [
                ("base_transformer.yaml", grandparent_yaml),
                ("transformer_base.yaml", parent_yaml),
                ("bert_accelerator.yaml", child_yaml)
            ]:
                with open(temp_path / name, 'w') as f:
                    f.write(content)
            
            # Load final child blueprint
            blueprint = load_blueprint(str(temp_path / "bert_accelerator.yaml"))
            
            # Verify all levels were inherited
            assert blueprint.name == "bert_accelerator"
            
            # Verify canonical ops from all levels
            canonical_ops = blueprint.nodes.canonical_ops.get_component_names()
            assert "LayerNorm" in canonical_ops  # From grandparent
            assert "Softmax" in canonical_ops  # From parent
            assert "MultiHeadAttention" in canonical_ops  # From child
            
            # Verify hw kernels
            hw_kernels = blueprint.nodes.hw_kernels.get_component_names()
            assert "MatMul" in hw_kernels  # From parent
            assert "Conv2D" in hw_kernels  # From child
            
            # Verify MatMul options were extended
            matmul_options = blueprint.nodes.hw_kernels.get_component_options("MatMul")
            assert "matmul_hls" in matmul_options  # From parent
            assert "matmul_rtl" in matmul_options  # From child
            
            # Verify exploration rules from all levels
            assert "LayerNorm" in blueprint.nodes.canonical_ops.exploration.required  # Grandparent
            assert "Softmax" in blueprint.nodes.canonical_ops.exploration.required  # Parent
            assert "MultiHeadAttention" in blueprint.nodes.canonical_ops.exploration.optional  # Child
            
            # Verify transforms inherited from grandparent
            assert "cleanup" in blueprint.transforms.model_topology.get_component_names()
            
            # Verify configuration files from child
            assert blueprint.configuration_files["folding_override"] == "configs/bert_folding.json"
    
    def test_component_merging_rules(self):
        """Test specific component merging behaviors."""
        # Test available component merging
        base_available = ["cleanup", {"MatMul": ["hls"]}]
        derived_available = ["streamlining", {"MatMul": ["rtl", "mixed"]}, {"Conv2D": ["hls"]}]
        
        merged = _merge_available_components(base_available, derived_available)
        
        # Verify string components
        assert "cleanup" in merged
        assert "streamlining" in merged
        
        # Verify dict components - MatMul should be overridden, Conv2D added
        matmul_found = False
        conv2d_found = False
        for item in merged:
            if isinstance(item, dict):
                if "MatMul" in item:
                    assert item["MatMul"] == ["rtl", "mixed"]  # Overridden
                    matmul_found = True
                elif "Conv2D" in item:
                    assert item["Conv2D"] == ["hls"]  # Added
                    conv2d_found = True
        
        assert matmul_found and conv2d_found
    
    def test_exploration_rules_merging(self):
        """Test exploration rules merging logic."""
        base_rules = {
            'required': ['LayerNorm'],
            'optional': ['Softmax'],
            'mutually_exclusive': [['opt1', 'opt2']],
            'dependencies': {'Softmax': ['LayerNorm']}
        }
        
        derived_rules = {
            'required': ['Softmax', 'MultiHead'],  # Extends required
            'optional': ['GELU'],  # Extends optional
            'mutually_exclusive': [['opt3', 'opt4']],  # Adds new group
            'dependencies': {'MultiHead': ['LayerNorm'], 'GELU': ['Softmax']}  # Adds dependencies
        }
        
        merged = _merge_exploration_rules(base_rules, derived_rules)
        
        # Verify extension behavior
        assert set(merged['required']) == {'LayerNorm', 'Softmax', 'MultiHead'}
        assert set(merged['optional']) == {'Softmax', 'GELU'}
        assert len(merged['mutually_exclusive']) == 2
        assert merged['dependencies']['Softmax'] == ['LayerNorm']  # Base preserved
        assert merged['dependencies']['MultiHead'] == ['LayerNorm']  # Derived added
        assert merged['dependencies']['GELU'] == ['Softmax']  # Derived added
    
    def test_circular_inheritance_detection(self):
        """Test detection of circular inheritance."""
        # Create circular inheritance: A -> B -> C -> A
        a_yaml = """
name: "blueprint_a"
base_blueprint: "blueprint_c"
nodes:
  canonical_ops: {available: []}
transforms:
  model_topology: {available: []}
"""
        
        b_yaml = """
name: "blueprint_b"
base_blueprint: "blueprint_a"
nodes:
  canonical_ops: {available: []}
transforms:
  model_topology: {available: []}
"""
        
        c_yaml = """
name: "blueprint_c"
base_blueprint: "blueprint_b"
nodes:
  canonical_ops: {available: []}
transforms:
  model_topology: {available: []}
"""
        
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for name, content in [
                ("blueprint_a.yaml", a_yaml),
                ("blueprint_b.yaml", b_yaml), 
                ("blueprint_c.yaml", c_yaml)
            ]:
                with open(temp_path / name, 'w') as f:
                    f.write(content)
            
            # Should detect circular dependency
            with pytest.raises(ValueError, match="Blueprint inheritance failed"):
                load_blueprint(str(temp_path / "blueprint_a.yaml"))
    
    def test_base_blueprint_resolution(self):
        """Test base blueprint path resolution."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            (temp_path / "base").mkdir()
            
            # Create base blueprint in subdirectory
            base_content = """
name: "base_blueprint"
nodes:
  canonical_ops: {available: []}
transforms:
  model_topology: {available: []}
"""
            with open(temp_path / "base" / "transformer_base.yaml", 'w') as f:
                f.write(base_content)
            
            # Test resolution from different paths
            blueprint_path = temp_path / "test_blueprint.yaml"
            
            # Should find in base/ subdirectory
            resolved = resolve_blueprint_path(blueprint_path, "transformer_base")
            assert resolved == temp_path / "base" / "transformer_base.yaml"
            
            # Test non-existent blueprint
            with pytest.raises(BlueprintInheritanceError, match="not found"):
                resolve_blueprint_path(blueprint_path, "non_existent")
    
    def test_inheritance_validation_chain(self):
        """Test inheritance chain validation."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid inheritance chain
            base_yaml = """
name: "base"
nodes:
  canonical_ops: {available: []}
transforms:
  model_topology: {available: []}
"""
            
            derived_yaml = """
name: "derived"
base_blueprint: "base"
nodes:
  canonical_ops: {available: []}
transforms:
  model_topology: {available: []}
"""
            
            for name, content in [
                ("base.yaml", base_yaml),
                ("derived.yaml", derived_yaml)
            ]:
                with open(temp_path / name, 'w') as f:
                    f.write(content)
            
            # Should validate successfully
            chain = validate_inheritance_chain(temp_path / "derived.yaml")
            assert "base" in chain
            assert "derived" in chain
    
    def test_complex_merging_scenario(self):
        """Test complex scenario with all merge types."""
        base_yaml = """
name: "complex_base"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
    exploration:
      required:
        - "LayerNorm"
      dependencies:
        Softmax: ["LayerNorm"]

  hw_kernels:
    available:
      - "MatMul":
          - "matmul_hls"
    exploration:
      required:
        - "MatMul"

transforms:
  model_topology:
    available:
      - "cleanup"
      - "streamlining"
    exploration:
      required:
        - "cleanup"
      mutually_exclusive:
        - ["fast_stream", "safe_stream"]

configuration_files:
  platform_config: "base_platform.yaml"

dse_strategies:
  primary_strategy: "base_strategy"
  strategies:
    base_strategy:
      max_evaluations: 50
      sampling: "random"

objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.0

constraints:
  - name: "max_power"
    operator: "<="
    value: 20.0
"""
        
        derived_yaml = """
name: "complex_derived"
version: "2.0"
base_blueprint: "complex_base"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
      - "MultiHeadAttention"  # Extended
      - "GELU"  # Extended
    exploration:
      required:
        - "LayerNorm"
        - "Softmax"  # Extended requirements
      optional:
        - "MultiHeadAttention"
        - "GELU"
      dependencies:
        MultiHeadAttention: ["LayerNorm"]  # New dependency
        GELU: ["LayerNorm"]  # New dependency

  hw_kernels:
    available:
      - "MatMul":
          - "matmul_hls"
          - "matmul_rtl"  # Extended options
      - "Conv2D":  # New component
          - "conv2d_hls"
    exploration:
      required:
        - "MatMul"
      optional:
        - "Conv2D"  # New optional

transforms:
  hw_kernel:  # New transform section
    available:
      - "apply_folding_config"
      - "target_fps_parallelization"
    exploration:
      required:
        - "apply_folding_config"

configuration_files:
  folding_override: "derived_folding.json"  # New file
  platform_config: "derived_platform.yaml"  # Override

dse_strategies:
  primary_strategy: "derived_strategy"  # Changed primary
  strategies:
    base_strategy:  # Keep base strategy
      max_evaluations: 50
      sampling: "random"
    derived_strategy:  # New strategy
      max_evaluations: 100
      sampling: "latin_hypercube"

objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.5  # Override weight
  - name: "latency"  # New objective
    direction: "minimize"
    weight: 0.8

constraints:
  - name: "max_power"
    operator: "<="
    value: 25.0  # Override value
  - name: "max_area"  # New constraint
    operator: "<="
    value: 0.8
"""
        
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write blueprints
            for name, content in [
                ("complex_base.yaml", base_yaml),
                ("complex_derived.yaml", derived_yaml)
            ]:
                with open(temp_path / name, 'w') as f:
                    f.write(content)
            
            # Load derived blueprint
            blueprint = load_blueprint(str(temp_path / "complex_derived.yaml"))
            
            # Verify all aspects were merged correctly
            
            # Canonical ops extended
            canonical_ops = blueprint.nodes.canonical_ops.get_component_names()
            assert len(canonical_ops) == 4
            assert "MultiHeadAttention" in canonical_ops
            assert "GELU" in canonical_ops
            
            # HW kernels extended and overridden
            matmul_options = blueprint.nodes.hw_kernels.get_component_options("MatMul")
            assert "matmul_rtl" in matmul_options
            assert "Conv2D" in blueprint.nodes.hw_kernels.get_component_names()
            
            # Transforms added new section
            assert len(blueprint.transforms.hw_kernel.get_component_names()) == 2
            
            # Configuration files merged
            assert blueprint.configuration_files["folding_override"] == "derived_folding.json"
            assert blueprint.configuration_files["platform_config"] == "derived_platform.yaml"
            
            # DSE strategies merged
            assert blueprint.dse_strategies.primary_strategy == "derived_strategy"
            assert len(blueprint.dse_strategies.strategies) == 2
            
            # Objectives merged with override
            throughput_obj = next(obj for obj in blueprint.objectives if obj.name == "throughput")
            assert throughput_obj.weight == 1.5
            assert len(blueprint.objectives) == 2
            
            # Constraints merged with override
            power_constraint = next(c for c in blueprint.constraints if c.name == "max_power")
            assert power_constraint.value == 25.0
            assert len(blueprint.constraints) == 2


if __name__ == "__main__":
    pytest.main([__file__])