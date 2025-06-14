"""
Comprehensive tests for Blueprint V2 validation system.
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from brainsmith.core.blueprint_v2 import (
    DesignSpaceDefinition, NodeDesignSpace, TransformDesignSpace,
    ComponentSpace, ExplorationRules, DSEStrategies, DSEStrategy,
    Objective, Constraint, OptimizationDirection, load_blueprint_v2
)


class TestValidationSystem:
    """Test comprehensive validation system."""
    
    def test_invalid_exploration_rules_combinations(self):
        """Test various invalid exploration rule combinations."""
        
        # Test 1: Component in both required and optional
        invalid_yaml = """
name: "invalid_test_1"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
    exploration:
      required:
        - "LayerNorm"
      optional:
        - "LayerNorm"  # Conflict: also in required

transforms:
  model_topology:
    available: []
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint_v2(f.name)
        
        Path(f.name).unlink()
        
        # Test 2: Required component in mutually exclusive group
        invalid_yaml2 = """
name: "invalid_test_2"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
      - "Alternative"
    exploration:
      required:
        - "LayerNorm"
      mutually_exclusive:
        - ["LayerNorm", "Alternative"]  # Conflict: LayerNorm is required

transforms:
  model_topology:
    available: []
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml2)
            f.flush()
            
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint_v2(f.name)
        
        Path(f.name).unlink()
        
        # Test 3: Single component in mutually exclusive group
        invalid_yaml3 = """
name: "invalid_test_3"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
    exploration:
      mutually_exclusive:
        - ["LayerNorm"]  # Invalid: only one component

transforms:
  model_topology:
    available: []
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml3)
            f.flush()
            
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint_v2(f.name)
        
        Path(f.name).unlink()
    
    def test_circular_dependency_validation(self):
        """Test circular dependency detection."""
        invalid_yaml = """
name: "circular_deps_test"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "A"
      - "B"
      - "C"
    exploration:
      required:
        - "A"
        - "B"
        - "C"
      dependencies:
        A: ["B"]
        B: ["C"]
        C: ["A"]  # Circular dependency: A -> B -> C -> A

transforms:
  model_topology:
    available: []
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint_v2(f.name)
        
        Path(f.name).unlink()
    
    def test_invalid_component_references(self):
        """Test validation of component references in exploration rules."""
        invalid_yaml = """
name: "invalid_refs_test"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
    exploration:
      required:
        - "LayerNorm"
        - "NonExistentComponent"  # References non-existent component
      dependencies:
        LayerNorm: ["AnotherNonExistent"]  # References non-existent dependency

transforms:
  model_topology:
    available: []
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint_v2(f.name)
        
        Path(f.name).unlink()
    
    def test_invalid_dse_strategy_configurations(self):
        """Test validation of DSE strategy configurations."""
        
        # Test 1: Invalid max_evaluations
        with pytest.raises(ValueError, match="max_evaluations must be positive"):
            DSEStrategy(name="invalid_strategy", max_evaluations=0)
        
        # Test 2: Invalid sampling strategy
        with pytest.raises(ValueError, match="Invalid sampling strategy"):
            DSEStrategy(name="invalid_strategy", sampling="invalid_sampling_method")
        
        # Test 3: Primary strategy not in strategies dict
        strategy = DSEStrategy(name="valid_strategy")
        
        with pytest.raises(ValueError, match="Primary strategy .* not found"):
            DSEStrategies(
                primary_strategy="missing_strategy",
                strategies={"valid_strategy": strategy}
            )
    
    def test_invalid_objectives_and_constraints(self):
        """Test validation of objectives and constraints."""
        
        # Test 1: Invalid objective weight
        with pytest.raises(ValueError, match="weight must be positive"):
            Objective(
                name="invalid_obj",
                direction=OptimizationDirection.MAXIMIZE,
                weight=-1.0
            )
        
        # Test 2: Invalid constraint operator
        with pytest.raises(ValueError, match="Invalid constraint operator"):
            Constraint(
                name="invalid_constraint",
                operator="invalid_op",
                value=10
            )
        
        # Test 3: Duplicate objective names
        invalid_yaml = """
name: "duplicate_objectives_test"
version: "2.0"

nodes:
  canonical_ops:
    available: []

transforms:
  model_topology:
    available: []

objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.0
  - name: "throughput"  # Duplicate name
    direction: "minimize"
    weight: 0.5
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint_v2(f.name)
        
        Path(f.name).unlink()
        
        # Test 4: Duplicate constraint names
        invalid_yaml2 = """
name: "duplicate_constraints_test"
version: "2.0"

nodes:
  canonical_ops:
    available: []

transforms:
  model_topology:
    available: []

constraints:
  - name: "max_power"
    operator: "<="
    value: 25.0
  - name: "max_power"  # Duplicate name
    operator: "<="
    value: 20.0
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml2)
            f.flush()
            
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint_v2(f.name)
        
        Path(f.name).unlink()
    
    def test_invalid_component_space_formats(self):
        """Test validation of component space formats."""
        invalid_yaml = """
name: "invalid_format_test"
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "ValidComponent"
      - 123  # Invalid: not a string or dict
      - "MatMul":
          - "option1"
          - 456  # Invalid: option not a string

transforms:
  model_topology:
    available: []
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint_v2(f.name)
        
        Path(f.name).unlink()
    
    def test_empty_blueprint_validation(self):
        """Test validation of empty/minimal blueprints."""
        
        # Test 1: Missing name
        invalid_yaml = """
version: "2.0"

nodes:
  canonical_ops:
    available: []

transforms:
  model_topology:
    available: []
"""
        
        blueprint = DesignSpaceDefinition(
            name="",  # Empty name
            nodes=NodeDesignSpace(),
            transforms=TransformDesignSpace()
        )
        
        is_valid, errors = blueprint.validate()
        assert not is_valid
        assert any("name is required" in err for err in errors)
        
        # Test 2: Missing version
        blueprint2 = DesignSpaceDefinition(
            name="test",
            version="",  # Empty version
            nodes=NodeDesignSpace(),
            transforms=TransformDesignSpace()
        )
        
        is_valid, errors = blueprint2.validate()
        assert not is_valid
        assert any("version is required" in err for err in errors)
    
    def test_component_space_empty_validation(self):
        """Test validation of empty component spaces."""
        
        # Empty component space should be valid (no components available)
        empty_space = ComponentSpace(available=[])
        is_valid, errors = empty_space.validate()
        assert not is_valid  # Should fail because no components available
        assert any("at least one available component" in err for err in errors)
        
        # Component space with empty exploration rules should be valid
        valid_space = ComponentSpace(
            available=["component1"],
            exploration=ExplorationRules()  # Empty rules
        )
        is_valid, errors = valid_space.validate()
        assert is_valid
    
    def test_complex_validation_scenario(self):
        """Test complex validation scenario with multiple issues."""
        complex_invalid_yaml = """
name: ""  # Missing name
version: "2.0"

nodes:
  canonical_ops:
    available:
      - "LayerNorm"
      - "Softmax"
      - 123  # Invalid type
    exploration:
      required:
        - "LayerNorm"
        - "NonExistent"  # Invalid reference
      optional:
        - "LayerNorm"  # Conflict with required
      mutually_exclusive:
        - ["SingleComponent"]  # Invalid: single component
      dependencies:
        LayerNorm: ["NonExistent2"]  # Invalid reference

  hw_kernels:
    available: []  # Empty space

transforms:
  model_topology:
    available:
      - "cleanup"
    exploration:
      dependencies:
        cleanup: ["cleanup"]  # Self-dependency (circular)

dse_strategies:
  primary_strategy: "missing_strategy"  # Invalid reference
  strategies:
    valid_strategy:
      max_evaluations: -10  # Invalid value
      sampling: "invalid_method"  # Invalid sampling

objectives:
  - name: "throughput"
    direction: "maximize"
    weight: -1.0  # Invalid weight
  - name: "throughput"  # Duplicate name
    direction: "minimize"
    weight: 1.0

constraints:
  - name: "max_power"
    operator: "invalid_op"  # Invalid operator
    value: 25.0
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(complex_invalid_yaml)
            f.flush()
            
            # Should catch multiple validation errors
            with pytest.raises(ValueError, match="Invalid blueprint"):
                load_blueprint_v2(f.name)
        
        Path(f.name).unlink()
    
    def test_validation_error_messages_quality(self):
        """Test that validation error messages are informative."""
        
        # Create blueprint with specific validation issues
        blueprint = DesignSpaceDefinition(
            name="error_message_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["LayerNorm", "Softmax"],
                    exploration=ExplorationRules(
                        required=["LayerNorm"],
                        optional=["LayerNorm"],  # Conflict
                        mutually_exclusive=[["LayerNorm", "NonExistent"]],  # Invalid ref
                        dependencies={"Softmax": ["NonExistent2"]}  # Invalid ref
                    )
                )
            ),
            transforms=TransformDesignSpace()
        )
        
        is_valid, errors = blueprint.validate()
        assert not is_valid
        
        # Check that error messages are specific and helpful
        error_text = " ".join(errors)
        assert "required and optional" in error_text.lower()
        assert "non-existent components" in error_text.lower()
        
        # Errors should mention the specific problematic components
        assert "LayerNorm" in error_text
        assert "NonExistent" in error_text


class TestRegistryValidation:
    """Test validation against component registries."""
    
    def test_mock_registry_validation(self):
        """Test validation against mock registries."""
        
        # Create component space that references non-existent registry components
        space = ComponentSpace(
            available=["real_component", "fake_component"],
            exploration=ExplorationRules(required=["real_component"])
        )
        
        # Basic validation should pass (no registry checking in base validate)
        is_valid, errors = space.validate()
        assert is_valid
        
        # Note: Registry validation would happen in higher-level blueprint validation
        # when integrated with actual kernel/transform registries
    
    def test_validation_with_registry_integration_placeholder(self):
        """Placeholder for future registry integration tests."""
        
        # This test would verify that blueprint validation checks:
        # 1. Canonical ops exist in custom ops registry
        # 2. HW kernels exist in kernels registry  
        # 3. Transforms exist in transforms registry
        
        # For now, just verify the structure is in place
        blueprint = DesignSpaceDefinition(
            name="registry_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(available=["LayerNorm"]),
                hw_kernels=ComponentSpace(available=["MatMul"])
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(available=["cleanup"])
            )
        )
        
        # Basic validation should pass
        is_valid, errors = blueprint.validate()
        assert is_valid
        
        # Get all components for future registry validation
        all_components = blueprint.get_all_components()
        assert "canonical_ops" in all_components
        assert "hw_kernels" in all_components
        assert "model_topology" in all_components


if __name__ == "__main__":
    pytest.main([__file__])