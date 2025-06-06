# Blueprint System Refactoring: Step Library Implementation Plan

## Overview

This focused implementation plan covers refactoring the current blueprint system into:
1. **Step Library**: Centralized registry of reusable build steps
2. **YAML Blueprints**: Pure configuration-based blueprints using the step library
3. **Backward Compatibility**: Seamless transition from current system

## Current State Analysis

### Existing Blueprint Structure
- Single BERT blueprint in `brainsmith/blueprints/bert.py`
- Hardcoded `BUILD_STEPS` list with 17 custom and FINN steps
- Mixed responsibilities: step definitions + execution order + configuration

### Current BERT Steps
```python
BUILD_STEPS = [
    custom_step_cleanup,
    custom_step_remove_head,
    custom_step_remove_tail,
    custom_step_qonnx2finn,
    custom_step_generate_reference_io,
    custom_streamlining_step,
    custom_step_infer_hardware,
    step_create_dataflow_partition,
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_measure_rtlsim_performance,
    step_set_fifo_depths,
    step_create_stitched_ip,
    custom_step_shell_metadata_handover
]
```

## Implementation Plan: 3 Weeks

### Week 1: Step Library Foundation

#### Day 1-2: Core Step Registry

**File: `brainsmith/steps/__init__.py`**
```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import importlib.util
from pathlib import Path

@dataclass
class StepMetadata:
    """Metadata for a build step."""
    name: str
    category: str  # "common", "transformer", "finn"
    description: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class StepRegistry:
    """Central registry for all build steps."""
    
    def __init__(self):
        self._steps: Dict[str, Callable] = {}
        self._metadata: Dict[str, StepMetadata] = {}
        self._loaded = False
    
    def register_step(self, metadata: StepMetadata, func: Callable):
        """Register a step function."""
        self._steps[metadata.name] = func
        self._metadata[metadata.name] = metadata
    
    def get_step(self, name: str) -> Callable:
        """Get step function by name."""
        if not self._loaded:
            self._load_all_steps()
        
        if name not in self._steps:
            raise ValueError(f"Step '{name}' not found")
        return self._steps[name]
    
    def list_steps(self, category: str = None) -> List[str]:
        """List available steps."""
        if not self._loaded:
            self._load_all_steps()
        
        if category:
            return [name for name, meta in self._metadata.items() 
                   if meta.category == category]
        return list(self._steps.keys())
    
    def validate_sequence(self, step_names: List[str]) -> List[str]:
        """Validate step sequence and return any errors."""
        errors = []
        for step_name in step_names:
            if step_name not in self._metadata:
                errors.append(f"Step '{step_name}' not found")
                continue
                
            # Check dependencies
            metadata = self._metadata[step_name]
            for dep in metadata.dependencies:
                if dep not in step_names:
                    errors.append(f"Step '{step_name}' requires '{dep}'")
                elif step_names.index(dep) > step_names.index(step_name):
                    errors.append(f"Dependency '{dep}' must come before '{step_name}'")
        
        return errors
    
    def _load_all_steps(self):
        """Auto-discover and load all step modules."""
        steps_dir = Path(__file__).parent
        for category_dir in steps_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('_'):
                self._load_category(category_dir)
        self._loaded = True
    
    def _load_category(self, category_path: Path):
        """Load all steps from a category directory."""
        for py_file in category_path.glob("*.py"):
            if py_file.name != "__init__.py":
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"brainsmith.steps.{category_path.name}.{py_file.stem}",
                        py_file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"Warning: Failed to load {py_file}: {e}")

# Global registry
STEP_REGISTRY = StepRegistry()

def register_step(name: str, category: str, description: str = "", 
                 dependencies: List[str] = None):
    """Decorator to register a step."""
    def decorator(func: Callable) -> Callable:
        metadata = StepMetadata(
            name=name,
            category=category,
            description=description,
            dependencies=dependencies or []
        )
        STEP_REGISTRY.register_step(metadata, func)
        return func
    return decorator
```

#### Day 3-4: Directory Structure & Migration

**Create directory structure:**
```bash
mkdir -p brainsmith/steps/{common,transformer}
touch brainsmith/steps/{common,transformer}/__init__.py
```

**File: `brainsmith/steps/common/cleanup.py`**
```python
from brainsmith.steps import register_step

@register_step(
    name="common.cleanup",
    category="common",
    description="General model cleanup and optimization"
)
def common_cleanup(model, cfg):
    """Copy implementation from custom_step_cleanup in bert.py"""
    # Import and apply transformations
    from qonnx.transformation.general import (
        SortCommutativeInputsInitializerLast,
        RemoveUnusedTensors,
        GiveReadableTensorNames,
        GiveUniqueNodeNames
    )
    from qonnx.transformation.remove import RemoveIdentityOps
    
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(RemoveUnusedTensors()) 
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(GiveUniqueNodeNames())
    return model
```

#### Day 5: Basic Testing Framework

**File: `tests/test_step_registry.py`**
```python
import pytest
from brainsmith.steps import STEP_REGISTRY, register_step

def test_step_registration():
    """Test step registration and retrieval."""
    
    @register_step(
        name="test.sample_step",
        category="test",
        description="A test step"
    )
    def sample_step(model, cfg):
        return "test_result"
    
    # Test retrieval
    step = STEP_REGISTRY.get_step("test.sample_step")
    assert step is sample_step
    
    # Test execution
    result = step(None, None)
    assert result == "test_result"

def test_step_listing():
    """Test step listing functionality."""
    steps = STEP_REGISTRY.list_steps()
    assert len(steps) > 0
    
    # Test category filtering
    common_steps = STEP_REGISTRY.list_steps(category="common")
    assert "common.cleanup" in common_steps

def test_dependency_validation():
    """Test step dependency validation."""
    sequence = ["nonexistent.step"]
    errors = STEP_REGISTRY.validate_sequence(sequence)
    assert len(errors) > 0
    assert "not found" in errors[0]
```

### Week 2: Step Migration & YAML Blueprints

#### Day 6-8: Migrate BERT Steps to Library

**File: `brainsmith/steps/transformer/surgery.py`**
```python
from brainsmith.steps import register_step

@register_step(
    name="transformer.remove_head",
    category="transformer", 
    description="Remove transformer head up to first LayerNormalization",
    dependencies=["common.cleanup"]
)
def transformer_remove_head(model, cfg):
    """Migrate from custom_step_remove_head in bert.py"""
    # Copy exact implementation from brainsmith.blueprints.bert.custom_step_remove_head
    assert len(model.graph.input) == 1, "Error the graph has more inputs than expected"
    
    to_remove = []
    current_tensor = model.graph.input[0].name
    current_node = model.find_consumer(current_tensor)
    
    while current_node.op_type != "LayerNormalization":
        to_remove.append(current_node)
        assert len(current_node.output) == 1, "Error expected an linear path to the first LN"
        current_tensor = current_node.output[0]
        current_node = model.find_consumer(current_tensor)

    # Send the global input to the consumers of the layernorm output
    LN_output = current_node.output[0]
    consumers = model.find_consumers(LN_output)

    # Remove nodes
    to_remove.append(current_node)
    for node in to_remove:
        model.graph.node.remove(node)

    in_vi = model.get_tensor_valueinfo(LN_output)
    model.graph.input.pop()
    model.graph.input.append(in_vi)
    model.graph.value_info.remove(in_vi)

    # Reconnect input
    for con in consumers:
        for i, ip in enumerate(con.input):
            if ip == LN_output:
                con.input[i] = model.graph.input[0].name

    from qonnx.transformation.general import RemoveUnusedTensors, GiveReadableTensorNames
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())
    return model

@register_step(
    name="transformer.remove_tail",
    category="transformer",
    description="Remove transformer tail from output back to LayerNorm",
    dependencies=["transformer.remove_head"]
)
def transformer_remove_tail(model, cfg):
    """Migrate from custom_step_remove_tail in bert.py"""
    # Copy exact implementation
    return model  # Placeholder - copy full implementation

@register_step(
    name="transformer.qonnx_to_finn",
    category="transformer",
    description="Convert QONNX to FINN with transformer optimizations",
    dependencies=["transformer.remove_tail"]
)
def transformer_qonnx_to_finn(model, cfg):
    """Migrate from custom_step_qonnx2finn in bert.py"""
    # Copy exact implementation
    return model  # Placeholder - copy full implementation
```

**Note on FINN Steps:** FINN build steps remain in the FINN repository and are imported directly. The step registry will handle both Brainsmith steps and FINN steps transparently.

**Updated Step Registry to Handle FINN Steps:**
```python
# In brainsmith/steps/__init__.py - add FINN step handling
def get_step(self, name: str) -> Callable:
    """Get step function by name."""
    if not self._loaded:
        self._load_all_steps()
    
    # Check Brainsmith step library first
    if name in self._steps:
        return self._steps[name]
    
    # Check FINN steps as fallback
    try:
        from finn.builder.build_dataflow_steps import __dict__ as finn_steps
        if name in finn_steps:
            return finn_steps[name]
    except ImportError:
        pass
    
    raise ValueError(f"Step '{name}' not found")
```

#### Day 9-10: YAML Blueprint System

**File: `brainsmith/core/blueprint.py`**
```python
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from brainsmith.steps import STEP_REGISTRY

@dataclass
class Blueprint:
    """YAML-based blueprint configuration."""
    name: str
    description: str
    architecture: str
    build_steps: List[str]
    extends: str = None
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Blueprint':
        """Load blueprint from YAML file."""
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        return cls(
            name=config['name'],
            description=config['description'], 
            architecture=config['architecture'],
            build_steps=config['build_steps'],
            extends=config.get('extends')
        )
    
    def validate(self) -> List[str]:
        """Validate blueprint configuration."""
        return STEP_REGISTRY.validate_sequence(self.build_steps)
    
    def get_steps(self) -> List[callable]:
        """Get executable step functions."""
        return [STEP_REGISTRY.get_step(name) for name in self.build_steps]

class BlueprintManager:
    """Manages YAML blueprints."""
    
    def __init__(self, blueprint_dirs: List[Path]):
        self.blueprint_dirs = [Path(d) for d in blueprint_dirs]
        self._cache: Dict[str, Blueprint] = {}
    
    def load_blueprint(self, name: str) -> Blueprint:
        """Load blueprint by name."""
        if name in self._cache:
            return self._cache[name]
        
        # Find blueprint file
        blueprint_path = self._find_blueprint(name)
        if not blueprint_path:
            raise FileNotFoundError(f"Blueprint '{name}' not found")
        
        # Load and handle inheritance
        blueprint = Blueprint.from_yaml(blueprint_path)
        if blueprint.extends:
            base_blueprint = self.load_blueprint(blueprint.extends)
            blueprint = self._merge_blueprints(base_blueprint, blueprint)
        
        # Validate
        errors = blueprint.validate()
        if errors:
            raise ValueError(f"Blueprint validation failed: {errors}")
        
        self._cache[name] = blueprint
        return blueprint
    
    def list_blueprints(self) -> List[str]:
        """List available blueprints."""
        blueprints = []
        for directory in self.blueprint_dirs:
            if directory.exists():
                for yaml_file in directory.glob("*.yaml"):
                    blueprints.append(yaml_file.stem)
        return sorted(set(blueprints))
    
    def _find_blueprint(self, name: str) -> Path:
        """Find blueprint YAML file."""
        for directory in self.blueprint_dirs:
            blueprint_path = directory / f"{name}.yaml"
            if blueprint_path.exists():
                return blueprint_path
        return None
    
    def _merge_blueprints(self, base: Blueprint, derived: Blueprint) -> Blueprint:
        """Merge derived blueprint with base."""
        # Simple override for now
        return Blueprint(
            name=derived.name,
            description=derived.description,
            architecture=derived.architecture or base.architecture,
            build_steps=derived.build_steps or base.build_steps
        )
```

#### Day 11: Create BERT YAML Blueprint

**File: `blueprints/bert.yaml`**
```yaml
name: "bert"
description: "BERT transformer model compilation pipeline"
architecture: "transformer"

build_steps:
  - "common.cleanup"
  - "transformer.remove_head"
  - "transformer.remove_tail"
  - "transformer.qonnx_to_finn"
  - "transformer.generate_reference_io"
  - "transformer.streamlining"
  - "transformer.infer_hardware"
  - "step_create_dataflow_partition"      # Direct FINN import
  - "step_specialize_layers"              # Direct FINN import
  - "step_target_fps_parallelization"     # Direct FINN import
  - "step_apply_folding_config"           # Direct FINN import
  - "step_minimize_bit_width"             # Direct FINN import
  - "step_generate_estimate_reports"      # Direct FINN import
  - "step_hw_codegen"                     # Direct FINN import
  - "step_hw_ipgen"                       # Direct FINN import
  - "step_measure_rtlsim_performance"     # Direct FINN import
  - "step_set_fifo_depths"               # Direct FINN import
  - "step_create_stitched_ip"            # Direct FINN import
  - "transformer.shell_metadata_handover"
```

### Week 3: Integration & Backward Compatibility

#### Day 12-14: Update Core Integration

**File: `brainsmith/blueprints/__init__.py`**
```python
from pathlib import Path
from brainsmith.core.blueprint import BlueprintManager

# Set up blueprint directories
BLUEPRINT_DIRS = [
    Path(__file__).parent.parent / "blueprints",  # Built-in blueprints
    Path.cwd() / "blueprints"  # User blueprints  
]

BLUEPRINT_MANAGER = BlueprintManager(BLUEPRINT_DIRS)

# Backward compatibility: maintain REGISTRY
try:
    # Load BERT blueprint from YAML
    bert_blueprint = BLUEPRINT_MANAGER.load_blueprint("bert")
    bert_steps = bert_blueprint.get_steps()
    
    REGISTRY = {
        "bert": bert_steps
    }
except Exception as e:
    # Fallback to legacy system
    print(f"Warning: Could not load YAML blueprints, using legacy: {e}")
    from brainsmith.blueprints.bert import BUILD_STEPS
    REGISTRY = {
        "bert": BUILD_STEPS
    }
```

**Update `brainsmith/core/hw_compiler.py`:**
```python
# Add at top of file
from brainsmith.blueprints import BLUEPRINT_MANAGER

def forge(blueprint, model, args):
    """Updated forge function with blueprint system support."""
    
    # Try new blueprint system first
    try:
        bp = BLUEPRINT_MANAGER.load_blueprint(blueprint)
        steps = bp.get_steps()
    except Exception as e:
        # Fallback to legacy REGISTRY
        print(f"Using legacy blueprint system: {e}")
        if blueprint in REGISTRY.keys():
            steps = REGISTRY[blueprint]
        else:
            raise ValueError(f"Blueprint {blueprint} not found")
    
    # Rest of function remains the same
    # ... existing implementation
```

#### Day 15-17: Testing & Validation

**File: `tests/test_blueprint_system.py`**
```python
import pytest
from pathlib import Path
import tempfile
import yaml
from brainsmith.core.blueprint import Blueprint, BlueprintManager

def test_yaml_blueprint_loading():
    """Test loading blueprint from YAML."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test blueprint
        blueprint_config = {
            'name': 'test_blueprint',
            'description': 'Test blueprint',
            'architecture': 'transformer',
            'build_steps': ['common.cleanup']
        }
        
        blueprint_path = Path(temp_dir) / "test_blueprint.yaml"
        with open(blueprint_path, 'w') as f:
            yaml.dump(blueprint_config, f)
        
        # Load blueprint
        blueprint = Blueprint.from_yaml(blueprint_path)
        assert blueprint.name == 'test_blueprint'
        assert 'common.cleanup' in blueprint.build_steps

def test_blueprint_validation():
    """Test blueprint validation."""
    blueprint = Blueprint(
        name="test",
        description="test",
        architecture="test",
        build_steps=["nonexistent.step"]
    )
    
    errors = blueprint.validate()
    assert len(errors) > 0

def test_blueprint_manager():
    """Test blueprint manager functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = BlueprintManager([Path(temp_dir)])
        
        # Test empty directory
        blueprints = manager.list_blueprints()
        assert len(blueprints) == 0

def test_backward_compatibility():
    """Test that legacy REGISTRY still works."""
    from brainsmith.blueprints import REGISTRY
    assert "bert" in REGISTRY
    assert len(REGISTRY["bert"]) > 0
```

**File: `tests/test_end_to_end_blueprint.py`**
```python
import pytest
from brainsmith.core.hw_compiler import forge

def test_yaml_blueprint_execution():
    """Test end-to-end execution with YAML blueprint."""
    # This would test the full pipeline with a simple model
    # Mock model and args for testing
    pass

def test_legacy_compatibility():
    """Test that existing BERT blueprint still works."""
    # Ensure existing tests still pass
    pass
```

## Migration Strategy

### Phase 1: Infrastructure (Week 1)
- Set up step registry and basic structure
- No changes to existing blueprint system
- New system runs in parallel

### Phase 2: Content Migration (Week 2)  
- Migrate BERT steps to step library
- Create YAML blueprint equivalent
- Test both systems work independently

### Phase 3: Integration (Week 3)
- Update core system to prefer YAML blueprints
- Maintain backward compatibility with REGISTRY
- Comprehensive testing

## Rollback Plan

If issues arise, rollback is simple:
1. **Week 1**: New code is isolated, no impact on existing system
2. **Week 2**: YAML blueprints are additive, legacy system untouched  
3. **Week 3**: Changes to `forge()` can be reverted to use only REGISTRY

## Success Criteria

### Week 1 Complete
- [ ] Step registry infrastructure works
- [ ] Basic step registration and lookup functional
- [ ] Tests pass for core components

### Week 2 Complete  
- [ ] All BERT steps migrated to step library
- [ ] YAML BERT blueprint created and validated
- [ ] Step library produces same results as legacy steps

### Week 3 Complete
- [ ] YAML blueprints integrate with `forge()` function
- [ ] Backward compatibility maintained (all existing tests pass)
- [ ] New system can execute BERT pipeline end-to-end

## Benefits of This Focused Approach

1. **Manageable Scope**: Only blueprint system, not full runner architecture
2. **Immediate Value**: Cleaner, more maintainable blueprint definitions
3. **Foundation**: Sets up infrastructure for future enhancements
4. **Low Risk**: Strong backward compatibility and rollback options
5. **User Impact**: Minimal disruption to existing workflows
6. **Clean Dependencies**: FINN steps remain in FINN repository, avoiding duplication

## Key Architectural Decisions

- **FINN Steps**: Remain as direct imports from `finn.builder.build_dataflow_steps`
- **Brainsmith Steps**: Focus on model-specific transformations and common utilities
- **Step Registry**: Handles both Brainsmith and FINN steps transparently
- **YAML Blueprints**: Reference steps by their actual function names

This focused plan delivers the core architectural improvement (step library + YAML blueprints) while respecting the existing FINN ecosystem and maintaining clean separation of concerns.