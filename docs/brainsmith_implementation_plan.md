# Brainsmith Step Library Architecture - Implementation Plan

## Overview

This document provides a detailed implementation plan for refactoring Brainsmith to use a centralized step library with pure configuration-based blueprints. The plan is designed to be executed incrementally while maintaining backward compatibility.

## Phase 1: Foundation - Step Library Core (Weeks 1-3)

### Week 1: Step Registry Infrastructure

**Deliverable 1.1: Core Step Registry System**

Create the foundational step registry infrastructure:

```python
# brainsmith/steps/__init__.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import importlib
import pkgutil
from pathlib import Path

@dataclass
class StepMetadata:
    """Metadata describing a build step."""
    name: str
    category: str  # "common", "transformer", "cnn", "rnn", "finn"
    architecture: Optional[str] = None
    dependencies: List[str] = None
    conflicts: List[str] = None
    description: str = ""
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.conflicts is None:
            self.conflicts = []
        if self.parameters is None:
            self.parameters = {}

class StepRegistry:
    """Central registry for all build steps."""
    
    def __init__(self):
        self._steps: Dict[str, Callable] = {}
        self._metadata: Dict[str, StepMetadata] = {}
        self._initialized = False
    
    def register_step(self, metadata: StepMetadata, func: Callable):
        """Register a step in the library."""
        self._steps[metadata.name] = func
        self._metadata[metadata.name] = metadata
    
    def get_step(self, name: str) -> Callable:
        """Get a step by name."""
        if not self._initialized:
            self._load_all_steps()
        
        if name not in self._steps:
            raise StepNotFoundError(f"Step '{name}' not found in library")
        return self._steps[name]
    
    def get_metadata(self, name: str) -> StepMetadata:
        """Get metadata for a step."""
        if name not in self._metadata:
            raise StepNotFoundError(f"Step '{name}' not found in library")
        return self._metadata[name]
    
    def list_steps(self, category: str = None, architecture: str = None) -> List[str]:
        """List available steps, optionally filtered."""
        if not self._initialized:
            self._load_all_steps()
            
        steps = []
        for name, metadata in self._metadata.items():
            if category and metadata.category != category:
                continue
            if architecture and metadata.architecture != architecture:
                continue
            steps.append(name)
        return sorted(steps)
    
    def validate_step_sequence(self, steps: List[str]) -> 'ValidationResult':
        """Validate that a sequence of steps is compatible."""
        result = ValidationResult()
        
        for step_name in steps:
            if step_name not in self._metadata:
                result.add_error(f"Step '{step_name}' not found in library")
                continue
                
            metadata = self._metadata[step_name]
            
            # Check dependencies
            for dep in metadata.dependencies:
                if dep not in steps or steps.index(dep) > steps.index(step_name):
                    result.add_error(f"Step '{step_name}' depends on '{dep}' which must come before it")
            
            # Check conflicts
            for conflict in metadata.conflicts:
                if conflict in steps:
                    result.add_error(f"Step '{step_name}' conflicts with '{conflict}'")
        
        return result
    
    def _load_all_steps(self):
        """Load all steps from the steps package."""
        steps_package = importlib.import_module('brainsmith.steps')
        self._discover_and_load_steps(steps_package)
        self._initialized = True
    
    def _discover_and_load_steps(self, package):
        """Recursively discover and load steps from package."""
        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            try:
                importlib.import_module(modname)
            except ImportError as e:
                print(f"Warning: Could not load step module {modname}: {e}")

# Global registry instance
STEP_REGISTRY = StepRegistry()

def register_step(name: str, category: str, architecture: str = None, 
                 dependencies: List[str] = None, conflicts: List[str] = None,
                 description: str = "", parameters: Dict[str, Any] = None):
    """Decorator to register a step in the library."""
    def decorator(func: Callable) -> Callable:
        metadata = StepMetadata(
            name=name,
            category=category,
            architecture=architecture,
            dependencies=dependencies,
            conflicts=conflicts,
            description=description,
            parameters=parameters or {}
        )
        STEP_REGISTRY.register_step(metadata, func)
        return func
    return decorator

class StepNotFoundError(Exception):
    """Raised when a step is not found in the registry."""
    pass

class ValidationResult:
    """Result of step sequence validation."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_error(self, message: str):
        self.errors.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def is_valid(self) -> bool:
        return len(self.errors) == 0
```

**Deliverable 1.2: Directory Structure Setup**

Create the step library directory structure:

```bash
# Create directory structure
mkdir -p brainsmith/steps/{common,transformer,cnn,rnn,finn}
touch brainsmith/steps/__init__.py
touch brainsmith/steps/common/__init__.py
touch brainsmith/steps/transformer/__init__.py
touch brainsmith/steps/cnn/__init__.py
touch brainsmith/steps/rnn/__init__.py
touch brainsmith/steps/finn/__init__.py
```

**Deliverable 1.3: Testing Infrastructure**

```python
# tests/test_step_registry.py
import pytest
from brainsmith.steps import STEP_REGISTRY, register_step, StepNotFoundError

class TestStepRegistry:
    
    def test_register_and_get_step(self):
        @register_step(
            name="test.simple_step",
            category="test",
            description="A test step"
        )
        def test_step(model, cfg):
            return "test_result"
        
        step = STEP_REGISTRY.get_step("test.simple_step")
        assert step is test_step
        assert step(None, None) == "test_result"
    
    def test_step_not_found(self):
        with pytest.raises(StepNotFoundError):
            STEP_REGISTRY.get_step("nonexistent.step")
    
    def test_list_steps_by_category(self):
        @register_step(name="test.cat1", category="category1")
        def step1(model, cfg): pass
        
        @register_step(name="test.cat2", category="category2")
        def step2(model, cfg): pass
        
        cat1_steps = STEP_REGISTRY.list_steps(category="category1")
        assert "test.cat1" in cat1_steps
        assert "test.cat2" not in cat1_steps
    
    def test_validate_dependencies(self):
        steps = ["test.step2", "test.step1"]  # Wrong order
        result = STEP_REGISTRY.validate_step_sequence(steps)
        assert not result.is_valid()
        assert any("depends on" in error for error in result.errors)
```

### Week 2: FINN Step Wrappers

**Deliverable 2.1: Standard FINN Step Wrappers**

Create wrappers for all standard FINN steps:

```python
# brainsmith/steps/finn/__init__.py
from brainsmith.steps import register_step

# Import all FINN builder steps
try:
    from finn.builder.build_dataflow_steps import (
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
        step_create_stitched_ip
    )
except ImportError as e:
    print(f"Warning: Could not import FINN steps: {e}")
    # Provide mock implementations for development/testing
    
# Register all FINN steps
@register_step(
    name="finn.create_dataflow_partition",
    category="finn",
    description="Create dataflow partition from ONNX model"
)
def finn_create_dataflow_partition(model, cfg):
    return step_create_dataflow_partition(model, cfg)

@register_step(
    name="finn.specialize_layers",
    category="finn",
    description="Specialize layers for hardware implementation",
    dependencies=["finn.create_dataflow_partition"]
)
def finn_specialize_layers(model, cfg):
    return step_specialize_layers(model, cfg)

@register_step(
    name="finn.target_fps_parallelization", 
    category="finn",
    description="Apply target FPS-based parallelization",
    dependencies=["finn.specialize_layers"]
)
def finn_target_fps_parallelization(model, cfg):
    return step_target_fps_parallelization(model, cfg)

@register_step(
    name="finn.apply_folding_config",
    category="finn", 
    description="Apply folding configuration to model",
    dependencies=["finn.target_fps_parallelization"]
)
def finn_apply_folding_config(model, cfg):
    return step_apply_folding_config(model, cfg)

@register_step(
    name="finn.minimize_bit_width",
    category="finn",
    description="Minimize bit width for optimization",
    dependencies=["finn.apply_folding_config"]
)
def finn_minimize_bit_width(model, cfg):
    return step_minimize_bit_width(model, cfg)

@register_step(
    name="finn.generate_estimate_reports",
    category="finn",
    description="Generate resource and performance estimates",
    dependencies=["finn.minimize_bit_width"]
)
def finn_generate_estimate_reports(model, cfg):
    return step_generate_estimate_reports(model, cfg)

@register_step(
    name="finn.hw_codegen",
    category="finn",
    description="Generate hardware code",
    dependencies=["finn.generate_estimate_reports"]
)
def finn_hw_codegen(model, cfg):
    return step_hw_codegen(model, cfg)

@register_step(
    name="finn.hw_ipgen",
    category="finn",
    description="Generate IP blocks",
    dependencies=["finn.hw_codegen"]
)
def finn_hw_ipgen(model, cfg):
    return step_hw_ipgen(model, cfg)

@register_step(
    name="finn.measure_rtlsim_performance",
    category="finn",
    description="Measure RTL simulation performance",
    dependencies=["finn.hw_ipgen"]
)
def finn_measure_rtlsim_performance(model, cfg):
    return step_measure_rtlsim_performance(model, cfg)

@register_step(
    name="finn.set_fifo_depths",
    category="finn",
    description="Set FIFO depths for optimization",
    dependencies=["finn.measure_rtlsim_performance"]
)
def finn_set_fifo_depths(model, cfg):
    return step_set_fifo_depths(model, cfg)

@register_step(
    name="finn.create_stitched_ip",
    category="finn",
    description="Create final stitched IP",
    dependencies=["finn.set_fifo_depths"]
)
def finn_create_stitched_ip(model, cfg):
    return step_create_stitched_ip(model, cfg)
```

### Week 3: Common Steps

**Deliverable 3.1: Common Utility Steps**

Create general-purpose steps that can be used across architectures:

```python
# brainsmith/steps/common/cleanup.py
from brainsmith.steps import register_step
from qonnx.transformation.general import (
    SortCommutativeInputsInitializerLast,
    RemoveUnusedTensors,
    GiveReadableTensorNames,
    GiveUniqueNodeNames
)
from qonnx.transformation.remove import RemoveIdentityOps

@register_step(
    name="common.cleanup",
    category="common",
    description="General model cleanup and optimization"
)
def common_cleanup(model, cfg):
    """Common cleanup steps for all model types."""
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(GiveUniqueNodeNames())
    return model

@register_step(
    name="common.remove_unused_tensors",
    category="common",
    description="Remove unused tensors from model"
)
def common_remove_unused_tensors(model, cfg):
    """Remove unused tensors."""
    model = model.transform(RemoveUnusedTensors())
    return model

@register_step(
    name="common.give_readable_names", 
    category="common",
    description="Give readable names to tensors and nodes"
)
def common_give_readable_names(model, cfg):
    """Give readable names to model components."""
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(GiveUniqueNodeNames())
    return model

# brainsmith/steps/common/validation.py
@register_step(
    name="common.validate_model",
    category="common", 
    description="Validate model structure and properties"
)
def common_validate_model(model, cfg):
    """Validate model structure."""
    # Add model validation logic
    return model

@register_step(
    name="common.save_checkpoint",
    category="common",
    description="Save model checkpoint"
)
def common_save_checkpoint(model, cfg):
    """Save model checkpoint for debugging."""
    # Add checkpoint saving logic
    return model
```

## Phase 2: Migration - Transformer Steps (Weeks 4-6)

### Week 4: Extract BERT Custom Steps

**Deliverable 4.1: Migrate Existing BERT Steps**

Extract and migrate all custom steps from the current BERT blueprint:

```python
# brainsmith/steps/transformer/surgery.py
from brainsmith.steps import register_step

@register_step(
    name="transformer.remove_head",
    category="transformer",
    architecture="transformer",
    dependencies=["common.cleanup"],
    description="Remove transformer head up to first LayerNormalization"
)
def transformer_remove_head(model, cfg):
    """Remove all nodes up to the first LayerNormalisation Node and rewire input."""
    # Copy implementation from brainsmith.blueprints.bert.custom_step_remove_head
    assert len(model.graph.input) == 1, "Error the graph has more inputs than expected"
    tensor_to_node = {output: node for node in model.graph.node for output in node.output}

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
    architecture="transformer",
    description="Remove transformer tail from global_out_1 back to LayerNorm"
)
def transformer_remove_tail(model, cfg):
    """Remove from global_out_1 all the way back to the first LayerNorm."""
    # Copy implementation from brainsmith.blueprints.bert.custom_step_remove_tail
    # ... (full implementation)
    return model

# brainsmith/steps/transformer/conversion.py
@register_step(
    name="transformer.qonnx_to_finn",
    category="transformer", 
    architecture="transformer",
    dependencies=["transformer.remove_head", "transformer.remove_tail"],
    description="Convert QONNX to FINN with transformer-specific handling"
)
def transformer_qonnx_to_finn(model, cfg):
    """BERT custom step for converting between QONNX and FINN-ONNX."""
    # Copy implementation from brainsmith.blueprints.bert.custom_step_qonnx2finn
    from brainsmith.transformation.expand_norms import ExpandNorms
    from qonnx.transformation.fold_constants import FoldConstants
    from qonnx.transformation.general import ConvertDivToMul
    from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
    
    model = model.transform(ExpandNorms())
    model = model.transform(FoldConstants())
    model = model.transform(ConvertDivToMul()) 
    model = model.transform(ConvertQONNXtoFINN())
    return model

@register_step(
    name="transformer.generate_reference_io",
    category="transformer",
    architecture="transformer",
    dependencies=["transformer.qonnx_to_finn"],
    description="Generate reference input/output for validation"
)
def transformer_generate_reference_io(model, cfg):
    """Generate reference IO pair for the model with head/tail removed."""
    # Copy implementation from brainsmith.blueprints.bert.custom_step_generate_reference_io
    # ... (full implementation)
    return model

# brainsmith/steps/transformer/streamlining.py
@register_step(
    name="transformer.streamlining",
    category="transformer",
    architecture="transformer", 
    dependencies=["transformer.generate_reference_io"],
    description="Transformer-specific streamlining optimizations"
)
def transformer_streamlining(model, cfg):
    """BERT custom step for streamlining with SoftMax handling."""
    # Copy implementation from brainsmith.blueprints.bert.custom_streamlining_step
    # ... (full implementation)
    return model
```

**Deliverable 4.2: Step Migration Tool**

Create a tool to help migrate existing step functions:

```python
# tools/migrate_steps.py
import ast
import importlib
from pathlib import Path

class StepMigrationTool:
    """Tool to help migrate existing steps to the new library format."""
    
    def extract_steps_from_module(self, module_path: Path) -> List[dict]:
        """Extract step functions from existing blueprint module."""
        with open(module_path) as f:
            tree = ast.parse(f.read())
        
        steps = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('custom_step_'):
                steps.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'lineno': node.lineno,
                    'source': ast.get_source_segment(f.read(), node)
                })
        return steps
    
    def generate_migration_template(self, step_info: dict, category: str, architecture: str) -> str:
        """Generate template for migrating a step."""
        template = f"""
from brainsmith.steps import register_step

@register_step(
    name="{category}.{step_info['name'].replace('custom_step_', '')}",
    category="{category}",
    architecture="{architecture}",
    description="{step_info['docstring'] or 'TODO: Add description'}",
    dependencies=[]  # TODO: Add dependencies
)
def {step_info['name']}(model, cfg):
    \"\"\"{step_info['docstring'] or 'TODO: Add description'}\"\"\"
    # TODO: Copy implementation from original function
    pass
"""
        return template

# Usage example:
# python tools/migrate_steps.py --input brainsmith/blueprints/bert.py --output brainsmith/steps/transformer/
```

### Week 5: Architecture Detection and Common Patterns

**Deliverable 5.1: Model Analysis Utils**

```python
# brainsmith/steps/common/analysis.py
from brainsmith.steps import register_step

@register_step(
    name="common.analyze_model",
    category="common",
    description="Analyze model architecture and characteristics"
)
def common_analyze_model(model, cfg):
    """Analyze model to determine architecture type and characteristics."""
    analysis = {
        'architecture': detect_architecture(model),
        'parameter_count': count_parameters(model),
        'layer_types': get_layer_types(model),
        'input_shapes': get_input_shapes(model),
        'precision': detect_precision(model)
    }
    
    # Store analysis in config for later steps
    cfg.model_analysis = analysis
    return model

def detect_architecture(model) -> str:
    """Detect model architecture based on layer patterns."""
    layer_types = [node.op_type for node in model.graph.node]
    
    # Simple heuristics for architecture detection
    if 'ScaledDotProductAttention' in layer_types or 'MultiHeadAttention' in layer_types:
        return 'transformer'
    elif 'Conv' in layer_types and 'Pool' in layer_types:
        return 'cnn'
    elif any('LSTM' in lt or 'GRU' in lt for lt in layer_types):
        return 'rnn'
    else:
        return 'unknown'
```

### Week 6: Testing and Validation

**Deliverable 6.1: Comprehensive Testing**

```python
# tests/test_transformer_steps.py
import pytest
from brainsmith.steps import STEP_REGISTRY

class TestTransformerSteps:
    
    def test_transformer_step_registration(self):
        """Test that all transformer steps are properly registered."""
        expected_steps = [
            "transformer.remove_head",
            "transformer.remove_tail", 
            "transformer.qonnx_to_finn",
            "transformer.generate_reference_io",
            "transformer.streamlining"
        ]
        
        transformer_steps = STEP_REGISTRY.list_steps(architecture="transformer")
        for step in expected_steps:
            assert step in transformer_steps
    
    def test_step_dependencies(self):
        """Test that step dependencies are properly defined."""
        sequence = [
            "common.cleanup",
            "transformer.remove_head", 
            "transformer.remove_tail",
            "transformer.qonnx_to_finn",
            "transformer.generate_reference_io",
            "transformer.streamlining"
        ]
        
        result = STEP_REGISTRY.validate_step_sequence(sequence)
        assert result.is_valid(), f"Validation errors: {result.errors}"
    
    @pytest.fixture
    def sample_bert_model(self):
        """Create a sample BERT model for testing."""
        # Create or load sample model
        pass
    
    def test_transformer_remove_head(self, sample_bert_model):
        """Test transformer head removal step."""
        step = STEP_REGISTRY.get_step("transformer.remove_head")
        result = step(sample_bert_model, {})
        # Add assertions about the result
```

## Phase 3: Blueprint System (Weeks 7-9)

### Week 7: Pure Configuration Blueprints

**Deliverable 7.1: Blueprint Class and YAML Support**

```python
# brainsmith/core/blueprint.py
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from brainsmith.steps import STEP_REGISTRY

@dataclass 
class SearchStrategy:
    """Search strategy configuration."""
    type: str
    parameters: Dict[str, Any]
    optimization_goals: Dict[str, Any]
    iteration_phases: Dict[str, Dict[str, str]]
    
    @classmethod
    def from_config(cls, config: dict) -> 'SearchStrategy':
        return cls(
            type=config['type'],
            parameters=config.get('parameters', {}),
            optimization_goals=config.get('optimization_goals', {}),
            iteration_phases=config.get('iteration_phases', {})
        )

class Blueprint:
    """Pure configuration-based blueprint using step library."""
    
    def __init__(self, config: dict):
        self.name = config['name']
        self.description = config['description']
        self.architecture = config['architecture']
        self.build_steps = config['build_steps']
        self.search_strategy = SearchStrategy.from_config(config['search_strategy'])
        self.extends = config.get('extends')
        self._config = config
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Blueprint':
        """Load blueprint from YAML file."""
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def validate(self) -> 'ValidationResult':
        """Validate blueprint configuration."""
        from brainsmith.steps import ValidationResult
        result = ValidationResult()
        
        # Validate all steps exist in library
        for step_name in self.build_steps:
            try:
                STEP_REGISTRY.get_step(step_name)
            except Exception as e:
                result.add_error(f"Step '{step_name}' not found: {e}")
        
        # Validate step sequence
        sequence_result = STEP_REGISTRY.validate_step_sequence(self.build_steps)
        result.errors.extend(sequence_result.errors)
        result.warnings.extend(sequence_result.warnings)
        
        return result
    
    def get_steps_for_phase(self, phase: str) -> List[callable]:
        """Get executable steps for an iteration phase."""
        if phase not in self.search_strategy.iteration_phases:
            raise ValueError(f"Phase '{phase}' not defined in search strategy")
            
        phase_config = self.search_strategy.iteration_phases[phase]
        start_step = phase_config['start_step']
        stop_step = phase_config['stop_step']
        
        try:
            start_idx = self.build_steps.index(start_step)
            stop_idx = self.build_steps.index(stop_step) + 1
        except ValueError as e:
            raise ValueError(f"Phase step not found in build_steps: {e}")
        
        step_names = self.build_steps[start_idx:stop_idx]
        return [STEP_REGISTRY.get_step(name) for name in step_names]
    
    def get_all_steps(self) -> List[callable]:
        """Get all steps as executable functions."""
        return [STEP_REGISTRY.get_step(name) for name in self.build_steps]
```

**Deliverable 7.2: Blueprint Manager**

```python
# brainsmith/core/blueprint_manager.py
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from brainsmith.core.blueprint import Blueprint
from brainsmith.steps import STEP_REGISTRY, ValidationResult

class BlueprintManager:
    """Manages pure configuration blueprints with step library integration."""
    
    def __init__(self, blueprint_dirs: List[Path]):
        self.blueprint_dirs = [Path(d) for d in blueprint_dirs]
        self.loaded_blueprints: Dict[str, Blueprint] = {}
    
    def discover_blueprints(self) -> List[str]:
        """Discover all available blueprint YAML files."""
        blueprints = []
        for directory in self.blueprint_dirs:
            if directory.exists():
                for yaml_file in directory.glob("*.yaml"):
                    blueprints.append(yaml_file.stem)
                for yml_file in directory.glob("*.yml"):
                    blueprints.append(yml_file.stem)
        return sorted(set(blueprints))
    
    def load_blueprint(self, name: str) -> Blueprint:
        """Load blueprint from YAML configuration."""
        if name in self.loaded_blueprints:
            return self.loaded_blueprints[name]
            
        config_path = self._find_blueprint_file(name)
        if not config_path:
            raise FileNotFoundError(f"Blueprint '{name}' not found in {self.blueprint_dirs}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Handle inheritance
        if 'extends' in config:
            base_config = self._load_base_config(config['extends'])
            config = self._merge_configs(base_config, config)
        
        blueprint = Blueprint(config)
        
        # Validate against step library
        validation_result = blueprint.validate()
        if not validation_result.is_valid():
            raise BlueprintValidationError(
                f"Invalid blueprint '{name}': {validation_result.errors}"
            )
        
        self.loaded_blueprints[name] = blueprint
        return blueprint
    
    def create_blueprint_template(self, architecture: str, name: str = None) -> dict:
        """Create a blueprint template for a given architecture."""
        if not name:
            name = f"{architecture}_custom"
            
        # Get relevant steps from library
        common_steps = STEP_REGISTRY.list_steps(category='common')
        arch_steps = STEP_REGISTRY.list_steps(architecture=architecture)
        finn_steps = STEP_REGISTRY.list_steps(category='finn')
        
        # Create a reasonable default sequence
        default_steps = (
            ["common.cleanup"] + 
            arch_steps + 
            finn_steps[:5]  # First few FINN steps
        )
        
        template = {
            'name': name,
            'description': f"Custom blueprint for {architecture} architecture",
            'architecture': architecture,
            'build_steps': default_steps,
            'search_strategy': {
                'type': 'single_run',
                'parameters': {},
                'iteration_phases': {
                    'full_build': {
                        'start_step': default_steps[0],
                        'stop_step': default_steps[-1]
                    }
                }
            }
        }
        return template
    
    def _find_blueprint_file(self, name: str) -> Optional[Path]:
        """Find blueprint file in search directories."""
        for directory in self.blueprint_dirs:
            for ext in ['.yaml', '.yml']:
                path = directory / f"{name}{ext}"
                if path.exists():
                    return path
        return None
    
    def _load_base_config(self, base_name: str) -> dict:
        """Load base blueprint configuration for inheritance."""
        base_path = self._find_blueprint_file(base_name)
        if not base_path:
            raise FileNotFoundError(f"Base blueprint '{base_name}' not found")
        
        with open(base_path) as f:
            return yaml.safe_load(f)
    
    def _merge_configs(self, base_config: dict, derived_config: dict) -> dict:
        """Merge derived config with base config."""
        merged = base_config.copy()
        
        # Simple merge - derived config overrides base
        for key, value in derived_config.items():
            if key == 'build_steps' and isinstance(value, dict):
                # Handle step modifications
                if 'add_before' in value:
                    # Add steps before a certain step
                    pass
                elif 'add_after' in value:
                    # Add steps after a certain step  
                    pass
                elif 'replace' in value:
                    # Replace entire step list
                    merged[key] = value['replace']
                else:
                    merged[key] = value
            else:
                merged[key] = value
                
        return merged

class BlueprintValidationError(Exception):
    """Raised when blueprint validation fails."""
    pass
```

### Week 8: Sample Blueprint Collection

**Deliverable 8.1: Core Blueprint Library**

Create a collection of standard blueprints:

```yaml
# blueprints/transformer_conservative.yaml
name: "transformer_conservative"
description: "Conservative search strategy for transformer architectures"
architecture: "transformer"

build_steps:
  - "common.cleanup"
  - "transformer.remove_head"
  - "transformer.remove_tail"
  - "transformer.qonnx_to_finn"
  - "transformer.generate_reference_io"
  - "transformer.streamlining"
  - "finn.create_dataflow_partition"
  - "finn.specialize_layers"
  - "finn.target_fps_parallelization"
  - "finn.apply_folding_config"
  - "finn.minimize_bit_width"
  - "finn.generate_estimate_reports"
  - "finn.hw_codegen"
  - "finn.hw_ipgen"
  - "finn.measure_rtlsim_performance"
  - "finn.set_fifo_depths"
  - "finn.create_stitched_ip"

search_strategy:
  type: "conservative_exploration"
  parameters:
    initial_folding_factor: 0.5
    max_iterations: 5
    convergence_threshold: 0.95
    
  optimization_goals:
    primary: "resource_utilization"
    secondary: "timing_closure"
    
  iteration_phases:
    initial_analysis:
      start_step: "common.cleanup"
      stop_step: "finn.generate_estimate_reports"
      
    refinement:
      start_step: "finn.apply_folding_config"
      stop_step: "finn.measure_rtlsim_performance"
      
    finalization:
      start_step: "finn.set_fifo_depths"
      stop_step: "finn.create_stitched_ip"
```

```yaml
# blueprints/transformer_base.yaml
name: "transformer_base"
description: "Base transformer blueprint with common steps"
architecture: "transformer"

build_steps:
  - "common.cleanup"
  - "transformer.remove_head"
  - "transformer.remove_tail"
  - "transformer.qonnx_to_finn"
  - "transformer.generate_reference_io"
  - "transformer.streamlining"
  - "finn.create_dataflow_partition"
  - "finn.specialize_layers"
  - "finn.target_fps_parallelization"
  - "finn.apply_folding_config"
  - "finn.minimize_bit_width"
  - "finn.generate_estimate_reports"
  - "finn.hw_codegen"
  - "finn.hw_ipgen"
  - "finn.measure_rtlsim_performance"
  - "finn.set_fifo_depths"
  - "finn.create_stitched_ip"

search_strategy:
  type: "single_run"
  parameters: {}
  iteration_phases:
    full_build:
      start_step: "common.cleanup"
      stop_step: "finn.create_stitched_ip"
```

```yaml
# blueprints/transformer_aggressive.yaml
name: "transformer_aggressive"
description: "Aggressive optimization for transformer architectures"
extends: "transformer_base"

search_strategy:
  type: "genetic_algorithm"
  parameters:
    population_size: 8
    generations: 10
    mutation_rate: 0.2
    
  optimization_goals:
    primary: "throughput"
    secondary: "power_efficiency"
    
  iteration_phases:
    exploration:
      start_step: "common.cleanup"
      stop_step: "finn.apply_folding_config"
      
    evaluation:
      start_step: "finn.hw_codegen"
      stop_step: "finn.measure_rtlsim_performance"
```

```yaml
# blueprints/minimal_test.yaml
name: "minimal_test"
description: "Minimal blueprint for testing and debugging"
architecture: "any"

build_steps:
  - "common.cleanup"
  - "common.analyze_model"
  - "finn.create_dataflow_partition"
  - "finn.generate_estimate_reports"

search_strategy:
  type: "single_run"
  parameters: {}
  iteration_phases:
    test_run:
      start_step: "common.cleanup"
      stop_step: "finn.generate_estimate_reports"
```

### Week 9: Backward Compatibility Layer

**Deliverable 9.1: Legacy Integration**

```python
# brainsmith/core/legacy.py
from typing import List
from brainsmith.core.blueprint_manager import BlueprintManager
from brainsmith.core.blueprint import Blueprint
from pathlib import Path

class LegacyBlueprintAdapter:
    """Adapter to use legacy BUILD_STEPS with new system."""
    
    def __init__(self, legacy_steps: List[callable], name: str, architecture: str):
        self.legacy_steps = legacy_steps
        self.name = name
        self.architecture = architecture
    
    def to_blueprint_config(self) -> dict:
        """Convert legacy BUILD_STEPS to blueprint configuration."""
        # Map function names to step library names
        step_mapping = {
            'custom_step_cleanup': 'common.cleanup',
            'custom_step_remove_head': 'transformer.remove_head',
            'custom_step_remove_tail': 'transformer.remove_tail',
            'custom_step_qonnx2finn': 'transformer.qonnx_to_finn',
            'custom_step_generate_reference_io': 'transformer.generate_reference_io',
            'custom_streamlining_step': 'transformer.streamlining',
            'custom_step_infer_hardware': 'transformer.infer_hardware',
            # Add mappings for all standard FINN steps
            'step_create_dataflow_partition': 'finn.create_dataflow_partition',
            'step_specialize_layers': 'finn.specialize_layers',
            # ... etc
        }
        
        step_names = []
        for step_func in self.legacy_steps:
            func_name = step_func.__name__
            if func_name in step_mapping:
                step_names.append(step_mapping[func_name])
            else:
                # Unknown step - may need manual mapping
                step_names.append(f"legacy.{func_name}")
        
        config = {
            'name': self.name,
            'description': f"Legacy blueprint converted from {self.name}",
            'architecture': self.architecture,
            'build_steps': step_names,
            'search_strategy': {
                'type': 'single_run',
                'parameters': {},
                'iteration_phases': {
                    'full_build': {
                        'start_step': step_names[0],
                        'stop_step': step_names[-1]
                    }
                }
            }
        }
        return config

# Update the existing REGISTRY to use new system
# brainsmith/blueprints/__init__.py
from brainsmith.core.blueprint_manager import BlueprintManager
from brainsmith.core.legacy import LegacyBlueprintAdapter
from pathlib import Path

# Legacy support
from brainsmith.blueprints.bert import BUILD_STEPS

# Create legacy adapter
legacy_bert = LegacyBlueprintAdapter(BUILD_STEPS, "bert", "transformer")

# Set up blueprint manager with default directories
default_blueprint_dirs = [
    Path(__file__).parent.parent / "blueprints",  # Built-in blueprints
    Path.cwd() / "blueprints",  # User blueprints
]

BLUEPRINT_MANAGER = BlueprintManager(default_blueprint_dirs)

# Register legacy blueprint
bert_config = legacy_bert.to_blueprint_config()
legacy_bert_blueprint = Blueprint(bert_config)
BLUEPRINT_MANAGER.loaded_blueprints["bert"] = legacy_bert_blueprint

# Maintain old REGISTRY for backward compatibility
REGISTRY = {
    "bert": BUILD_STEPS,  # Keep existing for now
}
```

## Phase 4: Runner Integration (Weeks 10-12)

### Week 10: Updated Runner System

**Deliverable 10.1: Integrate with Runner**

```python
# brainsmith/core/runner.py
from brainsmith.core.blueprint_manager import BLUEPRINT_MANAGER
from brainsmith.core.blueprint import Blueprint

class BrainsmithRunner:
    """Updated runner using step library and blueprints."""
    
    def __init__(self, blueprint_manager=None):
        self.blueprint_manager = blueprint_manager or BLUEPRINT_MANAGER
    
    def execute(self, blueprint_name: str, model, config: dict):
        """Execute build using step library."""
        # Load blueprint
        blueprint = self.blueprint_manager.load_blueprint(blueprint_name)
        
        # Validate configuration
        validation_result = blueprint.validate()
        if not validation_result.is_valid():
            raise RuntimeError(f"Blueprint validation failed: {validation_result.errors}")
        
        # Execute steps
        if blueprint.search_strategy.type == "single_run":
            return self._execute_single_run(blueprint, model, config)
        else:
            return self._execute_iterative_search(blueprint, model, config)
    
    def _execute_single_run(self, blueprint: Blueprint, model, config: dict):
        """Execute all steps in sequence."""
        steps = blueprint.get_all_steps()
        
        current_model = model
        for step in steps:
            try:
                current_model = step(current_model, config)
            except Exception as e:
                raise RuntimeError(f"Step {step.__name__} failed: {e}")
        
        return current_model
    
    def _execute_iterative_search(self, blueprint: Blueprint, model, config: dict):
        """Execute iterative search strategy."""
        # Implementation for iterative execution
        # This would integrate with the search strategy system
        pass

# Update hw_compiler.py to use new system
def forge(blueprint: str, model, args) -> str:
    """
    Legacy forge function wrapper for backward compatibility.
    """
    # Convert legacy args to new configuration format
    config = _convert_legacy_args(args)
    
    # Create runner 
    runner = BrainsmithRunner()
    
    # Execute build
    result = runner.execute(blueprint, model, config)
    
    # Return legacy format result
    return args.output  # or extract from result
```

### Week 11: CLI Integration

**Deliverable 11.1: Updated CLI Commands**

```python
# brainsmith/cli/main.py
import click
from pathlib import Path
from brainsmith.core.blueprint_manager import BLUEPRINT_MANAGER
from brainsmith.core.runner import BrainsmithRunner

@click.group()
def cli():
    """Brainsmith - Neural Network to Hardware Compiler"""
    pass

@cli.command()
@click.option('--blueprint', '-b', required=True, help='Blueprint name')
@click.option('--model', '-m', required=True, type=click.Path(exists=True), help='Model file')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--output', '-o', required=True, help='Output directory')
def build(blueprint, model, config, output):
    """Build hardware implementation using specified blueprint."""
    click.echo(f"Building {model} using blueprint {blueprint}")
    
    # Load model
    import onnx
    model_obj = onnx.load(model)
    
    # Load config if provided
    build_config = {'output_dir': output}
    if config:
        import yaml
        with open(config) as f:
            build_config.update(yaml.safe_load(f))
    
    # Execute build
    runner = BrainsmithRunner()
    try:
        result = runner.execute(blueprint, model_obj, build_config)
        click.echo(f"Build completed successfully. Output: {output}")
    except Exception as e:
        click.echo(f"Build failed: {e}", err=True)
        raise click.Abort()

@cli.group()
def blueprint():
    """Blueprint management commands."""
    pass

@blueprint.command('list')
@click.option('--architecture', '-a', help='Filter by architecture')
def list_blueprints(architecture):
    """List available blueprints."""
    blueprints = BLUEPRINT_MANAGER.discover_blueprints()
    
    if architecture:
        # Filter by architecture (would need to load each blueprint)
        pass
    
    for bp in blueprints:
        click.echo(bp)

@blueprint.command('validate')
@click.argument('name')
def validate_blueprint(name):
    """Validate a blueprint."""
    try:
        bp = BLUEPRINT_MANAGER.load_blueprint(name)
        result = bp.validate()
        
        if result.is_valid():
            click.echo(f"Blueprint '{name}' is valid")
        else:
            click.echo(f"Blueprint '{name}' validation failed:")
            for error in result.errors:
                click.echo(f"  ERROR: {error}")
            for warning in result.warnings:
                click.echo(f"  WARNING: {warning}")
    except Exception as e:
        click.echo(f"Failed to load blueprint '{name}': {e}", err=True)

@blueprint.command('create')
@click.option('--architecture', '-a', required=True, help='Target architecture')
@click.option('--name', '-n', help='Blueprint name')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def create_blueprint(architecture, name, output):
    """Create a new blueprint template."""
    template = BLUEPRINT_MANAGER.create_blueprint_template(architecture, name)
    
    if output:
        import yaml
        with open(output, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        click.echo(f"Blueprint template created: {output}")
    else:
        import yaml
        click.echo(yaml.dump(template, default_flow_style=False, indent=2))

@cli.group()
def steps():
    """Step library management commands."""
    pass

@steps.command('list')
@click.option('--category', '-c', help='Filter by category')
@click.option('--architecture', '-a', help='Filter by architecture')
def list_steps(category, architecture):
    """List available steps."""
    from brainsmith.steps import STEP_REGISTRY
    
    step_names = STEP_REGISTRY.list_steps(category=category, architecture=architecture)
    
    for step_name in step_names:
        metadata = STEP_REGISTRY.get_metadata(step_name)
        click.echo(f"{step_name:30} {metadata.description}")

@steps.command('info')
@click.argument('name')
def step_info(name):
    """Show detailed information about a step."""
    from brainsmith.steps import STEP_REGISTRY
    
    try:
        metadata = STEP_REGISTRY.get_metadata(name)
        click.echo(f"Name: {metadata.name}")
        click.echo(f"Category: {metadata.category}")
        click.echo(f"Architecture: {metadata.architecture or 'Any'}")
        click.echo(f"Description: {metadata.description}")
        
        if metadata.dependencies:
            click.echo(f"Dependencies: {', '.join(metadata.dependencies)}")
        
        if metadata.conflicts:
            click.echo(f"Conflicts: {', '.join(metadata.conflicts)}")
            
    except Exception as e:
        click.echo(f"Step '{name}' not found: {e}", err=True)

if __name__ == '__main__':
    cli()
```

### Week 12: Documentation and Testing

**Deliverable 12.1: Complete Documentation**

```markdown
# Brainsmith Step Library Guide

## Overview

The Brainsmith step library provides a modular, extensible architecture for neural network hardware compilation. Steps are organized by category and can be composed into blueprints using simple YAML configuration.

## Step Categories

### Common Steps
- `common.cleanup` - General model cleanup
- `common.analyze_model` - Model architecture analysis
- `common.validate_model` - Model validation

### Transformer Steps  
- `transformer.remove_head` - Remove transformer head
- `transformer.remove_tail` - Remove transformer tail
- `transformer.qonnx_to_finn` - QONNX to FINN conversion
- `transformer.streamlining` - Transformer-specific optimizations

### FINN Steps
- `finn.create_dataflow_partition` - Create dataflow partition
- `finn.specialize_layers` - Specialize layers for hardware
- `finn.apply_folding_config` - Apply folding configuration

## Creating Blueprints

### Basic Blueprint
```yaml
name: "my_blueprint"
description: "Custom blueprint for my model"
architecture: "transformer"

build_steps:
  - "common.cleanup"
  - "transformer.remove_head"
  - "finn.create_dataflow_partition"

search_strategy:
  type: "single_run"
  parameters: {}
```

### Blueprint with Inheritance
```yaml
name: "my_optimized_blueprint"
extends: "transformer_base"

search_strategy:
  type: "conservative_exploration"
  parameters:
    max_iterations: 3
```

## CLI Usage

### Build a Model
```bash
brainsmith build --blueprint transformer_conservative --model model.onnx --output ./output
```

### List Available Blueprints
```bash
brainsmith blueprint list
brainsmith blueprint list --architecture transformer
```

### Create New Blueprint
```bash
brainsmith blueprint create --architecture transformer --name my_blueprint --output my_blueprint.yaml
```

### Validate Blueprint
```bash
brainsmith blueprint validate my_blueprint
```

### Explore Steps
```bash
brainsmith steps list
brainsmith steps list --category transformer
brainsmith steps info transformer.remove_head
```

## Adding Custom Steps

1. Create step file in appropriate directory:
```python
# brainsmith/steps/transformer/my_custom_step.py
from brainsmith.steps import register_step

@register_step(
    name="transformer.my_custom_step",
    category="transformer",
    architecture="transformer",
    dependencies=["common.cleanup"],
    description="My custom transformation"
)
def my_custom_step(model, cfg):
    # Implementation here
    return model
```

2. Add step to blueprint:
```yaml
build_steps:
  - "common.cleanup"
  - "transformer.my_custom_step"
  - "finn.create_dataflow_partition"
```

## Migration from Legacy Blueprints

Legacy blueprints are automatically supported through the compatibility layer. To migrate:

1. Create new YAML blueprint
2. Map legacy step functions to library steps
3. Test with new CLI
4. Update any custom scripts
```

**Deliverable 12.2: Integration Tests**

```python
# tests/test_integration.py
import pytest
import tempfile
from pathlib import Path
import yaml
import onnx
from brainsmith.core.runner import BrainsmithRunner
from brainsmith.core.blueprint_manager import BlueprintManager

class TestIntegration:
    
    @pytest.fixture
    def temp_blueprint_dir(self):
        """Create temporary directory with test blueprints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            blueprint_dir = Path(temp_dir) / "blueprints"
            blueprint_dir.mkdir()
            
            # Create test blueprint
            test_blueprint = {
                'name': 'test_blueprint',
                'description': 'Test blueprint',
                'architecture': 'transformer',
                'build_steps': [
                    'common.cleanup',
                    'common.analyze_model'
                ],
                'search_strategy': {
                    'type': 'single_run',
                    'parameters': {},
                    'iteration_phases': {
                        'test': {
                            'start_step': 'common.cleanup',
                            'stop_step': 'common.analyze_model'
                        }
                    }
                }
            }
            
            with open(blueprint_dir / "test_blueprint.yaml", 'w') as f:
                yaml.dump(test_blueprint, f)
            
            yield blueprint_dir
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple ONNX model for testing."""
        # Create minimal ONNX model
        pass
    
    def test_end_to_end_build(self, temp_blueprint_dir, sample_model):
        """Test complete build process with new system."""
        # Set up blueprint manager
        manager = BlueprintManager([temp_blueprint_dir])
        runner = BrainsmithRunner(manager)
        
        # Execute build
        config = {'output_dir': '/tmp/test_output'}
        result = runner.execute('test_blueprint', sample_model, config)
        
        # Verify result
        assert result is not None
    
    def test_backward_compatibility(self, sample_model):
        """Test that legacy forge() function still works."""
        from brainsmith.core.hw_compiler import forge
        
        # Mock args object
        class Args:
            output = '/tmp/test_output'
            # Add other required args
        
        args = Args()
        result = forge('bert', sample_model, args)
        assert result is not None
    
    def test_cli_integration(self, temp_blueprint_dir):
        """Test CLI commands."""
        from brainsmith.cli.main import cli
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test blueprint list
        result = runner.invoke(cli, ['blueprint', 'list'])
        assert result.exit_code == 0
        
        # Test step list
        result = runner.invoke(cli, ['steps', 'list'])
        assert result.exit_code == 0
```

## Risk Mitigation and Rollback Plan

### Rollback Strategy
1. **Feature Flags**: Use environment variables to toggle between old and new systems
2. **Gradual Migration**: Run both systems in parallel during transition
3. **Automated Testing**: Comprehensive regression tests ensure compatibility

### Risk Mitigation
1. **Backward Compatibility**: Legacy `forge()` API maintained
2. **Performance Monitoring**: Benchmark build times throughout migration
3. **Documentation**: Clear migration guides for users
4. **Community Support**: Gradual rollout with user feedback

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | Weeks 1-3 | Step Registry, FINN Wrappers, Common Steps |
| Phase 2 | Weeks 4-6 | Transformer Steps Migration, Testing |
| Phase 3 | Weeks 7-9 | Blueprint System, YAML Support, Legacy Compatibility |
| Phase 4 | Weeks 10-12 | Runner Integration, CLI, Documentation |

This implementation plan provides a structured approach to transforming Brainsmith into a modular, extensible system while maintaining full backward compatibility and improving the user experience.