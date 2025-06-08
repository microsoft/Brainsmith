# 📋 Blueprint System Architecture
## Configuration-Driven Design Framework

---

## 🎯 Blueprint System Overview

The Blueprint System provides a powerful, YAML-based configuration framework that enables users to define complete FPGA accelerator design workflows declaratively. It abstracts the complexity of FPGA design while providing fine-grained control over optimization parameters and objectives.

### Key Features

- **YAML-Based Configuration**: Human-readable, version-controllable design specifications
- **Multi-Model Support**: Templates for various neural network architectures
- **Hierarchical Parameters**: Nested configuration with inheritance and overrides
- **Target-Driven Optimization**: Performance goal specification with automatic optimization
- **Template System**: Reusable design patterns and best practices
- **Validation Framework**: Comprehensive input checking and error reporting

---

## 🏗️ Blueprint Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   BLUEPRINT SYSTEM                      │
├─────────────────────────────────────────────────────────┤
│                  Blueprint Manager                      │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Blueprint Loader                       │ │
│  │  • YAML parsing and validation                      │ │
│  │  • Template expansion and inheritance               │ │
│  │  • Variable substitution                            │ │
│  │  • Schema validation                                │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Configuration Processing                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│  │     Model       │ │    Design       │ │   Target    │ │
│  │  Configuration  │ │     Space       │ │   Goals     │ │
│  │                 │ │   Generation    │ │             │ │
│  │ • Architecture  │ │ • Parameter     │ │ • Performance│ │
│  │ • Layers        │ │   ranges        │ │ • Power     │ │
│  │ • Quantization  │ │ • Constraints   │ │ • Resources │ │
│  │ • Custom ops    │ │ • Objectives    │ │ • Latency   │ │
│  └─────────────────┘ └─────────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────┤
│                 Integration Layer                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Integration Engine                     │ │
│  │  • Design space orchestrator integration            │ │
│  │  • Library mapper configuration                    │ │
│  │  • DSE engine parameter mapping                    │ │
│  │  • Workflow orchestration setup                    │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Blueprint Structure

```yaml
# Example Brainsmith Blueprint
name: "bert_edge_optimization"
description: "BERT model optimization for edge deployment"
version: "1.0"

# Model specification
model:
  name: "bert_base"
  type: "transformer"
  architecture: "bert"
  layers: 12
  hidden_size: 768
  attention_heads: 12
  
  # Model file location
  source:
    format: "onnx"
    path: "./models/bert_base.onnx"
    
  # Preprocessing requirements
  preprocessing:
    - normalize_inputs
    - quantization_calibration

# Target performance goals
targets:
  performance:
    throughput_ops_sec: 1000000  # 1M inferences/sec
    latency_ms: 50              # Maximum 50ms latency
    accuracy_retention: 0.95    # Maintain 95% of original accuracy
    
  resources:
    max_lut_utilization: 0.8    # 80% LUT utilization limit
    max_dsp_utilization: 0.9    # 90% DSP utilization limit
    max_bram_utilization: 0.7   # 70% BRAM utilization limit
    
  power:
    max_static_power_w: 5.0     # 5W static power limit
    max_dynamic_power_w: 15.0   # 15W dynamic power limit

# Design space specification
design_space:
  parameters:
    # Parallelism parameters
    pe_count:
      type: integer
      range: [2, 16]
      default: 8
      description: "Number of processing elements"
      
    simd_factor:
      type: integer
      range: [1, 8]
      default: 4
      description: "SIMD width for parallel operations"
      
    # Memory configuration
    memory_mode:
      type: categorical
      values: ["internal", "external", "hybrid"]
      default: "external"
      description: "Memory hierarchy configuration"
      
    # Clock frequency
    clock_frequency_mhz:
      type: float
      range: [100.0, 300.0]
      default: 250.0
      description: "Target clock frequency in MHz"
      
    # Quantization settings
    quantization_bits:
      type: categorical
      values: [8, 16]
      default: 8
      description: "Quantization bit width"
  
  # Parameter constraints
  constraints:
    - name: "resource_limit"
      expression: "pe_count * simd_factor <= 64"
      description: "Total parallel units constraint"
      
    - name: "memory_bandwidth"
      expression: "memory_mode != 'internal' or pe_count <= 8"
      description: "Internal memory bandwidth limitation"

# Design space exploration configuration
dse:
  strategy: "adaptive"
  max_evaluations: 100
  objectives:
    - name: "throughput_ops_sec"
      direction: "maximize"
      weight: 0.4
      
    - name: "power_efficiency"
      direction: "maximize"
      weight: 0.3
      
    - name: "resource_efficiency"
      direction: "maximize"
      weight: 0.3
  
  # Early stopping criteria
  convergence:
    enabled: true
    patience: 10
    min_improvement: 0.01

# Library configurations
libraries:
  transforms:
    enabled: true
    pipeline:
      - quantization
      - layer_folding
      - streamlining
    
    config:
      quantization:
        calibration_dataset_size: 1000
        per_channel: true
        
  hw_optim:
    enabled: true
    strategy: "genetic"
    config:
      population_size: 50
      generations: 25
      crossover_rate: 0.8
      mutation_rate: 0.1
      
  analysis:
    enabled: true
    analyses:
      - roofline_analysis
      - resource_profiling
      - bottleneck_identification
    
    config:
      roofline:
        platform_specs:
          peak_ops_sec: 1e12
          memory_bandwidth_gbps: 512

# Platform and build settings
platform:
  board: "ZCU104"
  part: "xczu7ev-ffvc1156-2-e"
  clock_period_ns: 4.0  # 250 MHz
  
build:
  output_dir: "./build/bert_edge"
  parallel_builds: 4
  verification_enabled: true
  synthesis_enabled: true
  
  # FINN-specific settings
  finn_config:
    auto_fifo_depths: true
    folding_config_file: "./configs/bert_folding.json"
    
# Reporting and analysis
reporting:
  formats: ["html", "json", "csv"]
  include_plots: true
  comparative_analysis: true
  research_export: true
```

---

## 🔧 Blueprint Processing Pipeline

### Loading and Validation Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   YAML      │───▶│   Parser    │───▶│  Template   │───▶│  Schema     │
│   File      │    │             │    │  Expansion  │    │ Validation  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Syntax      │    │ Variable    │    │ Inheritance │    │ Constraint  │
│ Checking    │    │ Resolution  │    │ Processing  │    │ Validation  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
                                    ┌─────────────┐
                                    │ Blueprint   │
                                    │ Object      │
                                    └─────────────┘
```

### Blueprint Manager Implementation

```python
class BlueprintManager:
    """Central manager for blueprint loading and processing."""
    
    def __init__(self):
        self.blueprint_registry = {}
        self.template_cache = {}
        self.validator = BlueprintValidator()
        
    def load_blueprint(self, blueprint_path: str) -> Blueprint:
        """Load and process blueprint from file."""
        
        # Load YAML content
        with open(blueprint_path, 'r') as f:
            raw_content = yaml.safe_load(f)
        
        # Process templates and inheritance
        processed_content = self._process_templates(raw_content)
        
        # Validate against schema
        validation_result = self.validator.validate(processed_content)
        if not validation_result.is_valid:
            raise BlueprintValidationError(
                f"Blueprint validation failed: {validation_result.errors}"
            )
        
        # Create blueprint object
        blueprint = Blueprint.from_dict(processed_content)
        blueprint.source_path = blueprint_path
        
        # Register blueprint
        self.blueprint_registry[blueprint.name] = blueprint
        
        return blueprint
    
    def _process_templates(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process template expansion and variable substitution."""
        
        # Handle template inheritance
        if 'extends' in content:
            base_template = self._load_template(content['extends'])
            content = self._merge_templates(base_template, content)
        
        # Variable substitution
        if 'variables' in content:
            content = self._substitute_variables(content, content['variables'])
        
        return content
    
    def _merge_templates(self, base: Dict[str, Any], 
                        override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge template configurations with inheritance."""
        
        result = copy.deepcopy(base)
        
        # Deep merge with override priority
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_templates(result[key], value)
            else:
                result[key] = value
        
        return result
```

### Blueprint Object Model

```python
@dataclass
class Blueprint:
    """Blueprint object representing a complete design configuration."""
    
    name: str
    description: str = ""
    version: str = "1.0"
    
    # Core configuration sections
    model: Optional[ModelConfig] = None
    targets: Optional[TargetConfig] = None
    design_space: Optional[DesignSpaceConfig] = None
    dse: Optional[DSEConfig] = None
    libraries: Optional[LibraryConfig] = None
    platform: Optional[PlatformConfig] = None
    build: Optional[BuildConfig] = None
    reporting: Optional[ReportingConfig] = None
    
    # Metadata
    source_path: Optional[str] = None
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Blueprint':
        """Create blueprint from dictionary representation."""
        
        # Parse core sections
        model_config = ModelConfig.from_dict(data.get('model', {}))
        targets_config = TargetConfig.from_dict(data.get('targets', {}))
        design_space_config = DesignSpaceConfig.from_dict(data.get('design_space', {}))
        
        # Create blueprint instance
        blueprint = cls(
            name=data['name'],
            description=data.get('description', ''),
            version=data.get('version', '1.0'),
            model=model_config,
            targets=targets_config,
            design_space=design_space_config
        )
        
        return blueprint
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate blueprint configuration."""
        errors = []
        
        # Validate required sections
        if not self.model:
            errors.append("Model configuration is required")
        
        if not self.design_space:
            errors.append("Design space configuration is required")
        
        # Validate model configuration
        if self.model:
            model_errors = self.model.validate()
            errors.extend([f"Model: {err}" for err in model_errors])
        
        # Validate design space
        if self.design_space:
            ds_errors = self.design_space.validate()
            errors.extend([f"Design Space: {err}" for err in ds_errors])
        
        # Cross-validation between sections
        cross_validation_errors = self._cross_validate()
        errors.extend(cross_validation_errors)
        
        return len(errors) == 0, errors
    
    def to_design_space(self) -> DesignSpace:
        """Convert blueprint to design space object."""
        
        if not self.design_space:
            raise ValueError("Blueprint has no design space configuration")
        
        design_space = DesignSpace(f"{self.name}_space")
        
        # Add parameters
        for param_name, param_config in self.design_space.parameters.items():
            param_def = self._create_parameter_definition(param_name, param_config)
            design_space.add_parameter(param_def)
        
        # Add constraints
        if self.design_space.constraints:
            for constraint_config in self.design_space.constraints:
                constraint = self._create_constraint(constraint_config)
                design_space.add_constraint(constraint)
        
        return design_space
    
    def to_dse_config(self) -> DSEConfig:
        """Convert blueprint to DSE configuration."""
        
        if not self.dse:
            # Use default DSE configuration
            return DSEConfig()
        
        return DSEConfig(
            strategy=self.dse.strategy,
            max_evaluations=self.dse.max_evaluations,
            objectives=[obj.name for obj in self.dse.objectives],
            objective_directions=[obj.direction for obj in self.dse.objectives]
        )
```

---

## 📊 Blueprint Templates and Patterns

### Model-Specific Templates

```yaml
# Base template for transformer models
# File: templates/transformer_base.yaml
name: "transformer_base_template"
description: "Base template for transformer architectures"

model:
  type: "transformer"
  preprocessing:
    - normalize_inputs
    - attention_mask_processing
  
design_space:
  parameters:
    attention_heads:
      type: integer
      range: [8, 16]
      default: 12
      
    hidden_size:
      type: categorical
      values: [512, 768, 1024]
      default: 768
      
    sequence_length:
      type: integer
      range: [128, 512]
      default: 256

dse:
  strategy: "adaptive"
  objectives:
    - name: "throughput_ops_sec"
      direction: "maximize"
      weight: 0.5
    - name: "latency_ms"
      direction: "minimize"
      weight: 0.5

libraries:
  transforms:
    enabled: true
    pipeline:
      - attention_optimization
      - layer_norm_fusion
      - quantization
```

### Usage Patterns

#### Model-Specific Blueprint
```yaml
# BERT-specific blueprint extending transformer template
extends: "templates/transformer_base.yaml"

name: "bert_mobile_optimization"
description: "BERT optimization for mobile deployment"

model:
  name: "bert_base"
  architecture: "bert"
  layers: 12
  # Inherits transformer-specific configuration

# Override specific parameters
design_space:
  parameters:
    layers:
      type: integer
      range: [6, 12]  # Allow layer pruning
      default: 12

targets:
  performance:
    latency_ms: 100  # Mobile latency constraint
  power:
    max_total_power_w: 2.0  # Mobile power constraint
```

### Blueprint Validation Schema

```python
class BlueprintValidator:
    """Comprehensive blueprint validation."""
    
    def __init__(self):
        self.schema = self._load_schema()
    
    def validate(self, blueprint_data: Dict[str, Any]) -> ValidationResult:
        """Validate blueprint against schema and business rules."""
        
        errors = []
        warnings = []
        
        # Schema validation
        schema_errors = self._validate_schema(blueprint_data)
        errors.extend(schema_errors)
        
        # Business rule validation
        business_errors = self._validate_business_rules(blueprint_data)
        errors.extend(business_errors)
        
        # Performance validation
        perf_warnings = self._validate_performance_targets(blueprint_data)
        warnings.extend(perf_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_business_rules(self, data: Dict[str, Any]) -> List[str]:
        """Validate business logic and constraints."""
        errors = []
        
        # Check parameter range consistency
        if 'design_space' in data and 'parameters' in data['design_space']:
            for param_name, param_config in data['design_space']['parameters'].items():
                if param_config.get('type') == 'integer':
                    range_val = param_config.get('range', [])
                    if len(range_val) == 2 and range_val[0] >= range_val[1]:
                        errors.append(f"Parameter {param_name}: invalid range {range_val}")
        
        # Check objective consistency
        if 'dse' in data and 'objectives' in data['dse']:
            total_weight = sum(obj.get('weight', 1.0) for obj in data['dse']['objectives'])
            if abs(total_weight - 1.0) > 0.01:
                errors.append(f"Objective weights sum to {total_weight}, should sum to 1.0")
        
        return errors
```

---

## 🔗 Integration with Core Systems

### Design Space Generation

```python
class DesignSpaceGenerator:
    """Generate design space from blueprint configuration."""
    
    def generate_from_blueprint(self, blueprint: Blueprint) -> DesignSpace:
        """Create design space object from blueprint."""
        
        design_space = DesignSpace(f"{blueprint.name}_space")
        
        # Process parameters
        for param_name, param_config in blueprint.design_space.parameters.items():
            param_def = self._create_parameter_definition(param_name, param_config)
            design_space.add_parameter(param_def)
        
        # Process constraints
        for constraint_config in blueprint.design_space.constraints:
            constraint = self._create_constraint(constraint_config)
            design_space.add_constraint(constraint)
        
        # Add target-derived objectives
        if blueprint.targets:
            objectives = self._extract_objectives(blueprint.targets)
            for objective in objectives:
                design_space.add_objective(objective)
        
        return design_space
    
    def _create_parameter_definition(self, name: str, 
                                   config: Dict[str, Any]) -> ParameterDefinition:
        """Create parameter definition from configuration."""
        
        param_type = ParameterType(config['type'])
        
        if param_type == ParameterType.INTEGER:
            range_vals = config['range']
            return ParameterDefinition(
                name=name,
                param_type=param_type,
                range_min=range_vals[0],
                range_max=range_vals[1],
                default=config.get('default')
            )
        elif param_type == ParameterType.CATEGORICAL:
            return ParameterDefinition(
                name=name,
                param_type=param_type,
                values=config['values'],
                default=config.get('default')
            )
        # ... handle other parameter types
```

### Library Configuration Mapping

```python
class LibraryMapper:
    """Map blueprint library configurations to library instances."""
    
    def configure_libraries(self, blueprint: Blueprint, 
                           library_registry: Dict[str, LibraryInterface]) -> Dict[str, Any]:
        """Configure libraries based on blueprint specifications."""
        
        configured_libraries = {}
        
        if not blueprint.libraries:
            return configured_libraries
        
        # Configure transforms library
        if blueprint.libraries.transforms and blueprint.libraries.transforms.enabled:
            transforms_lib = library_registry.get('transforms')
            if transforms_lib:
                config = self._create_transforms_config(blueprint.libraries.transforms)
                transforms_lib.configure(config)
                configured_libraries['transforms'] = transforms_lib
        
        # Configure hardware optimization library
        if blueprint.libraries.hw_optim and blueprint.libraries.hw_optim.enabled:
            hw_optim_lib = library_registry.get('hw_optim')
            if hw_optim_lib:
                config = self._create_hw_optim_config(blueprint.libraries.hw_optim)
                hw_optim_lib.configure(config)
                configured_libraries['hw_optim'] = hw_optim_lib
        
        # Configure analysis library
        if blueprint.libraries.analysis and blueprint.libraries.analysis.enabled:
            analysis_lib = library_registry.get('analysis')
            if analysis_lib:
                config = self._create_analysis_config(blueprint.libraries.analysis)
                analysis_lib.configure(config)
                configured_libraries['analysis'] = analysis_lib
        
        return configured_libraries
```

---

*Next: [Getting Started Guide](07_GETTING_STARTED.md)*