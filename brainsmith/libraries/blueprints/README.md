# BrainSmith Blueprint Libraries

This directory contains YAML blueprint collections for FPGA accelerator design space exploration. Blueprints define configurable accelerator templates with parameter spaces for optimization.

## Directory Structure

```
libraries/blueprints/
├── basic/          # Simple, beginner-friendly blueprints
├── advanced/       # Complex, high-performance blueprints
├── experimental/   # Research and prototype blueprints
└── custom/         # User-defined blueprints
```

## Blueprint Format

Each blueprint is a YAML file with the following structure:

```yaml
name: "blueprint_name"
description: "Blueprint description"
version: "1.0"
category: "basic|advanced|experimental|custom"

# Base configuration
model_type: "cnn|mobilenet|transformer|..."
target_platform: "zynq|ultrascale|versal"
optimization_level: "basic|moderate|aggressive"

# Configurable parameters for DSE
parameters:
  parameter_name:
    range: [min, max]           # Numeric range
    values: [val1, val2, val3]  # Discrete choices  
    default: value              # Default value
    description: "Parameter description"

# Fixed configuration
fpga_part: "device_part_number"
board: "board_name"
# ... other fixed settings

# Performance targets
targets:
  throughput_fps: 30
  power_budget_w: 5.0
  accuracy_drop_max: 0.05
```

## Usage

Blueprints are managed by the `BlueprintManager` in the DSE infrastructure:

```python
from brainsmith.infrastructure.dse import BlueprintManager

# Create manager
manager = BlueprintManager()

# Load a blueprint
config = manager.load_blueprint("cnn_accelerator")

# Create design points
design_point = manager.create_design_point(
    "cnn_accelerator", 
    {"conv_pe": 16, "fc_pe": 8}
)

# Get parameter space for DSE
param_space = manager.get_blueprint_parameter_space("cnn_accelerator")
```

## Available Blueprints

### Basic Category
- **cnn_accelerator**: Simple CNN accelerator for image classification
  - Target: Basic image classification tasks
  - Platform: Pynq-Z1
  - Parameters: Folding factors, memory mode, quantization

### Advanced Category  
- **mobilenet_accelerator**: Optimized for MobileNet architectures
  - Target: Efficient CNN inference
  - Platform: Ultra96-V2
  - Parameters: Advanced folding, memory hierarchy, optimization flags

## Creating Custom Blueprints

1. Create a new YAML file in the appropriate category directory
2. Follow the blueprint format specification
3. Test with the BlueprintManager
4. Add documentation for parameters and expected usage

## Blueprint Guidelines

- **Parameters**: Include meaningful parameter ranges for exploration
- **Defaults**: Provide sensible default values
- **Documentation**: Document each parameter's purpose and impact
- **Validation**: Ensure parameter combinations are feasible
- **Targets**: Set realistic performance and resource targets

## Integration with DSE

Blueprints integrate seamlessly with the DSE engine:

```python
from brainsmith.infrastructure.dse import parameter_sweep

# Run parameter sweep on a blueprint
results = parameter_sweep(
    blueprint_name="cnn_accelerator",
    parameter_space={"conv_pe": [8, 16, 32], "fc_pe": [4, 8, 16]},
    evaluation_function=my_eval_function
)
```

This enables systematic exploration of accelerator design spaces using predefined, validated templates.