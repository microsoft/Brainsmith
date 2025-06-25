# Brainsmith Core V3

A clean reimplementation of the Brainsmith DSE (Design Space Exploration) system with improved separation of concerns, extensibility, and simplicity.

## Overview

Core V3 implements a three-phase architecture:

1. **Design Space Constructor** - Transforms ONNX models and Blueprint YAML files into structured design spaces
2. **Design Space Explorer** - Systematically explores design spaces to find optimal configurations
3. **Build Runner** - Executes builds with preprocessing, backend compilation, and postprocessing

## Key Improvements

- **Cleaner Separation** - Each phase has a single, well-defined responsibility
- **Simpler Configuration** - Blueprints use intuitive YAML without unnecessary complexity
- **Better Extensibility** - Hook-based architecture for future enhancements
- **Explicit Behavior** - No hidden magic or implicit assumptions

## Quick Start

```python
from brainsmith.core_v3 import forge, explore

# Create design space from model and blueprint
design_space = forge(
    model_path="models/bert.onnx",
    blueprint_path="blueprints/bert_exploration.yaml"
)

# Explore all configurations
results = explore(design_space)

# Get best configuration
print(f"Best config: {results.best}")
```

## Blueprint Format

```yaml
version: "3.0"
hw_compiler:
  kernels:
    - "MatMul"                          # Auto-import all backends
    - ("Softmax", ["hls", "rtl"])      # Specific backends
    - ["LayerNorm", "RMSNorm"]          # Mutually exclusive
    
  transforms:
    - "quantization"
    - [~, "folding"]                    # Optional
    - ["stream_v1", "stream_v2"]        # Mutually exclusive

search:
  strategy: "exhaustive"
  constraints:
    - metric: "lut_utilization"
      operator: "<="
      value: 0.85

global:
  output_stage: "rtl"
  working_directory: "./builds"
```

## Development Status

🚧 **Currently Implementing Phase 1**

See [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) for detailed progress.

## Documentation

- [Implementation Plan](implementation_plan.md) - Detailed development roadmap
- [Phase 1 Design](../../docs/dse_v3/phase1_design_space_constructor.md) - Design Space Constructor details
- [Architecture Overview](../../ai_cache/designs/brainsmith_core_v3_architecture.md) - High-level architecture

## Contributing

This is a clean reimplementation. Please:
1. Follow the implementation checklist
2. Write tests for all new code
3. Keep the design simple and explicit
4. Document all public APIs