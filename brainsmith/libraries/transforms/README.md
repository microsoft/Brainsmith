# Transform Libraries

This directory contains transformation implementations for model processing and compilation pipeline steps.

## Structure

### Pipeline Steps
- **`steps/`** - Transformation steps for the compilation pipeline
  - `bert.py` - BERT-specific transformations
  - `cleanup.py` - Model cleanup and optimization
  - `conversion.py` - Format conversion transformations
  - `hardware.py` - Hardware-specific transformations
  - `metadata.py` - Metadata handling transformations
  - `optimizations.py` - General optimization passes
  - `streamlining.py` - Model streamlining transformations
  - `validation.py` - Validation and verification steps

### Model Operations
- **`operations/`** - Direct model transformation operations
  - `convert_to_hw_layers.py` - Hardware layer conversion
  - `expand_norms.py` - Normalization layer expansion
  - `shuffle_helpers.py` - Channel shuffle operations

### Extension Points
- **`contrib/`** - Stakeholder-contributed transformation implementations

## Usage

### Using Existing Transforms
```python
from brainsmith.libraries.transforms.steps import optimizations
from brainsmith.libraries.transforms.operations import convert_to_hw_layers

# Apply optimization transformations
optimized_model = optimizations.apply_optimizations(model)

# Convert to hardware layers
hw_model = convert_to_hw_layers.convert(model)
```

### Pipeline Integration
```python
from brainsmith.core.api import forge

# Transforms are automatically applied during compilation
result = forge(
    model=model,
    blueprint="efficient_inference",
    transforms=["cleanup", "optimizations", "hardware"]
)
```

### Adding New Transforms
1. Create transformation step following existing patterns
2. Implement operation functions with proper error handling
3. Add to transformation registry for automatic discovery
4. Include comprehensive tests and documentation

## Integration
- Transforms are part of the compilation pipeline
- Compatible with FINN model formats
- Support for both model-level and node-level operations
- Automatic validation and error reporting