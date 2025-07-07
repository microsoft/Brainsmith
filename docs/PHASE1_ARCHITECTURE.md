# Phase 1: Design Space Constructor - Simplified Architecture

## Overview

Phase 1 transforms user inputs (ONNX model + Blueprint YAML) into a `DesignSpace` object with minimal validation.

## Philosophy

- **No babysitting**: Trust engineers to provide valid input
- **Let Python fail naturally**: No custom error messages
- **Minimal validation**: Only check critical safety constraints

## Component Overview

### Core Components (847 lines total)

1. **parser.py** (~243 lines): Parse Blueprint YAML into DesignSpace
2. **validator.py** (~70 lines): Check only model existence and combination limits
3. **forge.py** (~93 lines): Simple orchestration of parsing and validation
4. **data_structures.py** (~369 lines): Core data models
5. **exceptions.py** (~30 lines): Simple exception classes
6. **__init__.py** (~42 lines): Public API exports

### Data Flow

```
Blueprint YAML → Parser → DesignSpace → Validator → Output
                   ↓
            Plugin Registry
             (for kernels)
```

## Key Changes from Original

1. **Removed all validation helpers** - Let KeyError/AttributeError happen naturally
2. **Removed helpful error messages** - Engineers can read stack traces
3. **Removed deprecated fields** - No backward compatibility warnings
4. **Removed optimization** - Premature optimization is evil
5. **Removed logging** - Less noise

## API Usage

```python
from brainsmith.core.phase1 import forge

# Simple usage - may throw Python errors if invalid
design_space = forge("model.onnx", "blueprint.yaml")
```

## Error Handling

- **Invalid YAML**: `yaml.YAMLError`
- **Missing fields**: `KeyError`
- **Wrong types**: `TypeError` or `AttributeError`
- **Invalid enums**: `ValueError`
- **Missing model**: `ValidationError`
- **Too many combinations**: `ValidationError`

## Minimal Validation

The validator only checks:
1. Model file exists
2. Total combinations < configured limit

Everything else is trusted to be correct or fail naturally.