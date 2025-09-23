# Deprecated Dataflow Components

This directory contains deprecated components that have been replaced by newer patterns.

## Deprecated Files:

### Original Mutable Models
- **kernel_model.py** - Original mutable KernelModel implementation (replaced by immutable models)
- **kernel_model_v3.py, kernel_model_v3_simple.py** - Previous attempts at solving staleness issues
- **kernel_model_validation.py** - Validation logic now embedded in factory functions

### Interface Classes (Deprecated 2025-09-22)
- **base_interface.py** - Base class for mutable interfaces with CSDF support
- **input_interface.py** - Mutable InputInterface with setters and caching
- **output_interface.py** - Mutable OutputInterface with streaming rate setters

### Tiling Complexity
- **tiling_spec.py** - Complex TilingSpec class hierarchy (replaced by simple `_resolve_tiling()` method)
- **tiling_strategy.py** - Strategy pattern no longer needed with simplified approach
- **tiling_functions.py** - Functionality consolidated into AutoHWCustomOp

### Schema Files (Deprecated 2025-09-23)
- **input_definition.py** - Original InputSchema location (moved to schemas.py)
- **output_definition.py** - Original OutputSchema location (moved to schemas.py)
- **kernel_definition.py** - Original KernelSchema location (moved to schemas.py)

## Migration Notes:

The new architecture uses:
1. **Immutable models** from `models.py` - created fresh via factory functions
2. **Consolidated schemas** in `schemas.py` - with InterfaceSchema base class
3. **Simplified tiling** in `AutoHWCustomOp._resolve_tiling()`
4. **No caching or mutable state** - models are always consistent

### Key Changes:
- All schemas now in `schemas.py` with improved design
- `InputInterface` → `InputModel` (via `create_input_model()`)
- `OutputInterface` → `OutputModel` (via `create_output_model()`)
- BaseModel class removed (not used by immutable models)
- InterfaceSchema provides common behavior for Input/OutputSchema

These files are kept for reference during migration but should not be used for new code.