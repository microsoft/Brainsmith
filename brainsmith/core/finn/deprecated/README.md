# Deprecated FINN Components

This directory contains deprecated components that have been replaced by the new factory-based approach in `auto_hw_custom_op.py`.

## Deprecated Files:

- **auto_hw_custom_op.py** (old version) - Original implementation with stored kernel models that could become stale

## Migration Notes:

The new `auto_hw_custom_op.py` (formerly auto_hw_custom_op_immutable.py):
- Creates fresh kernel models on every access via `create_kernel_model()`
- Eliminates staleness issues when nodeattrs change during FINN transformations
- Uses immutable model pattern from `brainsmith.core.dataflow.models`
- Consolidates tiling resolution in `_resolve_tiling()` method

Key changes for subclasses:
- No more stored `self.kernel_model` 
- Access models via `self.create_kernel_model(model)`
- Models are immutable snapshots, never modified in place

This change follows the Arete principle: correctness through simplicity.