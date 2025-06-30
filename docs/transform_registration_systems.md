# Transform Registration Systems in QONNX, FINN, and BrainSmith

## Overview

This document details the transform registration systems across three frameworks: QONNX, FINN, and BrainSmith. Each system has evolved to address different needs, from lightweight discovery to comprehensive hardware compilation metadata management.

## QONNX Transform Registration

### Philosophy
QONNX aims to be a lightweight, general-purpose quantized neural network framework with minimal overhead for transform developers.

### Registration System

**Registry Architecture:**
- **Location**: `qonnx/transformation/registry.py`
- **Storage**: Global dictionaries `TRANSFORMATION_REGISTRY` and `TRANSFORMATION_METADATA`
- **Discovery**: Entry point based (`qonnx_transformations` group)

**Key Components:**

1. **Registry Decorator** (`@register_transformation`):
```python
@register_transformation(
    description="Fold constant expressions into initializers",
    tags=["optimization", "graph-simplification"]
)
class FoldConstants(Transformation):
    pass
```

2. **Features**:
   - Optional metadata (description, tags, author, version)
   - Automatic description extraction from docstrings
   - Metadata stored both globally and on class (`_qonnx_metadata`)
   - Entry point loading for external transforms

3. **Query Capabilities**:
   - `get_transformation(name)` - Direct lookup
   - `list_transformations()` - List all registered names
   - `get_transformation_info(name)` - Get metadata
   - `query_transformations(**filters)` - Rich querying by tags or metadata

### Current State
- **48 transforms** registered in QONNX
- Categories via tags: optimization, cleanup, graph-simplification
- No built-in hardware compilation stages
- Transforms are pure ONNX graph manipulations

## FINN Transform Registration

### Philosophy
FINN extends QONNX for FPGA hardware compilation, requiring richer metadata about compilation stages and hardware implications.

### Registration System

**Dual Registry Architecture:**
1. **QONNX Registry**: Leverages parent framework's registry
2. **FINN Plugin Registry**: Additional hardware-specific metadata

**Key Components:**

1. **Plugin Decorator** (`@transform` from `finn.plugin`):
```python
@transform(
    name="AbsorbSignBiasIntoMultiThreshold",
    stage="topology_opt",
    description="Absorb scalar bias into MultiThreshold"
)
class AbsorbSignBiasIntoMultiThreshold(Transformation):
    pass
```

2. **FINN Plugin Registry** (`finn/plugin/registry.py`):
   - Singleton pattern for global access
   - Flat storage: `{type:name -> metadata}`
   - Permissive registration (no validation failures)
   - Rich query interface matching BrainSmith

3. **Adapter Pattern** (`finn/plugin/adapters.py`):
   - Registers with both QONNX and FINN registries
   - Validates stages (cleanup, topology_opt, kernel_opt, dataflow_opt)
   - Stores `_plugin_metadata` on class
   - Handles registration failures gracefully

4. **Additional Plugin Types**:
   - **Kernels**: Hardware accelerator implementations
   - **Backends**: HLS/RTL implementations for kernels

### Current State
- **112 transforms** in FINN
- Hardware compilation stages clearly defined
- Kernel/backend discovery for hardware generation
- Categories: streamline, fpgadataflow, qonnx extensions

## BrainSmith Transform Registration

### Philosophy
BrainSmith provides a unified plugin system that both creates native transforms AND discovers transforms from other frameworks (QONNX, FINN). It adds hardware compilation metadata and provides specialized transform types. Steps are a separate, higher-level concept that orchestrate collections of transforms.

### Registration System

**Unified Plugin Architecture:**
- **Core**: `brainsmith/plugin/core.py` - UnifiedRegistry
- **Discovery**: Automatic for QONNX/FINN transforms
- **Registration**: Native decorators for BrainSmith transforms

**Key Components:**

1. **Native Transform Registration** (`@transform` decorator):
   - **Regular transforms**: Use `stage` parameter
   - **Kernel inference transforms**: Use `kernel` parameter
   - Mutually exclusive - cannot specify both
   
```python
# Regular transform
@transform(
    name="ExpandNorms",
    stage="topology_opt",
    description="Expand LayerNorms/RMSNorms into functional components"
)
class ExpandNorms(Transformation):
    pass

# Kernel inference transform
@transform(
    name="InferLayerNorm", 
    kernel="LayerNorm",
    description="Convert FuncLayerNorm to LayerNorm hardware operations"
)
class InferLayerNorm(Transformation):
    pass
```

2. **Transform Discovery** (`brainsmith/plugin/discovery.py`):
   - Discovers QONNX transforms via `qonnx_discovery.py`
   - Discovers FINN transforms via `finn_discovery.py`
   - Adds BrainSmith-specific metadata (stages, hardware info)

3. **Unified Registry**:
   - Single registry for all plugin types
   - Stores ALL transforms with framework prefixes (qonnx:, finn:, brainsmith:)
   - Conflict detection for same-named transforms
   - Optional prefix support for unique names

4. **Transform Resolution** (`brainsmith/steps/transform_resolver.py`):
   - Resolves transform names to classes
   - Handles framework prefixes (qonnx:, finn:, brainsmith:)
   - **NEW**: Optional prefix for unique names
   - Conflict detection with helpful error messages

5. **Steps** (Separate Concept):
   - Steps are NOT transforms but orchestrators
   - Use `@finn_step` decorator
   - Compose multiple transforms with logic
```python
@finn_step(
    name="prepare_for_synthesis",
    transforms=[
        "GiveUniqueNodeNames",
        "qonnx:InferShapes",  # Explicit framework prefix
        "Streamline"  # Multi-transform macro
    ],
    description="Prepare model for hardware synthesis"
)
def prepare_for_synthesis(model, cfg, transforms):
    # Step implementation with custom logic
    for transform in transforms:
        model = transform.apply(model)
        # Additional logic between transforms
    return model
```

6. **BrainSmith Transform Types**:
   - **Regular transforms**: Standard graph manipulations with stages
   - **Kernel inference transforms**: Convert patterns to hardware kernels
   - Both registered in unified registry with framework="brainsmith"

### Current State
- **255+ total transforms** (48 QONNX + 112 FINN + 95+ BrainSmith native)
- **Native BrainSmith transforms** in various categories:
  - `topology_opt/`: ExpandNorms
  - `kernel_opt/`: SetPumpedCompute, TempShuffleFixer
  - `model_specific/`: RemoveBertHead, RemoveBertTail
  - `metadata/`: ExtractShellIntegrationMetadata
  - `graph_cleanup/`: RemoveIdentity
  - Kernel inference transforms in `kernels/*/infer_*.py`
- **22 naming conflicts** between frameworks
- **116+ unique transforms** usable without prefixes
- Steps provide higher-level orchestration of transforms

## Comparison Summary

| Aspect | QONNX | FINN | BrainSmith |
|--------|--------|------|------------|
| **Transforms** | Native transforms | Native transforms | Native + discovered |
| **Registration** | `@register_transformation` | `@transform` | `@transform` |
| **Storage** | Global dicts | Plugin Registry | Unified Registry |
| **Discovery** | Entry points | Entry points + decorators | Auto + native decorators |
| **Metadata** | Tags, description | Stages, categories | Stages + kernel inference |
| **Validation** | None | Soft warnings | Soft warnings |
| **Query** | Tag-based | Rich queries | Cross-framework |
| **Conflicts** | N/A | N/A | Smart resolution |
| **Transform Types** | Regular only | Regular only | Regular + kernel inference |
| **Steps** | N/A | N/A | Transform orchestration |

## Design Evolution

1. **QONNX**: Started simple - decorators for basic registration
2. **FINN**: Added hardware stages and dual registration
3. **BrainSmith**: Unified system with native transforms + discovery

The evolution shows different philosophies:
- QONNX: Pure transform registration for graph manipulation
- FINN: Hardware-aware transforms with compilation stages
- BrainSmith: Comprehensive system with multiple transform types

## Key Distinctions

1. **Transform Types**:
   - **Regular Transforms**: Graph manipulations with stages (all frameworks)
   - **Kernel Inference Transforms**: Pattern to hardware conversion (BrainSmith only)
   - **Steps**: Orchestration of transforms with custom logic (BrainSmith only)

2. **BrainSmith's Unique Features**:
   - Creates native transforms for hardware-specific operations
   - Discovers and enriches transforms from QONNX/FINN
   - Provides kernel inference transforms for hardware mapping
   - Unified registry with conflict resolution
   - Step-based orchestration layer

3. **Registration Flexibility**:
   - BrainSmith's `@transform` decorator is overloaded:
     - With `stage`: Regular transform
     - With `kernel`: Kernel inference transform
     - Cannot have both (mutually exclusive)

## Key Innovation: Optional Prefixes

BrainSmith's latest enhancement allows:
- **Unique names**: Use without framework prefix (116 available)
- **Conflicts**: Require explicit prefix (22 conflicts)
- **User-friendly**: Reduces typing for common cases
- **Clear errors**: Guides users when disambiguation needed

This makes the common case simple while maintaining clarity for conflicts.