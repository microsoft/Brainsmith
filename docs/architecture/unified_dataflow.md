# Unified Data Flow Architecture

## Overview

This document describes the unified data flow architecture for BrainSmith, showing how blueprint data flows through the system to FINN builds.

## Current Architecture

### Three Separate Data Paths

1. **Configuration Path**: Blueprint config → finn_config → DataflowBuildConfig
2. **Steps Path**: Blueprint steps → ExecutionTree → Step functions  
3. **Kernel Path**: Blueprint kernels → Step metadata → infer_kernels step

### Problems with Current Approach

- Kernel data takes a convoluted path through step metadata
- Three different mechanisms for three types of data
- Difficult to trace data flow
- Prone to disconnects between components

## Proposed Unified Architecture

### Core Principle: Single Data Container

All blueprint data flows through a unified `BuildContext` that carries:
- Configuration (FINN parameters)
- Steps (execution plan)
- Domain-specific metadata (kernels, constraints, etc.)

### Implementation

```python
@dataclass
class BuildContext:
    """Unified container for all build data."""
    # Core FINN configuration
    finn_config: DataflowBuildConfig
    
    # Execution plan
    steps: List[Union[str, Callable]]
    
    # Domain-specific metadata
    metadata: Dict[str, Any]  # Includes kernels, design constraints, etc.
    
    # Runtime state
    runtime: Dict[str, Any]  # For passing data between steps
```

### Data Flow

```
Blueprint YAML
     ↓
BlueprintParser.parse()
     ↓
BuildContext (unified)
     ├── finn_config: DataflowBuildConfig
     ├── steps: List[step]
     └── metadata: {
           "kernels": [...],
           "design_space": {...},
           "constraints": {...}
         }
     ↓
ExecutionTree uses BuildContext
     ↓
Executor passes BuildContext
     ↓
Steps access BuildContext.metadata
```

### Benefits

1. **Single Source of Truth**: All data in one container
2. **Extensible**: Easy to add new metadata types
3. **Traceable**: Clear data flow path
4. **Type-Safe**: Strong typing with dataclasses
5. **FINN-Compatible**: Still creates standard DataflowBuildConfig

## Migration Strategy

### Phase 1: Add BuildContext (Parallel)
- Create BuildContext alongside existing flow
- Gradually migrate steps to use it
- Maintain backward compatibility

### Phase 2: Deprecate Old Paths
- Mark old data paths as deprecated
- Update all steps to use BuildContext
- Add migration warnings

### Phase 3: Remove Legacy Code
- Remove old data flow paths
- Simplify explorer and executor
- Clean up step interfaces

## Example Usage

### Blueprint
```yaml
# All data unified in blueprint
metadata:
  kernels:
    - LayerNorm: layernorm_rtl
    - MVAU: mvau_hls
  constraints:
    max_dsp: 1000
    max_bram: 500

finn_config:
  synth_clk_period_ns: 5.0
  board: U250

design_space:
  steps:
    - cleanup
    - infer_kernels  # Reads from context.metadata.kernels
    - optimize       # Reads from context.metadata.constraints
```

### Step Implementation
```python
@step(name="infer_kernels")
def infer_kernels_step(model, context: BuildContext):
    """Infer kernels using unified context."""
    kernels = context.metadata.get("kernels", {})
    
    for kernel_name, backend in kernels.items():
        # Apply inference
        transform = get_transform_for_kernel(kernel_name)
        model = model.transform(transform)
    
    # Store results in runtime state
    context.runtime["inferred_kernels"] = list(kernels.keys())
    
    return model
```

### Executor Usage
```python
class Executor:
    def execute(self, context: BuildContext):
        """Execute build with unified context."""
        # FINN still gets standard DataflowBuildConfig
        finn_config = context.finn_config
        
        # But steps can access full context
        for step in context.steps:
            if callable(step):
                model = step(model, context)
            else:
                # Legacy FINN steps get just the config
                model = finn_step(model, finn_config)
```

## Advantages for BrainSmith

1. **Clean Integration**: FINN sees standard DataflowBuildConfig
2. **Rich Metadata**: Steps can access all blueprint data
3. **Evolution Path**: Can add new features without changing FINN
4. **Better Debugging**: Single context to inspect
5. **Testability**: Easy to create test contexts

## Implementation Priority

1. **Immediate**: Fix kernel_selections in DataflowBuildConfig ✓
2. **Short-term**: Create BuildContext prototype
3. **Medium-term**: Migrate critical steps
4. **Long-term**: Full unified architecture

This architecture provides a clean path forward while maintaining compatibility with FINN's expectations.