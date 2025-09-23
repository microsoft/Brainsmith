# Cyclo-Static Dataflow (CSDF) Design Analysis

## Overview

This document captures the key innovations and design decisions in Brainsmith's CSDF implementation before simplification to pure SDF. While CSDF proved to be over-engineered for current needs, the design contains valuable insights for future hardware dataflow modeling.

## What is CSDF?

Cyclo-Static Dataflow extends Synchronous Dataflow (SDF) by allowing token production/consumption rates to vary in a cyclic pattern. Instead of fixed rates, CSDF actors cycle through phases, each with potentially different dataflow characteristics.

### Example Use Case
```
Phase 0: Read 8 tokens, produce 1 (reduction)
Phase 1: Read 1 token, produce 4 (expansion)  
Phase 2: Read 4 tokens, produce 4 (pass-through)
[Repeat cycle]
```

## Key Innovations in Brainsmith's CSDF

### 1. RaggedShape Type System

```python
Shape = Tuple[int, ...]
RaggedShape = Union[Shape, List[Shape]]
```

**Innovation**: Elegantly represents both SDF and CSDF with a single type:
- Single shape → SDF (backwards compatible)
- List of shapes → CSDF (one per phase)

**Example**:
```python
# SDF: Constant block size
block_dims = (8, 16, 32)  

# CSDF: Variable block sizes per phase
block_dims = [(8, 16, 32), (4, 8, 16), (2, 4, 8)]
```

### 2. Phase-Aware Skip Probability

```python
skip_prob: List[float] = field(default_factory=list)  # Sparsity per phase
```

**Innovation**: Models sparse computation patterns that vary cyclically:
- Phase 0: skip_prob=0.0 (dense computation)
- Phase 1: skip_prob=0.8 (sparse, skip 80% of blocks)
- Phase 2: skip_prob=0.5 (medium sparsity)

**Hardware Benefit**: Enables power gating and resource optimization based on expected sparsity patterns.

### 3. Automatic Phase Detection

```python
@property
def n_phases(self) -> int:
    """Number of CSDF phases."""
    if isinstance(self.block_dims, list):
        return len(self.block_dims)
    return 1  # SDF has single phase
```

**Innovation**: Seamlessly distinguishes between SDF and CSDF without explicit mode flags.

### 4. Phase-Aware Performance Modeling

The design allows different performance characteristics per phase:
- Bandwidth requirements
- Initiation intervals
- Resource utilization

This models real hardware behaviors like:
- Pipeline fill/drain stages
- Burst vs. scattered memory access
- Variable computation intensity

## Hardware-Specific Benefits

### 1. Pipeline Priming/Draining
CSDF naturally models non-steady-state behaviors:
```
Phase 0: Pipeline fill (read more, produce less)
Phase 1-N: Steady state  
Phase N+1: Pipeline drain (read less, produce more)
```

### 2. Resource Time-Multiplexing
Different phases can share hardware resources:
```
Phase 0: Use multiply-accumulate units
Phase 1: Use activation function units
Phase 2: Use pooling units
```

### 3. Memory Access Pattern Optimization
```
Phase 0: Burst read from DRAM
Phase 1: Stream from on-chip buffer
Phase 2: Scatter write results
```

## Why CSDF Was Over-Engineered

Despite these innovations, CSDF added complexity that wasn't justified:

1. **Most kernels are pure SDF**: Hardware accelerators typically have fixed, regular patterns
2. **Validation complexity**: Multi-phase constraints are hard to verify
3. **State management overhead**: Tracking phase transitions complicates implementations
4. **Limited tool support**: FINN/QONNX frameworks assume SDF models

## Lessons Learned

### What to Keep
1. **Phase concept**: Useful for modeling initialization/cleanup
2. **Variable patterns**: Some kernels do have mode-dependent behavior
3. **Skip probability**: Sparsity handling is valuable

### What to Simplify  
1. **RaggedShape**: Use simple Shape everywhere
2. **Phase tracking**: Remove n_phases, is_csdf properties
3. **Complex validation**: Single-phase validation is sufficient

## Future Considerations

If CSDF support is needed later:

1. **Create separate CSDF classes** rather than mixing with SDF
2. **Make phases explicit** in the model (not inferred from shapes)
3. **Focus on specific use cases** (e.g., video codecs with I/P/B frames)
4. **Build tooling support** before adding language features

## Conclusion

The CSDF design in Brainsmith represents sophisticated thinking about hardware dataflow patterns. While simplification to pure SDF is the right choice for now, this analysis preserves the key insights for future reference. The main lesson: start simple (SDF) and add complexity (CSDF) only when real use cases demand it.

### Key Takeaway
> "The best design is not when there is nothing more to add, but when there is nothing left to take away." - Antoine de Saint-Exupéry

CSDF support can always be added back when needed, but carrying unused complexity violates the Arete principle.