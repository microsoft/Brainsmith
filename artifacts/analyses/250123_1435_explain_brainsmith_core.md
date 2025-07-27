# Brainsmith Core Module: Architecture and Design Explanation

## Executive Summary

Brainsmith Core is an FPGA compilation framework that transforms ONNX neural network models through a design space exploration pipeline. Its key innovation is a **segment-based execution tree** that dramatically reduces redundant computation by sharing common transformation paths.

The system achieves high performance through:
- Segment-based execution trees that group operations between branch points
- Zero-discovery plugin architecture with O(1) access times
- Clean separation between parsing, representation, and execution layers
- Pragmatic integration with external frameworks (QONNX/FINN)

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Key Components](#key-components)
4. [Design Innovations](#design-innovations)
5. [Workflow and Usage](#workflow-and-usage)
6. [Plugin System](#plugin-system)
7. [Performance Characteristics](#performance-characteristics)
8. [Design Philosophy](#design-philosophy)

## System Overview

### Purpose
Brainsmith automates the exploration of different compilation strategies for neural networks targeting FPGAs. Instead of manually trying different optimization passes, users define a "blueprint" that describes the design space, and Brainsmith automatically explores all valid combinations.

### High-Level Flow
```
ONNX Model + Blueprint YAML → Parse → Build Execution Tree → Explore → Optimized Models
```

### Key Problem Solved
Without Brainsmith, optimizing a neural network for FPGA deployment requires:
- Manual application of dozens of transformation passes
- Trial and error to find the best combination
- Redundant re-execution of common transformation prefixes

Brainsmith automates this process while minimizing redundant computation through its segment-based architecture.

## Core Architecture

### Layered Design
```
┌─────────────────────────────────────┐
│         User Interface              │
│    forge(), print_tree_summary()    │
├─────────────────────────────────────┤
│        Blueprint Parser             │
│    YAML → DesignSpace conversion    │
├─────────────────────────────────────┤
│         Design Space                │
│   Intermediate representation       │
├─────────────────────────────────────┤
│       Execution Tree                │
│    Segment-based structure          │
├─────────────────────────────────────┤
│          Explorer                   │
│    Tree execution engine            │
├─────────────────────────────────────┤
│       Plugin Registry               │
│  Transforms, kernels, backends      │
└─────────────────────────────────────┘
```

### Module Structure
```
brainsmith/core/
├── __init__.py          # Public API
├── forge.py            # Main entry point
├── blueprint_parser.py # YAML parsing
├── design_space.py     # Intermediate representation
├── execution_tree.py   # Tree structure
├── explorer/           # Execution subsystem
│   ├── explorer.py    # Main execution
│   ├── executor.py    # Segment execution
│   ├── finn_adapter.py # FINN integration
│   └── types.py       # Data structures
└── plugins/           # Plugin system
    ├── registry.py    # Core registry
    ├── decorators.py  # Registration
    └── framework_adapters.py # External frameworks
```

## Key Components

### 1. Blueprint (Input)
A YAML file that declaratively specifies:
- **Transform stages**: Named groups of transformation options
- **Build pipeline**: Sequence of operations referencing stages
- **Configuration**: Target platform, optimization goals

Example:
```yaml
design_space:
  transforms:
    cleanup:
      - RemoveIdentityOps
      - [FoldConstants, ~]  # Optional
    optimize:
      - [ConvertToChannelLast, ConvertToChannelFirst]  # Exclusive choice
```

### 2. DesignSpace (Intermediate Representation)
Clean data structure representing:
- Global configuration settings
- Transform stages with their options
- Kernel definitions (advanced feature)
- Build pipeline with stage references

Purpose: Validates and normalizes the blueprint into a format suitable for tree building.

### 3. ExecutionNode (Tree Structure)
The core innovation - nodes represent **segments** not individual steps:
```python
class ExecutionNode:
    segment_steps: List[str]      # Steps in this segment
    branch_decision: str          # How we got here
    children: List[ExecutionNode] # Branches from here
    status: str                   # pending/completed/failed
    output_dir: str              # Where results are stored
```

### 4. Explorer (Execution Engine)
Executes the tree using:
- Depth-first traversal with a stack
- Full directory copying at branch points (FINN requirement)
- Artifact sharing between parent and child segments
- Simple caching based on output file existence

### 5. Plugin Registry
High-performance plugin management:
- **Zero-discovery**: Plugins register at decoration time
- **O(1) access**: Direct dictionary lookups
- **Multi-index**: Separate indexes for different query patterns
- **Framework support**: Unified interface for QONNX/FINN/custom

## Design Innovations

### 1. Segment-Based Execution Trees

**Traditional Approach**: One node per transformation
```
Root → Transform1 → Transform2 → Transform3
     ↘ Transform1 → Transform2 → Transform4
```
Problem: Transform1 and Transform2 executed twice

**Brainsmith Approach**: Segments between branch points
```
Root[Transform1, Transform2] → Branch1[Transform3]
                            ↘ Branch2[Transform4]
```
Result: Common prefix executed once

### 2. Prefix Sharing Optimization

The segment construction algorithm:
1. Accumulate linear steps (no branches)
2. When branch point detected:
   - Flush accumulated steps to current segment
   - Create child segments for each branch
   - Continue from branch point

This reduces execution from O(paths × steps) to O(unique segments).

### 3. Zero-Discovery Plugin Architecture

Traditional plugin systems:
```python
# Expensive discovery
plugins = discover_plugins("./plugins/")  # Filesystem scan
for plugin in plugins:
    load_and_register(plugin)  # Dynamic import
```

Brainsmith approach:
```python
# Registration at decoration time
@transform(name="MyTransform", stage="cleanup")
class MyTransform:
    pass
# Already registered when decorator executes
```

Benefits:
- No startup overhead
- Predictable behavior
- IDE-friendly (full type information)

### 4. Direct Class Access

Plugin collections return actual classes:
```python
# Direct access - no wrapper
transform_class = transforms.RemoveIdentityOps
instance = transform_class(model)

# Not a wrapper function
# Not a proxy object
# Just the class itself
```

## Workflow and Usage

### 1. Define Blueprint
```yaml
version: "4.0"
name: "bert-optimization"

global_config:
  output_stage: synthesize_bitstream
  
design_space:
  transforms:
    cleanup:
      - RemoveIdentityOps
      - RemoveUnusedTensors
    quantize:
      - [Int8Quantization, Int4Quantization]
      
build_pipeline:
  steps:
    - "{cleanup}"
    - "{quantize}"
    - GenerateHLS
```

### 2. Create Execution Tree
```python
from brainsmith.core import forge, print_tree_summary

design_space, tree = forge("bert.onnx", "bert_blueprint.yaml")
print_tree_summary(tree)
```

Output:
```
root (2 steps) [pending]
├── cleanup_quantize_0 (1 step) [pending]
└── cleanup_quantize_1 (1 step) [pending]
```

### 3. Execute Exploration
```python
from brainsmith.core.explorer import explore_execution_tree

result = explore_execution_tree(
    tree, 
    "bert.onnx",
    "output/",
    finn_config
)
```

### 4. Analyze Results
Each segment produces:
- Transformed ONNX model
- Execution logs
- Performance metrics
- Build artifacts (if applicable)

## Plugin System

### Registration Pattern
```python
@transform(name="OptimizeGEMM", stage="optimize")
class OptimizeGEMM(Transformation):
    def apply(self, model):
        # Implementation
        pass

@backend(name="Conv_HLS", kernel="Conv", language="hls")
class ConvHLS(ConvKernel, HLSBackend):
    def generate(self):
        # HLS generation
        pass
```

### Access Patterns
```python
# Direct access
transforms.OptimizeGEMM

# Framework-qualified
transforms.finn.ConvertToFINN

# Query-based
backends.find(kernel="Conv", language="hls")

# Collections for blueprints
collections = load_blueprint_plugins("model.yaml")
```

### Multi-Index Architecture
The registry maintains indexes for:
- Direct name lookup: O(1)
- Framework-specific access: O(1)
- Stage/category filtering: O(1)
- Attribute-based queries: O(1) + intersection

## Performance Characteristics

### Time Complexity
- Tree building: O(stages × avg_branches)
- Segment execution: O(unique_segments)
- Plugin lookup: O(1)
- Without segments: O(total_paths × steps_per_path)

### Space Complexity
- Execution tree: O(total_segments)
- Plugin registry: O(plugins × indexes)
- Artifact storage: O(segments × model_size)

### Optimization Metrics
For a typical neural network with 10 branch points and 3 options each:
- Traditional: 3^10 = 59,049 full executions
- Brainsmith: ~100-300 segment executions
- Reduction: 99.5%+ fewer transform applications

## Design Philosophy

### 1. Performance First
Every architectural decision prioritizes runtime efficiency:
- Segment-based trees minimize computation
- Zero-discovery plugins eliminate startup overhead
- Direct class access avoids wrapper indirection

### 2. Clean Abstractions
Clear separation of concerns:
- Parsing is isolated from execution
- FINN workarounds contained in adapter
- Plugin system independent of core logic

### 3. Pragmatic Engineering
Real-world usability over theoretical purity:
- Full directory copies for FINN compatibility
- Simple file-based caching
- Explicit imports over magic discovery

### 4. User-Centric Design
Developer experience matters:
- Declarative YAML blueprints
- Clear error messages
- Visual tree summaries
- Predictable behavior

## Common Patterns and Best Practices

### 1. Blueprint Design
- Keep stages focused (3-5 transforms max)
- Use exclusive choices `[A, B]` for incompatible options
- Use optional syntax `[T, ~]` for experiments
- Reference stages in pipeline with `{stage_name}`

### 2. Custom Transforms
```python
# Always specify stage
@transform(name="MyTransform", stage="optimize")
class MyTransform(Transformation):
    def apply(self, model):
        # Modify model in-place
        return model
```

### 3. Exploration Strategy
- Start with small design spaces
- Use `max_combinations` to limit explosion
- Enable caching for iterative development
- Monitor segment efficiency metrics

### 4. Integration Points
- Custom transforms via plugin decorators
- External tools via build steps
- Framework adapters for new backends
- Result processors for analysis

## Conclusion

Brainsmith Core exemplifies thoughtful system design:
- **Innovation**: Segment-based trees solve a real performance problem
- **Clarity**: Clean abstractions make the system understandable
- **Performance**: Every decision optimizes for production use
- **Pragmatism**: Real-world compatibility over theoretical purity

The architecture achieves its goal of making FPGA compilation exploration both efficient and accessible, turning an exponential problem into a tractable one through clever tree segmentation and shared computation.