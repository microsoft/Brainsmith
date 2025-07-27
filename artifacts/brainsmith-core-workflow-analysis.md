# BrainSmith Core Workflow Analysis

## Overview

This document traces the complete workflow of BrainSmith Core from blueprint YAML to hardware generation, identifying gaps, unused features, and opportunities for simplification.

## Complete Workflow: Blueprint to Hardware

### Phase 1: Entry Point & Parsing

```
blueprint.yaml → forge() → BlueprintParser.parse() → DesignSpace + ExecutionTree
```

#### 1.1 forge.py (Entry Point)
```python
def forge(model_path: str, blueprint_path: str) -> Tuple[DesignSpace, ExecutionNode]:
    parser = BlueprintParser()
    return parser.parse(blueprint_path, model_path)
```
- **Purpose**: Simple API entry point
- **Status**: ✅ Clean and minimal

#### 1.2 BlueprintParser (YAML → Objects)
- Loads YAML with inheritance support (`extends` field)
- Validates step names against registry
- Resolves kernel backends
- Creates DesignSpace and ExecutionTree

**Key Features Used:**
- ✅ YAML loading with deep merge inheritance
- ✅ Step validation via `has_step()`
- ✅ Kernel backend resolution via `list_backends_by_kernel()`
- ✅ FINN config mapping (platform→board, target_clk→synth_clk_period_ns)

**Issues:**
- ⚠️ Special handling for "infer_kernels" step (hardcoded magic)
- ⚠️ No validation that steps are compatible or in correct order

### Phase 2: Design Space Representation

#### 2.1 DesignSpace (Intermediate Representation)
```python
@dataclass
class DesignSpace:
    model_path: str
    steps: List[Union[str, List[Optional[str]]]]  # Step names or variations
    kernel_backends: List[Tuple[str, List[Type]]]  # (kernel_name, backend_classes)
    global_config: GlobalConfig
    finn_config: Dict[str, Any]
```

**What's Used:**
- ✅ Steps list for execution
- ✅ Kernel backends for infer_kernels
- ✅ Config for FINN builds

**What's Not:**
- ❌ Combination validation (checks size but doesn't use it)
- ❌ Kernel summary (just for display)

### Phase 3: Execution Tree Building

#### 3.1 ExecutionTree (Segment-Based Structure)
```
Root
├── Segment1 (steps: ["cleanup", "streamline"])
│   ├── Branch1 (steps: ["quantize_int8"])
│   └── Branch2 (steps: ["quantize_int4"])
└── Segment2 (steps: ["optimize"])
```

**Clever Design:**
- ✅ Segments group sequential steps
- ✅ Branches only at variation points
- ✅ Prefix sharing reduces redundant computation

**Issues:**
- ⚠️ Complex segment ID generation could be simpler
- ⚠️ Tree stats calculated but not used for optimization

### Phase 4: Execution

#### 4.1 Explorer (Orchestration)
```python
def explore_execution_tree(
    tree: ExecutionNode,
    model_path: Path,
    output_dir: Path,
    output_product: str = "compile_and_package"
) -> TreeExecutionResult:
```

**Flow:**
1. Creates FINNAdapter and Executor
2. Executor traverses tree depth-first
3. Each segment becomes a FINN build
4. Results cached based on output file existence

#### 4.2 Executor (Segment Processing)
```python
def execute(self, tree: ExecutionNode, base_output_dir: Path) -> TreeExecutionResult:
    # Stack-based traversal
    # Artifact sharing at branch points
    # Failure isolation per branch
```

**Good Design:**
- ✅ Clean separation of tree traversal and execution
- ✅ Artifact sharing reduces redundant work
- ✅ Simple file-based caching

**Issues:**
- ⚠️ Hardcoded output stage mapping
- ⚠️ No progress tracking beyond print statements
- ⚠️ Limited error recovery

#### 4.3 FINNAdapter (Framework Integration)
```python
def build(self, input_model: Path, config_dict: Dict, output_dir: Path) -> Optional[Path]:
    # Changes working directory (FINN requirement)
    # Discovers output model in intermediate_models/
    # Handles all FINN quirks
```

**Necessary Evils (Well-Isolated):**
- ✅ Working directory changes
- ✅ Output model discovery
- ✅ Model copying to avoid corruption

### Phase 5: Plugin System

#### 5.1 Registry (Central Plugin Store)
**What's Registered:**
- 243 external components (QONNX/FINN)
- Custom transforms, kernels, backends, steps
- Metadata for querying

**Actually Used:**
- ✅ Step lookup for validation
- ✅ Transform retrieval in steps
- ✅ Backend resolution for kernels
- ❌ Most registered components never used
- ❌ Metadata queries unused
- ❌ Framework namespacing unused

## What's Missing?

### 1. Default Pipeline
There's no default sequence of steps. Every blueprint must specify everything:
```yaml
# Current requirement
steps:
  - "cleanup"
  - "streamline"
  - "quantize"
  - "optimize"

# Missing: default pipelines
preset: "standard_quantization"  # Would expand to common steps
```

### 2. Step Dependencies & Ordering
No way to express or validate step dependencies:
```python
@step(
    name="quantize",
    requires=["cleanup", "streamline"],  # Not supported
    provides=["quantized_model"]         # Not supported
)
```

### 3. Kernel Inference Implementation
The "infer_kernels" step is handled specially but has no actual implementation:
```python
# In blueprint_parser.py
if step_spec == "infer_kernels":
    pending_steps.append({
        "kernel_backends": space.kernel_backends,
        "name": "infer_kernels"
    })
# But no actual step implementation exists!
```

### 4. Progress Tracking
No structured progress reporting:
- No progress bars
- No estimated completion time
- No structured logging (just prints)

### 5. Result Analysis
Execution produces models but no analysis:
- No performance metrics
- No resource utilization estimates
- No Pareto frontier visualization

## What's Not Being Used?

### 1. Plugin System Overhead
**Registered but Unused:**
- ~200 transforms that are never called
- Metadata system for plugin discovery
- Framework namespacing (qonnx:, finn:)
- Plugin versioning and authorship

### 2. Design Space Features
**Defined but Unused:**
- Combination estimation
- Design space validation beyond size
- Multiple backends per kernel

### 3. Tree Analysis
**Calculated but Unused:**
- Segment efficiency metrics
- Tree depth statistics
- Step distribution analysis

### 4. Advanced Plugin Features
**Available but Unused:**
- `find()` for metadata-based discovery
- `get_transforms_by_metadata()`
- Kernel inference transforms
- Default backend selection

## Recommendations

### 1. Simplify Plugin System
Remove unused features:
- Drop metadata if not used
- Remove framework namespacing
- Only register actually-used components

### 2. Add Default Pipelines
```python
PRESETS = {
    "standard": ["cleanup", "streamline", "quantize", "optimize"],
    "minimal": ["cleanup", "quantize"],
    "aggressive": ["cleanup", "streamline", "quantize", "optimize", "verify"]
}
```

### 3. Implement Missing Features
Priority order:
1. Kernel inference step implementation
2. Progress tracking with rich CLI
3. Step dependency validation
4. Result analysis tools

### 4. Remove Complexity
- Simplify segment ID generation
- Remove unused tree statistics
- Consolidate output stage mappings

### 5. Better Documentation
- Document required vs optional steps
- Explain step ordering requirements
- Provide blueprint templates

## Conclusion

The BrainSmith Core workflow is **architecturally sound** but **overengineered** for current usage. The clean separation of concerns and segment-based execution are excellent designs. However, the system registers hundreds of unused plugins and calculates unused metrics.

The workflow successfully transforms blueprints into hardware through:
1. Clean parsing with inheritance
2. Efficient segment-based execution  
3. Well-isolated FINN integration

But it could be significantly simplified by:
1. Removing unused plugin features
2. Adding sensible defaults
3. Implementing missing core features (kernel inference, progress tracking)
4. Focusing on the actual usage patterns rather than theoretical extensibility

The system achieves its goal but with unnecessary complexity. A focused simplification would improve usability while maintaining the elegant core architecture.