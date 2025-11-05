# Design Space Exploration

Design Space Exploration (DSE) is the process of systematically evaluating different hardware configurations to find implementations that meet your performance, resource, and power requirements. For FPGA accelerators, the design space is vast—hundreds or thousands of valid configurations exist for a single neural network.

## The Problem: Combinatorial Explosion

Consider a simple 3-layer neural network:

- **3 kernel types** × **2 backends each** (HLS vs RTL) = 8 combinations
- **2 optimization strategies** (minimize bit-width vs. apply folding) = 16 combinations
- **3 folding configurations per layer** = 48 combinations
- **2 FIFO sizing strategies** = 96 combinations

For just 3 layers, you already have **96 potential designs**—and this grows exponentially with network depth. Evaluating each design independently would take days or weeks.

## The Solution: Segment-Based DSE

Brainsmith's execution tree exploits the insight that **most design configurations share common pipeline stages**. Instead of redundantly building similar configurations from scratch, the execution tree:

1. **Identifies shared computation** (e.g., initial model cleanup and quantization)
2. **Splits at branch points** where configurations diverge (e.g., HLS vs RTL backend choice)
3. **Reuses artifacts** from parent segments, only rebuilding what changed

This reduces exploration time from O(n) to approximately O(log n) for branching configurations.

---

## Tiered Design Space Exploration

Brainsmith distinguishes between two levels of design space:

- **Global Design Space** - All potential dataflow *architectures* to implement a neural network on FPGA
- **Build Search Space** - All potential *implementations* of a single dataflow architecture

| Local Search Space (FINN)                  | Global Design Space (Brainsmith)             |
| ------------------------------------------ | -------------------------------------------- |
| Network optimizations                      | Platform (board, `fpga_part`)                |
| FIFO sizing                                | Kernel implementations                       |
| Kernel parallelism                         | DSE model transforms (streamlining)          |
| Kernel variations (RTL vs HLS, LUT vs DSP) | DSE HW transforms (auto‑folding)             |
|                                            | HW targets (e.g., `target_clk`)              |

**Brainsmith operates at the global level**, letting you explore different kernel choices, optimization strategies, and high-level architecture decisions. Each path through Brainsmith's execution tree produces a dataflow architecture that can then be further optimized by FINN's local search (folding parameters, FIFO depths, etc.).

---

## Execution Tree Architecture

The execution tree organizes how Brainsmith explores the design space for neural network accelerators. Each path through the tree represents different design choices (kernel implementations, optimization strategies, parallelization parameters).

Design space exploration often involves paths with significant overlap, differing only in specific optimizations or kernel choices. The execution tree exploits this by merging shared pipeline segments and splitting at *branch points*, enabling artifact reuse and reducing redundant computation.

```
                                                    ┌→ step_minimize_bit_width → step_hw_codegen
cleanup → qonnx_to_finn → build_dataflow_graph →┤
                                                    └→ step_apply_folding_config → step_hw_codegen
```

Steps are collected into *segments* of contiguous, non-branching steps that are run as a single FINN build.

### Segments

Segments group contiguous transformations that execute together in a single FINN build, reducing overhead and enabling efficient caching.

```python
@dataclass
class DSESegment:
    transforms: List[Dict[str, Any]]     # Steps to execute
    branch_choice: Optional[str]         # Which branch was taken
    parent: Optional['DSESegment']
    children: Dict[str, 'DSESegment']    # branch_id → child

    @property
    def segment_id(self) -> str:        # Path-based ID like "root" or "streamline/fold"
    @property
    def is_branch_point(self) -> bool:  # Has multiple children
```


### Tree Structure

The execution tree supports several branching patterns:

- **Sequential steps** → Single segment
- **Alternatives** → Branch points with child segments
- **Kernel inference** → Expands to kernel/backend combinations
- **Skip option** → Use `~` to create optional paths

```yaml
steps:
  - "qonnx_to_finn"      # These become
  - "cleanup"            # one segment
  - ["step_minimize_bit_width", "step_apply_folding_config"]  # Branch point
  - ["~", "step_set_fifo_depths"]  # Optional step (skip or execute)

kernels:
  - LayerNorm: [LayerNorm_hls, LayerNorm_rtl]  # Creates paths
```

The skip indicator `~` allows creating paths that bypass certain optimizations, useful for comparing performance with and without specific transformations.

---

## Execution Flow

### 1. Tree Building

`DSETreeBuilder` in `brainsmith/dse/_builder.py` converts blueprint → execution tree:
- Groups sequential steps into segments
- Creates branches for alternatives
- Expands kernel inference into transforms

**Example Blueprint:**
```yaml
design_space:
  steps:
    - streamline
    - qonnx_to_finn
    - [step_minimize_bit_width, step_apply_folding_config]
  kernels:
    - MVAU
    - LayerNorm: [LayerNorm_hls, LayerNorm_rtl]
```

**Resulting Tree:**
```
root (streamline, qonnx_to_finn)
├── minimize_bit_width (step_minimize_bit_width)
│   ├── LayerNorm_hls (infer LayerNorm=LayerNorm_hls)
│   └── LayerNorm_rtl (infer LayerNorm=LayerNorm_rtl)
└── apply_folding (step_apply_folding_config)
    ├── LayerNorm_hls (infer LayerNorm=LayerNorm_hls)
    └── LayerNorm_rtl (infer LayerNorm=LayerNorm_rtl)
```

This creates **4 paths** (2 optimization strategies × 2 kernel backends), but executes only **5 segments** instead of 12 (4 paths × 3 steps each), saving 58% of the computation.

### 2. Traversal

`runner.py` executes segments using a stack-based depth-first approach:

```python
stack = [(tree.root, initial_model, 0)]
while stack:
    segment, input_model, depth = stack.pop()

    # Execute segment (or use cached result)
    result = self._execute_segment(segment, input_model, output_dir)

    # Queue children for execution
    if result.success:
        for child in segment.children.values():
            stack.append((child, result.output_model, depth + 1))
```

**Depth-first traversal** ensures memory-efficient exploration—only one branch is active in memory at a time.

### 3. Segment Execution

Each segment corresponds to one FINN build:

1. **Create output directory**: `output_dir/segment_id/`
2. **Check cache**: Skip if valid output exists from previous run
3. **Prepare FINN config**: Set build directory, folding config, target platform
4. **Run transformations**: Execute all transforms in the segment sequentially
5. **Discover output**: Locate final model in `intermediate_models/`

```python
def _execute_segment(
    self,
    segment: DSESegment,
    input_model: Path,
    output_dir: Path
) -> SegmentResult:
    segment_dir = output_dir / segment.segment_id

    # Cache check
    if segment_dir.exists() and self._is_valid_cache(segment_dir):
        return SegmentResult.from_cache(segment_dir)

    # Execute FINN build
    finn_config = self._prepare_finn_config(segment, segment_dir)
    transformed_model = run_finn_transforms(
        input_model, segment.transforms, finn_config
    )

    return SegmentResult(
        segment=segment,
        output_model=transformed_model,
        output_dir=segment_dir,
        success=True
    )
```

### 4. Artifact Sharing

At branch points, the parent's output is shared with all children to avoid redundant computation:

```python
def share_artifacts_at_branch(
    parent_result: SegmentResult,
    child_segments: List[DSESegment],
    base_output_dir: Path
) -> None:
    """Copy build artifacts to child segments."""
    if not parent_result.success:
        return

    for child in child_segments:
        child_dir = base_output_dir / child.segment_id
        # Full directory copy for FINN compatibility
        if child_dir.exists():
            shutil.rmtree(child_dir)
        shutil.copytree(parent_result.output_dir, child_dir)
```

This ensures each child segment starts with the parent's output model and intermediate artifacts, enabling incremental builds.

---

## Blueprint Integration

Blueprints define the design space declaratively using YAML:

### Basic DSE

```yaml
design_space:
  steps:
    - streamline
    - qonnx_to_finn
    - step_create_dataflow_partition
  kernels:
    - MVAU
    - ConvolutionInputGenerator
```

**Result**: Single path (no branching)

### Branching Strategies

```yaml
design_space:
  steps:
    - streamline
    - qonnx_to_finn
    - [step_minimize_bit_width, step_apply_folding_config]  # 2 branches
  kernels:
    - MVAU
    - LayerNorm: [LayerNorm_hls, LayerNorm_rtl]  # 2 branches per strategy
```

**Result**: 4 paths (2 optimization strategies × 2 LayerNorm backends)

### Optional Steps

```yaml
design_space:
  steps:
    - streamline
    - qonnx_to_finn
    - ["~", "step_tidy_up"]  # 2 branches: skip or execute
```

**Result**: 2 paths (with/without tidy_up)

---

## Performance Impact

### Example: BERT Accelerator

**Naive Approach** (no artifact reuse):
- 16 configurations (4 kernel choices × 2 optimizations × 2 backends)
- 16 full builds × 45 minutes each = **12 hours**

**Segment-Based DSE**:
- 5 shared segments + 11 branch-specific segments = 16 total segments
- Most segments reused across branches
- Total time: **2.5 hours** (5× speedup)

### Caching Across Runs

Segment outputs are cached to disk. If you modify only one branch's configuration and re-run DSE:

- **Unchanged segments**: Load from cache (seconds)
- **Modified segment**: Rebuild (minutes)
- **Downstream segments**: Rebuild if affected

This enables iterative design space refinement without full rebuilds.

---

## Design Patterns

### Pattern 1: Compare Backends

Evaluate HLS vs RTL implementations:

```yaml
design_space:
  kernels:
    - MVAU: [MVAU_hls, MVAU_rtl]
    - Thresholding: [Thresholding_hls, Thresholding_rtl]
```

**Result**: 4 paths exploring all backend combinations

### Pattern 2: Optimization Strategies

Test different optimization approaches:

```yaml
design_space:
  steps:
    - streamline
    - qonnx_to_finn
    - [step_minimize_bit_width, step_apply_folding_config, step_minimize_accumulator_width]
```

**Result**: 3 paths, one per optimization strategy

### Pattern 3: Ablation Studies

Measure impact of specific transforms:

```yaml
design_space:
  steps:
    - streamline
    - qonnx_to_finn
    - ["~", "step_absorb_add_into_topk"]  # With/without optimization
    - step_create_dataflow_partition
```

**Result**: 2 paths enabling A/B comparison

---

## Limitations and Future Work

### Current Limitations

1. **No cross-segment parallelism**: Segments execute sequentially (depth-first)
2. **Full directory copies**: Artifact sharing copies entire directories (can be GBs)
3. **Manual branch specification**: No automatic search strategies yet

### Planned Enhancements

- **Parallel segment execution**: Execute independent branches concurrently
- **Incremental artifact sharing**: Copy only changed files
- **Auto-tuning**: Bayesian optimization over folding parameters
- **Multi-objective optimization**: Pareto frontier exploration (latency vs resources vs power)

---

## Summary

Brainsmith's segment-based DSE:
- **Reduces exploration time** from O(n) to O(log n) for branching configurations
- **Enables iterative refinement** through caching and incremental builds
- **Declarative specification** via Blueprint YAML
- **Scales to large design spaces** (hundreds of configurations in hours, not days)

The execution tree architecture makes design space exploration tractable for complex neural networks on FPGAs, turning what was once a manual, days-long process into an automated, hours-long workflow.

## Next Steps

- [Blueprints](../3-reference/blueprints.md) - Learn the YAML syntax for defining design spaces
- [Kernels](../3-reference/kernels.md) - Understand kernel choices and their impact on DSE
- [Component Registry](component-registry.md) - See how custom components integrate with DSE
