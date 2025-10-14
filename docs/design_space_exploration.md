# Execution Tree and Design Space Exploration

The execution tree organizes how Brainsmith explores the design space for neural network accelerators. Each path through the tree represents different design choices (kernel implementations, optimization strategies, parallelization parameters).

## Tiered Design Space Exploration

- **Global Design Space** - All potential dataflow *architectures* to implement a neural network on FPGA
- **Build Search Space** - All potential *implementations* of a single dataflow architecture.

| Local Search Space (FINN)                  | Global Design Space (Brainsmith)             |
| ------------------------------------------ | -------------------------------------------- |
| Network optimizations                      | Platform (board, `fpga_part`)                |
| FIFO sizing                                | Kernel implementations                       |
| Kernel parallelism                         | DSE model transforms (streamlining)          |
| Kernel variations (RTL vs HLS, LUT vs DSP) | DSE HW transforms (auto‑folding)             |
|                                            | HW targets (e.g., `target clk`) |

## Execution Tree Architecture

Design space exploration often involves paths with significant overlap, differing only in specific optimizations or kernel choices. The execution tree exploits this by merging shared pipeline segments and splitting at *branch points*, enabling artifact reuse and reducing redundant computation.

```
                                          ┌→ step_minimize_bit_width → step_hw_codegen
cleanup → qonnx_to_finn → infer_kernels →┤
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

## Execution Flow

### 1. Tree Building
`DSETreeBuilder` in `brainsmith/core/design/builder.py` converts blueprint → execution tree:
- Groups sequential steps into segments
- Creates branches for alternatives
- Expands kernel inference into transforms

### 2. Traversal
`runner.py` executes segments using a stack-based approach:
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

### 3. Segment Execution
Each segment = one FINN build:
1. Create output directory: `output_dir/segment_id/`
2. Check cache (skip if valid output exists)
3. Prepare FINN config
4. Run transformations
5. Discover output in `intermediate_models/`

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
