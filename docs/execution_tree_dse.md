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
                                 ┌→ streamline → fold_constants → finalize
start → tidy_up → convert_to_hw →┤
                                 └→ streamline_aggressive → minimize_bit_width → finalize
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

```yaml
steps:
  - "qonnx_to_finn"      # These become
  - "tidy_up"            # one segment
  - ["streamline", "streamline_aggressive"]  # Branch point
  
kernels:
  - LayerNorm: [LayerNorm_hls, LayerNorm_rtl]  # Creates paths
```

## Execution Flow

### 1. Tree Building
`tree_builder.py` converts blueprint → execution tree:
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
def share_artifacts_at_branch(self, parent: DSESegment, children: List[DSESegment]):
    """Share parent output with all children."""
    parent_out = parent.output_dir / "output.onnx"
    for child in children:
        child_out = child.output_dir / "output.onnx"
        shutil.copy2(parent_out, child_out)
```
