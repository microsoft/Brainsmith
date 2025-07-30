# Design Space Exploration

The execution tree organizes how Brainsmith explores different points in the Design Space for a neural network accelerator. Each path through the tree represents different design choices (optimizations, hardware implementations, parallelization).

## Tiered Design Space Exploration

- **Global Design Space** - All potential dataflow *architectures* to implement a neural network on FPGA
- **Build Search Space** - All potenital *implementations* of a single dataflow architecture.

| Local Search Space (FINN)                  | Global Design Space (Brainsmith)             |
| ------------------------------------------ | -------------------------------------------- |
| Network optimizations                      | Platform (board, `fpga_part`)                |
| FIFO sizing                                | Kernel implementations                       |
| Kernel parallelism (PE, SIMD, Tiling)      | DSE model transforms (streamlining)          |
| Kernel variations (RTL vs HLS, LUT vs DSP) | DSE HW transforms (auto‑folding)             |
|                                            | HW targets (target `clk`, `mvau_wwidth_max`) |

## What is the Execution Tree?

When exploring a model's design space, most search spaces will have a significant amount of overlap, only varying in single kernel or optimization transform. To exploit this, build pipelines with shared beginnings are merged, splitting at a *branch point*. This enables artifact reuse between branching builds, cutting down redundant compute.

```
                                 ┌→ streamline → fold_constants → finalize
start → tidy_up → convert_to_hw →┤
                                 └→ streamline_aggressive → minimize_bit_width → finalize
``` 

Steps are collected into *segments* of contiguous, non-branching steps that are run as a single FINN build.

### Segments
Groups of FINN transformations that execute together in a single build. Reduces overhead by batching related operations.

```python
@dataclass
class ExecutionSegment:
    segment_id: str                    # "root_0_1"
    steps: List[str]                   # ["Streamline", "ConvertBipolarToXnor"]
    parent: Optional['ExecutionSegment']
    children: List['ExecutionSegment']
    is_branch_point: bool
    input_artifact: Optional[Path]
    output_artifact: Optional[Path]
```
```python
def build(self, model_path, output_dir, config_dict):
    finn_config = DataflowBuildConfig(**config_dict)
    os.chdir(output_dir)  # FINN requirement
    
    model = ModelWrapper(model_path)
    for step_name in segment.steps:
        transform = get_step(step_name)
        model = transform(model, finn_config)
    
    return self._discover_output_model(output_dir)
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
`executor.py` runs depth-first:
```python
stack = [(root, initial_model, 0)]
while stack:
    segment, input_model, depth = stack.pop()
    result = self._execute_segment(segment, input_model, output_dir)
    for child in segment.children:
        stack.append((child, result.output_model, depth + 1))
```

### 3. Segment Execution
Each segment = one FINN build:
1. Create output directory: `output_dir/segment_id/`
2. Check cache (skip if valid output exists)
3. Prepare FINN config
4. Run transformations
5. Discover output in `intermediate_models/`

### 4. Branch Points
Full artifact copy for each child (FINN modifies in-place):
```python
if parent.is_branch_point:
    for child in parent.children:
        shutil.copytree(parent.output_artifact, child_output_dir)
```
