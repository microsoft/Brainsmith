# Design Space Exploration

Evaluate multiple hardware configurations to find optimal designs.

Brainsmith uses segment-based DSE to efficiently explore large design spaces by reusing computation across similar configurations.

---

::: brainsmith.dse.explore_design_space

**Example:**

```python
from brainsmith import explore_design_space
from brainsmith.dse.types import SegmentStatus

# Run complete DSE pipeline
results = explore_design_space(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml",
    output_dir="./build"
)

# Analyze results
stats = results.compute_stats()
print(f"Successful: {stats['successful']}/{stats['total']}")
print(f"Total time: {results.total_time:.2f}s")

# Access successful outputs
for seg_id, result in results.segment_results.items():
    if result.status == SegmentStatus.COMPLETED:
        print(f"Output: {result.output_model}")
```

---

::: brainsmith.dse.parse_blueprint

**Example:**

```python
from brainsmith.dse import parse_blueprint

design_space, config = parse_blueprint(
    blueprint_path="blueprint.yaml",
    model_path="model.onnx"
)

print(f"Steps: {len(design_space.steps)}")
print(f"Kernels: {design_space.kernel_backends}")
```

---

::: brainsmith.dse.build_tree

**Example:**

```python
from brainsmith.dse import parse_blueprint, build_tree

design_space, config = parse_blueprint(
    blueprint_path="blueprint.yaml",
    model_path="model.onnx"
)

tree = build_tree(design_space, config)

# Inspect before execution
stats = tree.get_statistics()
print(f"Paths: {stats['total_paths']:,}")
print(f"Segments: {stats['total_segments']:,}")
```

---

::: brainsmith.dse.execute_tree

**Example:**

```python
from brainsmith.dse import build_tree, execute_tree

tree = build_tree(design_space, config)

result = execute_tree(
    tree=tree,
    model_path="model.onnx",
    config=config,
    output_dir="./build"
)
```

---

::: brainsmith.dse.SegmentRunner

---

::: brainsmith.dse.DSEConfig

---

::: brainsmith.dse.GlobalDesignSpace

---

::: brainsmith.dse.TreeExecutionResult

**Example:**

```python
results = explore_design_space(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml",
    output_dir="./build"
)

# Compute statistics
stats = results.compute_stats()
print(stats['successful'], stats['failed'], stats['total'])

# Access individual segments
for seg_id, seg_result in results.segment_results.items():
    print(f"{seg_id}: {seg_result.status}")
```

---

::: brainsmith.dse.SegmentResult

---

::: brainsmith.dse.SegmentStatus

---

::: brainsmith.dse.OutputType

---

::: brainsmith.dse.ExecutionError

---

::: brainsmith.dse.DSETree

---

::: brainsmith.dse.DSESegment

---

## See Also

- [Getting Started](../getting-started.md) - Installation and quickstart
- [GitHub Examples](https://github.com/microsoft/brainsmith/tree/main/examples) - Working code examples
