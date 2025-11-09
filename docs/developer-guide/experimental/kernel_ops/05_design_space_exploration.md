# Design Space Exploration

This chapter teaches you how to systematically explore hardware configurations to find optimal implementations.

## DSE Fundamentals

Design Space Exploration (DSE) is the process of navigating the space of valid configurations to optimize for your objectives (latency, area, power, etc.).

### The DSE Workflow

```
1. Build Design Space (once)
   ↓
2. Define Optimization Objectives
   ↓
3. Explore Configurations (systematic or heuristic)
   ↓
4. Evaluate Each Configuration
   ↓
5. Select Winner
   ↓
6. Apply to Hardware
```

### Quick Start Example

```python
# 1. Build design space
op.build_design_space(model_wrapper)

# 2. Define objective: minimize latency
def objective(point):
    return point.initiation_interval  # Lower is better

# 3. Explore all configurations
best_point = None
best_cycles = float('inf')

for config in iter_valid_configurations(op, model_wrapper):
    point = op.design_space.configure(config)
    cycles = objective(point)

    if cycles < best_cycles:
        best_cycles = cycles
        best_point = point

# 4. Apply winner
op.apply_design_point(best_point)
print(f"Best config: {best_point.config}, Cycles: {best_cycles}")
```

## Exploration Strategies

### Strategy 1: Exhaustive Search

Evaluate **all** valid configurations.

```python
from brainsmith.dataflow.utils import iter_valid_configurations

# Iterate through all combinations
results = []
for config in iter_valid_configurations(op, model_wrapper):
    point = op.design_space.configure(config)

    results.append({
        "config": config,
        "cycles": point.initiation_interval,
        "area": estimate_area(point),  # Your area model
    })

# Sort by objective
results.sort(key=lambda x: x["cycles"])
print(f"Best: {results[0]}")
```

**Pros:**
- Guaranteed to find global optimum
- Simple to implement
- Useful for analysis/visualization

**Cons:**
- Expensive for large spaces (thousands of configs)
- Doesn't scale to complex multi-dimensional spaces

**Use when:** Design space is small (<1000 configs) or need guarantee of optimality.

### Strategy 2: Greedy Search

Start at one corner, iteratively improve.

```python
# Start at minimum resources
current = op.design_space.configure({
    "SIMD": op.design_space.dim_min("SIMD"),
    "PE": op.design_space.dim_min("PE")
})

improved = True
while improved:
    improved = False

    # Try stepping up each dimension
    for dim_name in current.config.keys():
        candidate = current.with_step_up(dim_name, 1)

        if objective(candidate) < objective(current):
            current = candidate
            improved = True
            print(f"Improved: {dim_name} → {current.config[dim_name]}")

print(f"Final: {current.config}, Cycles: {objective(current)}")
```

**Pros:**
- Fast (explores only path to local optimum)
- Scales to large spaces
- Good for latency-only optimization

**Cons:**
- May miss global optimum (gets stuck in local minima)
- Doesn't handle tradeoffs well (area vs latency)

**Use when:** Single objective, large space, local optimum acceptable.

### Strategy 3: Percentage Sweep

Sample at regular intervals.

```python
# Define sampling points (quartiles)
percentages = [0.0, 0.25, 0.5, 0.75, 1.0]

# Sweep SIMD dimension
base = op.design_space.configure({"PE": 1})
for point in base.sweep_percentage("SIMD", percentages):
    print(f"SIMD={point.config['SIMD']:3d}: {point.initiation_interval} cycles")
```

**Pros:**
- Predictable sampling (always hits min/mid/max)
- Good for visualization
- Handles non-uniform dimension spacing

**Cons:**
- May miss interesting intermediate points
- Fixed sampling (not adaptive)

**Use when:** Analyzing dimension sensitivity, creating plots.

### Strategy 4: Pareto Frontier

Find all non-dominated solutions (no other point is better on all objectives):

```python
# Find Pareto frontier
pareto = []
for config in iter_valid_configurations(op, model_wrapper):
    point = op.design_space.configure(config)
    dominated = any(is_dominated(point, p) for p in pareto)
    if not dominated:
        pareto = [p for p in pareto if not is_dominated(p, point)]
        pareto.append(point)
```

**Use when:** Multiple conflicting objectives (latency vs area vs power). Result is set of optimal tradeoff points.

## Navigation Methods

### Basic Navigation

```python
base = design_space.configure({"SIMD": 32, "PE": 4})

# Direct assignment
point1 = base.with_dimension("SIMD", 64)

# Relative movement
point2 = base.with_step_up("SIMD", 2)   # 2 steps higher
point3 = base.with_step_down("PE", 1)   # 1 step lower

# Boundary access
fastest = base.with_max("SIMD")
smallest = base.with_min("SIMD")

# Percentage-based
balanced = base.with_percentage("SIMD", 0.5)
```

### Multi-Dimensional Navigation

```python
# Navigate multiple dimensions
point = base
point = point.with_dimension("SIMD", 128)
point = point.with_dimension("PE", 8)
point = point.with_dimension("ram_style", "block")

# Or in one statement (method chaining)
point = (base
    .with_dimension("SIMD", 128)
    .with_dimension("PE", 8)
    .with_dimension("ram_style", "block")
)
```

### Interface-Based Navigation

When you don't know the parameter names:

```python
# Set first input's parallelism to 32
point = base.with_input_stream(0, 32)

# Set output parallelism to match
output_param = point.get_output_stream_param(0)  # → "PE"
point = point.with_output_stream(0, 32)

# Sweep input parallelism
for p in base.sweep_input_stream(0):
    print(f"Input PE: {p.get_input_stream_value(0)}")
```

**Use when:** Writing generic DSE code that works across different kernels.

## Evaluation and Estimation

### Built-In Metrics

```python
point = design_space.configure(config)

# Latency metrics
cycles = point.initiation_interval
max_block_cycles = point.max_block_folding_factor
max_tensor_blocks = point.max_tensor_folding_factor

# Bandwidth metrics
input0_bw = point.input_list[0].stream_width_bits
output_bw = point.output_stream_width_bits(0)

# Computational throughput
total_ops = point.total_output_values
throughput = total_ops / cycles
```

### Custom Area Models

Estimate resource usage:

```python
def estimate_area(point):
    """Simple area model (DSPs + BRAMs)."""
    # DSPs for processing elements
    pe_count = point.config.get("PE", 1)
    simd_count = point.config.get("SIMD", 1)
    dsps = pe_count * simd_count

    # BRAMs for buffering
    total_buffer = 0
    for inp in point.input_list:
        if inp.is_weight:  # Static inputs buffered
            total_buffer += prod(inp.block_shape) * inp.datatype.bitwidth()

    brams = total_buffer // (36 * 1024)  # 36Kb per BRAM

    return {"DSPs": dsps, "BRAMs": brams, "total": dsps + brams * 10}

# Use in DSE
for point in design_space.sweep_dimension("SIMD"):
    area = estimate_area(point)
    print(f"SIMD={point.config['SIMD']}: {area['DSPs']} DSPs, {area['BRAMs']} BRAMs")
```

### Power Estimation

```python
def estimate_power(point, clock_freq_mhz=100):
    """Estimate dynamic power (mW)."""
    # Switching activity proportional to throughput
    activity = point.total_output_values / point.initiation_interval

    # Area contributes to leakage
    area = estimate_area(point)

    # Simple model
    dynamic = activity * clock_freq_mhz * 0.01  # mW
    static = area["total"] * 0.5  # mW

    return dynamic + static
```

## Filtering and Constraints

### Filter by Dimension Range

```python
from brainsmith.dataflow.utils import iter_valid_configurations

# Only explore SIMD >= 16
filters = {"SIMD": lambda x: x >= 16}

for config in iter_valid_configurations(op, model_wrapper, param_filters=filters):
    point = design_space.configure(config)
    # Only sees SIMD ∈ {16, 32, 64, ...}
```

### Multi-Constraint Filtering

```python
# Complex filter: SIMD >= 16 AND PE <= 8
filters = {
    "SIMD": lambda x: x >= 16,
    "PE": lambda x: x <= 8
}

configs = list(iter_valid_configurations(op, model_wrapper, param_filters=filters))
print(f"Filtered design space: {len(configs)} configurations")
```

### Runtime Constraints

```python
def meets_requirements(point):
    """Check if point satisfies runtime constraints."""
    # Latency budget
    if point.initiation_interval > MAX_CYCLES:
        return False

    # Area budget
    area = estimate_area(point)
    if area["DSPs"] > MAX_DSPS or area["BRAMs"] > MAX_BRAMS:
        return False

    # Bandwidth budget
    if point.input_list[0].stream_width_bits > MAX_BANDWIDTH:
        return False

    return True

# Apply during exploration
feasible = [p for p in all_points if meets_requirements(p)]
```

## Visualization

### Latency vs Parallelism

```python
import matplotlib.pyplot as plt

# Sweep SIMD
base = design_space.configure({"PE": 1})
simd_values = []
cycles_values = []

for point in base.sweep_dimension("SIMD"):
    simd_values.append(point.config["SIMD"])
    cycles_values.append(point.initiation_interval)

plt.figure(figsize=(10, 6))
plt.plot(simd_values, cycles_values, marker='o')
plt.xlabel("SIMD (Parallelism)")
plt.ylabel("Cycles (Latency)")
plt.title("Latency vs Parallelism")
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()
```

### Pareto Frontier Plot

```python
# Extract pareto points (from earlier example)
cycles = [p.initiation_interval for p in pareto]
areas = [estimate_area(p)["total"] for p in pareto]

plt.figure(figsize=(10, 6))
plt.scatter(cycles, areas, marker='o', s=100, alpha=0.6)
plt.xlabel("Cycles (Latency)")
plt.ylabel("Area (DSPs + BRAMs)")
plt.title("Pareto Frontier: Latency vs Area")
plt.grid(True)

# Annotate points
for p, c, a in zip(pareto, cycles, areas):
    plt.annotate(f"SIMD={p.config['SIMD']}", (c, a))

plt.show()
```

### Heatmap (2D Parameter Space)

```python
import numpy as np
import seaborn as sns

# Create grid
simd_range = design_space.get_dimension("SIMD").values
pe_range = design_space.get_dimension("PE").values

# Evaluate all combinations
results = np.zeros((len(pe_range), len(simd_range)))
for i, pe in enumerate(pe_range):
    for j, simd in enumerate(simd_range):
        point = design_space.configure({"SIMD": simd, "PE": pe})
        results[i, j] = point.initiation_interval

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    results,
    xticklabels=simd_range,
    yticklabels=pe_range,
    annot=True,
    fmt='.0f',
    cmap='viridis_r'
)
plt.xlabel("SIMD")
plt.ylabel("PE")
plt.title("Latency (cycles) - Lower is Better")
plt.show()
```

## Advanced Patterns

### Hierarchical Exploration

```python
# Phase 1: Coarse sweep (quartiles only)
coarse_points = list(
    base.sweep_percentage("SIMD", [0.0, 0.25, 0.5, 0.75, 1.0])
)
best_coarse = min(coarse_points, key=lambda p: objective(p))

# Phase 2: Fine sweep around winner (±2 steps)
fine_points = []
for step in range(-2, 3):
    try:
        point = best_coarse.with_step_up("SIMD", step)
        fine_points.append(point)
    except ValueError:
        pass  # Out of range

best_fine = min(fine_points, key=lambda p: objective(p))
```

### Multi-Objective Ranking

```python
def rank_configs(points):
    """Rank by weighted objectives."""
    weights = {
        "cycles": 0.5,   # 50% weight on latency
        "area": 0.3,     # 30% weight on area
        "power": 0.2     # 20% weight on power
    }

    scores = []
    for point in points:
        # Normalize metrics (0-1 scale)
        cycles_norm = point.initiation_interval / max_cycles
        area_norm = estimate_area(point)["total"] / max_area
        power_norm = estimate_power(point) / max_power

        # Weighted sum (lower is better)
        score = (
            weights["cycles"] * cycles_norm +
            weights["area"] * area_norm +
            weights["power"] * power_norm
        )
        scores.append((point, score))

    # Sort by score
    scores.sort(key=lambda x: x[1])
    return [p for p, s in scores]

# Use it
ranked = rank_configs(all_points)
winner = ranked[0]
```

### Sensitivity Analysis

```python
def sensitivity_analysis(base_point, dim_name):
    """Analyze how changing one dimension affects objectives."""
    results = []

    for point in base_point.sweep_dimension(dim_name):
        value = point.config[dim_name]
        cycles = point.initiation_interval
        area = estimate_area(point)["total"]

        results.append({
            "value": value,
            "cycles": cycles,
            "area": area,
            "cycles_change": (cycles - results[0]["cycles"]) / results[0]["cycles"] if results else 0,
            "area_change": (area - results[0]["area"]) / results[0]["area"] if results else 0,
        })

    return results

# Analyze SIMD sensitivity
sensitivity = sensitivity_analysis(base, "SIMD")
for r in sensitivity:
    print(f"SIMD={r['value']:3d}: Cycles={r['cycles']:6d} ({r['cycles_change']:+.1%}), "
          f"Area={r['area']:4d} ({r['area_change']:+.1%})")
```

## Complete DSE Workflow

```python
def run_dse(op, model_wrapper, objectives):
    """Complete DSE: build space, evaluate configs, find Pareto frontier, select winner."""
    op.build_design_space(model_wrapper)

    results = []
    for config in iter_valid_configurations(op, model_wrapper):
        point = op.design_space.configure(config)
        scores = {name: fn(point) for name, fn in objectives.items()}
        results.append({"config": config, "point": point, "scores": scores})

    pareto = compute_pareto_frontier(results, objectives.keys())
    winner = rank_by_weighted_objectives(pareto, objectives.keys())[0]
    return {"all_results": results, "pareto": pareto, "winner": winner}
```

See Chapter 7 for complete helper functions (`compute_pareto_frontier`, `rank_by_weighted_objectives`).

## Best Practices

### DO: Cache Design Space

```python
# Good - build once
op.build_design_space(model_wrapper)
for config in configs:
    point = op.design_space.configure(config)  # Fast!

# Bad - rebuild every time
for config in configs:
    op.set_nodeattr("SIMD", config["SIMD"])  # Invalidates!
    point = op.design_point  # Slow rebuild
```

### DO: Use Appropriate Strategy

```python
# Small space (<100 configs): Exhaustive
if num_configs < 100:
    all_points = [design_space.configure(c) for c in all_configs]

# Medium space (100-1000): Percentage sweep
elif num_configs < 1000:
    points = list(base.sweep_percentage("SIMD", np.linspace(0, 1, 20)))

# Large space (>1000): Greedy or sampling
else:
    winner = greedy_search(design_space, objective)
```

### DO: Validate Winners

```python
# Always validate before applying
winner = find_best_config(design_space)

# Sanity checks
assert winner.initiation_interval > 0
assert all(v > 0 for v in winner.config.values())

# Requirements
assert meets_latency_budget(winner)
assert meets_area_budget(winner)

# Then apply
op.apply_design_point(winner)
```

### DON'T: Ignore Constraints

```python
# Bad - may generate invalid configurations
point = design_space.configure({"SIMD": 5})  # Might not divide!

# Good - check valid first
valid_simd = design_space.get_dimension("SIMD").values
assert 5 in valid_simd, "Invalid SIMD value"
```

## Next Steps

You now understand DSE:

✓ Exploration strategies (exhaustive, greedy, Pareto)
✓ Navigation methods (direct, relative, interface-based)
✓ Evaluation and estimation (latency, area, power)
✓ Filtering and constraints
✓ Visualization techniques

**Next chapter:** Advanced topics - broadcasting, static optimization, custom derivation.
