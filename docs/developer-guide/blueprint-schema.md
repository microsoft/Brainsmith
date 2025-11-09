# Blueprint Schema Reference

Blueprints are YAML files defining the design space for FPGA accelerator generation.

## Quick Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | No | Blueprint name |
| `description` | string | No | Blueprint description |
| `extends` | path | No | Parent blueprint for inheritance |
| `clock_ns` | float | **Yes** | Target clock period (nanoseconds) |
| `output` | string | No | Output type: `estimates` \| `rtl` \| `bitfile` (default: `estimates`) |
| `board` | string | Conditional | Target FPGA board (required for `rtl`/`bitfile`) |
| `start_step` | string | No | Pipeline start step (inclusive) |
| `stop_step` | string | No | Pipeline stop step (inclusive) |
| `finn_config` | dict | No | FINN parameter overrides |
| `design_space.kernels` | list | **Yes** | Hardware kernels to use |
| `design_space.steps` | list | **Yes** | Transformation pipeline |

---

## Minimal Blueprint

```yaml
name: "My Accelerator"
clock_ns: 5.0

design_space:
  kernels:
    - MVAU
    - Thresholding

  steps:
    - "qonnx_to_finn"
    - "infer_kernels"
    - "specialize_layers"
```

---

## Core Configuration

### clock_ns (required)

Target clock period in nanoseconds. Determines timing constraints for synthesis.

```yaml
clock_ns: 5.0    # 200MHz (1/5ns)
clock_ns: 4.0    # 250MHz (1/4ns)
```

### output

How far to proceed in the build pipeline:

```yaml
output: "estimates"    # Resource estimates only (default)
output: "rtl"          # Generate RTL + IP blocks
output: "bitfile"      # Full synthesis to bitstream
```

### board

Target FPGA board. Required when `output` is `rtl` or `bitfile`.

```yaml
board: "Pynq-Z1"
board: "ZCU104"
board: "V80"
```

### start_step / stop_step

Control execution range for debugging or incremental builds:

```yaml
start_step: "streamline"           # Start from this step (inclusive)
stop_step: "generate_estimates"    # Stop at this step (inclusive)
```

**CLI overrides** (take precedence):
```bash
smith model.onnx blueprint.yaml --start-step streamline --stop-step streamline
```

### finn_config

Direct FINN parameter overrides (deep-merged during inheritance):

```yaml
finn_config:
  target_fps: 3000
  standalone_thresholds: true
  rtlsim_batch_size: 100
```

---

## Kernels

Define hardware implementations available for layer mapping.

**All backends (auto-sorted: RTL → HLS):**
```yaml
kernels:
  - MVAU
  - LayerNorm
```

**Specific backends (explicit priority order):**
```yaml
kernels:
  - MVAU: [MVAU_hls, MVAU_rtl]              # HLS first
  - Softmax: Softmax_hls                     # Single backend
  - Crop: [brainsmith:Crop_rtl]              # Fully-qualified name
```

**Backend resolution:**
- String format → all registered backends, sorted by priority
- Dict format → only specified backends, in given order
- Supports short names (`MVAU_hls`) and qualified names (`brainsmith:MVAU_hls`)

---

## Steps

Transformation pipeline with support for variations and optional steps.

**Linear pipeline:**
```yaml
steps:
  - "qonnx_to_finn"
  - "streamline"
  - "infer_kernels"
```

**Branch points (design space exploration):**
```yaml
steps:
  - "tidy_up"
  - ["streamline", "streamline_aggressive"]    # Try both
  - "convert_to_hw"
  - ["minimize_bit_width", ~]                  # Optional step
```

The second example creates 4 execution paths (2 × 2 combinations).

**Skip indicators:** `~`, `null`, `""` (all equivalent)

**Constraints:**
- Maximum 1 skip per branch point
- Minimum 1 non-skip per branch point
- No nested lists in branch points (use double brackets `[[...]]` for operations)

---

## Inheritance

Reuse and extend existing blueprints via `extends`:

```yaml
# parent.yaml
name: "Base Pipeline"
clock_ns: 5.0

design_space:
  kernels:
    - MVAU
  steps:
    - "qonnx_to_finn"
    - "streamline"
```

```yaml
# child.yaml
extends: "parent.yaml"
name: "Extended Pipeline"
output: "rtl"
board: "Pynq-Z1"

design_space:
  kernels:
    - MVAU
    - Thresholding    # Replaces parent kernels entirely

  steps:
    - after: "streamline"
      insert: "custom_step"
```

**Inheritance rules:**
1. Simple fields (name, clock_ns, etc.) → Child overrides parent
2. `finn_config` → Deep merge (child fields override parent fields)
3. `kernels` → Child replaces parent entirely (or inherits if not specified)
4. `steps` → Child replaces parent entirely (or inherits if not specified)
5. Step operations (`after`, `before`, etc.) → Applied after determining base steps

---

## Step Operations

Modify inherited or complex step lists:

**Insert after/before:**
```yaml
steps:
  - after: "streamline"
    insert: "custom_optimization"

  - before: "specialize_layers"
    insert:
      - "validation_step"
      - ["option1", "option2"]    # Insert branch point
```

**Replace/remove:**
```yaml
steps:
  - replace: "old_step"
    with: "new_step"

  - replace: "branch_step"
    with: [["new_option1", "new_option2"]]    # Replace with branch

  - remove: "unwanted_step"
```

**Insert at start/end:**
```yaml
steps:
  - at_start:
      insert: "initialization"

  - at_end:
      insert: ["package_ip", "validate"]
```

---

## Environment Variables

Use `${VAR}` syntax for dynamic path resolution:

```yaml
extends: "${BSMITH_DIR}/examples/blueprints/base.yaml"
board: "${TARGET_BOARD}"
```

**Available variables:**
- `${BLUEPRINT_DIR}` - Directory containing current blueprint
- `${BSMITH_DIR}` - Brainsmith installation directory
- Any shell environment variable

**Notes:**
- Context variables override environment variables
- Undefined variables remain unexpanded (safe substitution)
- Thread-safe (no `os.environ` mutation)

---

## Design Space Size

Design space size = product of all branch point sizes.

**Limits:**
- Default: 100,000 combinations
- Environment override: `export BRAINSMITH_MAX_COMBINATIONS=500000`
- Validation: Exceeding limit raises `ValueError` before execution

**Example:**
```yaml
steps:
  - ["opt1", "opt2"]           # 2 options
  - ["opt3", "opt4", "opt5"]   # 3 options
  - ["opt6", ~]                # 2 options (with skip)
# Total: 2 × 3 × 2 = 12 combinations
```

---

## Execution Semantics

Brainsmith builds an execution tree where:

- Nodes = execution segments (sequential steps)
- Branches = variation points (lists)
- Leaves = complete execution paths

**Segment-based execution:** Steps between branch points form single segments. Each segment executes as one FINN build. Artifacts are shared at branch points to avoid redundant computation.

**Example tree:**
```yaml
steps:
  - "tidy_up"
  - ["streamline", "streamline_aggressive"]
  - "convert_to_hw"
```

Creates 2 paths:
1. `tidy_up → streamline → convert_to_hw`
2. `tidy_up → streamline_aggressive → convert_to_hw`

Both paths share the `tidy_up` segment.

---

## See Also

- **[Hardware Kernels](hardware-kernels.md)** - Kernel architecture and implementation
- **[Component Registry](registry.md)** - Registering kernels, backends, and steps
- **[CLI Reference](../api/cli.md)** - Command-line interface and options
