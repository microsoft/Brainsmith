# Phase 1: Design Space Constructor - v4.0 Architecture

## Why This Refactor?

### The Problem
Brainsmith was over-engineered for a hypothetical future:
- **SearchConfig abstraction** for different search strategies (only exhaustive was ever used)
- **Validator ceremony** that just checked if files exist
- **Complex backend selection** for multiple backend types (only FINN exists)
- **Implicit kernel defaults** that hid what was actually happening

### The Solution
**Commit fully to FINN** and embrace simplicity:
- Delete abstractions that serve no purpose
- Make everything explicit and direct
- Use existing infrastructure instead of reinventing

## What Changed

### 🗑️ Deleted (200+ lines removed)
- `validator.py` - Pure ceremony, the OS tells us if files don't exist
- `SearchConfig`, `SearchStrategy`, `SearchConstraint` - Unnecessary wrappers
- `ValidationResult` - Validation happens in `__post_init__` now

### 📝 Simplified
- **Direct limits** on DesignSpace: `max_combinations`, `timeout_minutes`
- **Explicit kernels**: No magic, declare what you use
- **Clear error messages**: "Design space too large: 50,000 > 10,000"

### ✨ Added
- **Blueprint inheritance**: Reuse common configurations
- **Dynamic stage registration**: Transform stages become build steps
- **Kernel inference step**: Clean separation of concerns

## New Structure

### `data_structures.py`
Core data models with direct fields and validation:

```python
@dataclass
class DesignSpace:
    model_path: str
    hw_compiler_space: HWCompilerSpace
    global_config: GlobalConfig
    max_combinations: int = 100000  # Direct field
    timeout_minutes: int = 60       # Direct field
    
    def __post_init__(self):
        # Fail fast if space too large
        if self._estimate_size() > self.max_combinations:
            raise ValueError(...)
```

**Key insight**: Validation belongs with the data, not in a separate file.

### `parser.py`
Simple blueprint parser with powerful features:

```python
class BlueprintParser:
    def parse(blueprint_data, model_path) -> DesignSpace
    def parse_with_inheritance(blueprint_path, model_path) -> DesignSpace
```

**Features**:
- **Inheritance**: `extends: base.yaml` for DRY blueprints
- **Dynamic stages**: Transform stages become executable steps
- **No interpretation**: Parser structures data, doesn't judge it

### `forge.py`
Minimal API that orchestrates design space construction:

```python
def forge(model_path: str, blueprint_path: str) -> DesignSpace:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    parser = BlueprintParser()
    design_space = parser.parse(load_blueprint(blueprint_path), model_path)
    
    logger.info(f"✅ Created design space with ~{size:,} configurations")
    return design_space
```

**No validator needed** - DesignSpace validates itself.

**Future Evolution**: This module will grow to become the end-to-end orchestrator,
coordinating all three phases to provide a unified `forge()` API that can optionally
run exploration and builds in addition to design space construction.

### `exceptions.py`
Unchanged - good error types for clear communication.

## Blueprint Format v4.0

### Explicit Kernels
```yaml
kernels:
  - ["MatrixVectorActivation", ["hls", "rtl"]]  # One kernel, multiple backends
  - ["~Thresholding", ["hls"]]                  # Optional kernel
```

### Transform Stages
```yaml
transforms:
  cleanup:
    - "RemoveIdentityOps"              # Required
    - "~OptionalTransform"             # Optional
    - ["Option1", "Option2", "~"]      # Mutually exclusive

build_pipeline:
  steps:
    - "{cleanup}"  # Expands to brainsmith_stage_cleanup
```

### Direct Limits
```yaml
max_combinations: 10000  # No SearchConfig wrapper
timeout_minutes: 60      # Direct and clear
```

## Design Principles

1. **Deletion First**: Less code is better code
2. **Direct Over Abstract**: `max_combinations` not `search.constraints[0].value`
3. **Explicit Over Magic**: Declare all kernels, no hidden defaults
4. **Fail Fast**: Validate in `__post_init__`, not later
5. **Use What Exists**: OS for file checks, plugins for transforms

## Usage

```python
from brainsmith.core.phase1 import forge

# Current API - Phase 1 only
design_space = forge("model.onnx", "blueprint.yaml")

# With inheritance
parser = BlueprintParser()
design_space = parser.parse_with_inheritance("child.yaml", "model.onnx")

# Future unified API (planned)
results = forge("model.onnx", "blueprint.yaml", explore=True, build=True)
best = results.pareto_front[0]  # Best configuration found
```

## Migration

Old v3.0 blueprints won't work. Update them:
- Remove `search:` section
- Add kernels explicitly
- Move limits to top level

No compatibility layer - users adapt or stay on old version.

---

*This is Arete: Code in its highest form where every line serves its purpose with crystalline clarity.*