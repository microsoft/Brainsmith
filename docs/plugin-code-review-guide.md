# Plugin System Code Review Guide

**Version**: 1.0
**Last Updated**: 2025-10-22
**For Reviewers**: Quick reference for reviewing plugin-related code changes

---

## Quick Checklist

**For any plugin-related PR, verify:**

- [ ] COMPONENTS dict uses relative imports (starts with `.`)
- [ ] All decorators use correct syntax (@kernel, @step, @backend)
- [ ] No manual registry calls (use decorators, not `Registry.kernel()`)
- [ ] No eager imports in `__init__.py` (lazy loading only)
- [ ] Component names match their class/function names
- [ ] Tests verify component discovery (not just import)
- [ ] Documentation updated if adding new component type
- [ ] No breaking changes to existing component names

---

## 1. COMPONENTS Dictionary Review

### ✅ **GOOD** - Type-Safe, Relative Imports

```python
from brainsmith.plugin_helpers import ComponentsDict, create_lazy_module

COMPONENTS: ComponentsDict = {
    'kernels': {
        'MyKernel': '.my_kernel',           # ✓ Relative import
        'AnotherKernel': '.subdir.kernel',   # ✓ Nested module
    },
    'backends': {
        'MyKernel_hls': '.my_backend',       # ✓ Relative import
    },
    'steps': {
        'my_step': '.my_step',               # ✓ Relative import
    },
}

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

**Why Good**:
- ✓ Uses `ComponentsDict` for type safety
- ✓ All paths start with `.` (relative imports)
- ✓ Component names match what will be in the module
- ✓ Uses `create_lazy_module()` helper

### ❌ **BAD** - Common Mistakes

```python
# ❌ Absolute imports (breaks package portability)
COMPONENTS = {
    'kernels': {
        'MyKernel': 'plugins.my_kernel',  # BAD: absolute
    },
}

# ❌ Eager imports (defeats lazy loading)
from .my_kernel import MyKernel  # BAD: imported at module load
COMPONENTS = {
    'kernels': {'MyKernel': '.my_kernel'},
}

# ❌ Name mismatch
COMPONENTS = {
    'kernels': {
        'WrongName': '.my_kernel',  # BAD: if module has 'MyKernel'
    },
}

# ❌ Missing type annotation
COMPONENTS = {  # BAD: no ComponentsDict type
    'kernels': {'MyKernel': '.my_kernel'},
}

# ❌ Manual __getattr__ (don't reinvent the wheel)
def __getattr__(name):  # BAD: use create_lazy_module instead
    if name == 'MyKernel':
        from .my_kernel import MyKernel
        return MyKernel
```

**Review Action**: Request changes if any bad patterns found.

---

## 2. Component Implementation Review

### 2.1 Kernel Review

#### ✅ **GOOD** - Proper Kernel Implementation

```python
from brainsmith import kernel
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

@kernel
class LayerNorm(HWCustomOp):
    """Hardware-accelerated LayerNorm kernel.

    Implements layer normalization for FPGA deployment.
    """
    op_type = "LayerNorm"

    def get_nodeattr_types(self):
        return {
            "epsilon": ("f", True, 1e-5),
            "normalized_shape": ("ints", True, []),
        }

    def infer_node_datatype(self, model):
        """Infer output datatype."""
        node = self.onnx_node
        return model.get_tensor_datatype(node.input[0])

    def execute_node(self, context, graph):
        """Execute kernel in simulation."""
        # Implementation
        pass

    def verify_node(self):
        """Verify kernel configuration is valid."""
        # Validation logic
        pass
```

**Why Good**:
- ✓ Uses `@kernel` decorator (not manual registration)
- ✓ Inherits from `HWCustomOp`
- ✓ Defines `op_type` (ONNX node type)
- ✓ Has docstring explaining purpose
- ✓ Implements required methods (`execute_node`, `verify_node`)
- ✓ Includes inference methods if needed

#### ❌ **BAD** - Kernel Anti-Patterns

```python
# ❌ No decorator
class LayerNorm(HWCustomOp):  # BAD: not registered
    pass

# ❌ Manual registration (don't do this)
from brainsmith.registry import registry
class LayerNorm(HWCustomOp):
    pass
registry.kernel(LayerNorm, name='LayerNorm')  # BAD: use decorator

# ❌ Missing op_type
@kernel
class LayerNorm(HWCustomOp):
    # BAD: no op_type defined
    pass

# ❌ Wrong base class
@kernel
class LayerNorm:  # BAD: should inherit HWCustomOp
    pass

# ❌ No docstring
@kernel
class LayerNorm(HWCustomOp):  # BAD: no documentation
    op_type = "LayerNorm"
```

**Review Questions**:
- Does the kernel use `@kernel` decorator?
- Does it inherit from `HWCustomOp`?
- Is `op_type` defined and correct?
- Are all required methods implemented?
- Is there a docstring explaining what it does?

### 2.2 Step Review

#### ✅ **GOOD** - Proper Step Implementation

```python
from brainsmith import step
import logging

logger = logging.getLogger(__name__)

@step
def minimize_bit_width(model, cfg):
    """Minimize bit widths throughout the model.

    Analyzes data ranges and sets optimal bit widths for each tensor
    to minimize FPGA resource usage while maintaining accuracy.

    Args:
        model: FINN ModelWrapper to optimize
        cfg: Build configuration with optimization settings

    Returns:
        Optimized ModelWrapper

    Config:
        min_bits: Minimum bit width (default: 2)
        max_bits: Maximum bit width (default: 16)
    """
    min_bits = cfg.get('min_bits', 2)
    max_bits = cfg.get('max_bits', 16)

    logger.info(f"Minimizing bit widths (range: {min_bits}-{max_bits})")

    # Implementation
    for node in model.graph.node:
        # Optimize bit width
        pass

    return model
```

**Why Good**:
- ✓ Uses `@step` decorator
- ✓ Function signature: `(model, cfg) -> model`
- ✓ Comprehensive docstring (purpose, args, returns, config)
- ✓ Uses logger for progress
- ✓ Returns modified model
- ✓ Documents config parameters

#### ❌ **BAD** - Step Anti-Patterns

```python
# ❌ No decorator
def my_step(model, cfg):  # BAD: not registered
    return model

# ❌ Wrong signature
@step
def my_step(model):  # BAD: missing cfg parameter
    return model

# ❌ Doesn't return model
@step
def my_step(model, cfg):
    model.transform(SomeTransform())
    # BAD: not returning model

# ❌ No docstring
@step
def my_step(model, cfg):  # BAD: no documentation
    return model

# ❌ Prints instead of logging
@step
def my_step(model, cfg):
    print("Processing...")  # BAD: use logger
    return model

# ❌ Hard-coded config
@step
def my_step(model, cfg):
    threshold = 0.5  # BAD: should read from cfg
    return model
```

**Review Questions**:
- Does the step use `@step` decorator?
- Is the signature `(model, cfg) -> model`?
- Does it return the modified model?
- Is there a docstring with config parameters?
- Does it use `logger` instead of `print()`?
- Are config values read from `cfg` (not hard-coded)?

### 2.3 Backend Review

#### ✅ **GOOD** - Proper Backend Implementation

```python
from brainsmith import backend
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from .my_kernel import MyKernel

@backend
class MyKernel_hls(MyKernel, HLSBackend):
    """HLS backend for MyKernel.

    Generates Vivado HLS C++ code for FPGA synthesis.
    """
    target_kernel = 'project:MyKernel'
    language = 'hls'

    def generate_params(self, model, path):
        """Generate HLS parameters header."""
        # Generate params.h
        pass

    def execute_node(self, context, graph):
        """Execute node in C++ simulation."""
        # C++ sim execution
        pass

    def code_generation_ipi(self):
        """Generate IP integrator TCL script."""
        # Generate IPI script
        return "set ip_dir ..."
```

**Why Good**:
- ✓ Uses `@backend` decorator
- ✓ Inherits from both kernel and backend base class
- ✓ Defines `target_kernel` (namespaced correctly)
- ✓ Defines `language` ('hls' or 'rtl')
- ✓ Implements required methods
- ✓ Has docstring

#### ❌ **BAD** - Backend Anti-Patterns

```python
# ❌ No decorator
class MyKernel_hls(MyKernel, HLSBackend):  # BAD: not registered
    pass

# ❌ Missing target_kernel
@backend
class MyKernel_hls(MyKernel, HLSBackend):
    language = 'hls'
    # BAD: no target_kernel

# ❌ Wrong target_kernel format
@backend
class MyKernel_hls(MyKernel, HLSBackend):
    target_kernel = 'MyKernel'  # BAD: should be 'namespace:MyKernel'
    language = 'hls'

# ❌ Missing language
@backend
class MyKernel_hls(MyKernel, HLSBackend):
    target_kernel = 'project:MyKernel'
    # BAD: no language defined

# ❌ Wrong inheritance order
@backend
class MyKernel_hls(HLSBackend, MyKernel):  # BAD: kernel should be first
    pass
```

**Review Questions**:
- Does the backend use `@backend` decorator?
- Does it inherit from both kernel and backend base class?
- Is `target_kernel` defined with namespace (e.g., `project:MyKernel`)?
- Is `language` set correctly ('hls' or 'rtl')?
- Are required methods implemented?

---

## 3. Plugin Package Structure Review

### ✅ **GOOD** - Proper Package Structure

```
plugins/
├── __init__.py              # COMPONENTS dict + create_lazy_module
├── my_kernel.py             # Kernel implementation with @kernel
├── my_backend.py            # Backend implementation with @backend
├── my_step.py               # Step implementation with @step
├── tests/                   # Tests for components
│   ├── test_my_kernel.py
│   └── test_my_step.py
└── README.md                # Plugin documentation
```

**`__init__.py`** (only this, nothing more):
```python
from brainsmith.plugin_helpers import ComponentsDict, create_lazy_module

COMPONENTS: ComponentsDict = {
    'kernels': {'MyKernel': '.my_kernel'},
    'backends': {'MyKernel_hls': '.my_backend'},
    'steps': {'my_step': '.my_step'},
}

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

### ❌ **BAD** - Package Structure Mistakes

```python
# ❌ Eager imports in __init__.py
from .my_kernel import MyKernel  # BAD: defeats lazy loading
from .my_step import my_step     # BAD: defeats lazy loading

COMPONENTS = {
    'kernels': {'MyKernel': '.my_kernel'},
    'steps': {'my_step': '.my_step'},
}

# ❌ Both COMPONENTS and eager imports (pick one!)
from .my_kernel import MyKernel  # BAD: confusing
COMPONENTS = {'kernels': {'MyKernel': '.my_kernel'}}  # BAD: redundant

# ❌ No COMPONENTS dict (breaks discovery)
# __init__.py with just imports - BAD

# ❌ Manual registry access
from brainsmith.registry import registry
from .my_kernel import MyKernel
registry.kernel(MyKernel)  # BAD: use decorators
```

**Review Action**: Ensure `__init__.py` ONLY has COMPONENTS dict and create_lazy_module.

---

## 4. Testing Requirements Review

### ✅ **GOOD** - Comprehensive Tests

```python
import pytest
from brainsmith.loader import get_kernel, list_kernels

def test_kernel_discovery():
    """Verify kernel appears in discovery."""
    kernels = list_kernels(source='project')
    assert 'project:MyKernel' in kernels

def test_kernel_load():
    """Verify kernel can be loaded."""
    MyKernel = get_kernel('project:MyKernel')
    assert MyKernel is not None
    assert MyKernel.op_type == 'MyKernel'

def test_kernel_instantiation():
    """Verify kernel can be instantiated."""
    MyKernel = get_kernel('project:MyKernel')
    node = helper.make_node('MyKernel', ['input'], ['output'])
    kernel = MyKernel(node)
    assert kernel is not None

def test_kernel_execution():
    """Verify kernel executes correctly."""
    # Create test model
    # Execute kernel
    # Verify output
    pass

def test_step_discovery():
    """Verify step appears in discovery."""
    steps = list_steps(source='project')
    assert 'project:my_step' in steps

def test_step_execution():
    """Verify step executes correctly."""
    from brainsmith.loader import get_step

    step_fn = get_step('project:my_step')

    # Create test model
    model = create_test_model()
    cfg = {'param': 'value'}

    # Execute step
    result = step_fn(model, cfg)

    # Verify model was modified
    assert result is not None
```

**Why Good**:
- ✓ Tests discovery (component appears in list)
- ✓ Tests loading (component can be retrieved)
- ✓ Tests instantiation (component can be created)
- ✓ Tests execution (component actually works)
- ✓ Uses loader functions (not direct imports)

### ❌ **BAD** - Insufficient Tests

```python
# ❌ Only tests import (not discovery)
def test_my_kernel():
    from plugins.my_kernel import MyKernel  # BAD: not testing discovery
    assert MyKernel is not None

# ❌ No discovery test
def test_my_kernel():
    MyKernel = get_kernel('project:MyKernel')
    assert MyKernel is not None
    # BAD: should also test that it appears in list_kernels()

# ❌ No execution test
def test_my_step():
    step_fn = get_step('project:my_step')
    assert step_fn is not None
    # BAD: should actually call it with test model
```

**Review Questions**:
- Are there tests for discovery (list_* functions)?
- Are there tests for loading (get_* functions)?
- Are there tests for execution (actually running the component)?
- Do tests use the loader API (not direct imports)?

---

## 5. Performance Considerations Review

### ✅ **GOOD** - Performance Best Practices

```python
# ✓ Lazy loading (imports only when needed)
COMPONENTS = {
    'kernels': {'ExpensiveKernel': '.expensive'},  # Not imported until accessed
}

# ✓ Conditional imports inside functions
@step
def my_step(model, cfg):
    if cfg.get('use_torch', False):
        import torch  # Only import if needed
        # Use torch
    return model

# ✓ Cache expensive computations
@kernel
class MyKernel(HWCustomOp):
    def __init__(self, onnx_node):
        super().__init__(onnx_node)
        self._cache = {}  # Instance-level cache

    def expensive_operation(self):
        if 'result' not in self._cache:
            self._cache['result'] = self._compute()
        return self._cache['result']
```

### ❌ **BAD** - Performance Killers

```python
# ❌ Eager imports (slows module load)
import torch  # BAD: imported even if never used
import numpy as np  # BAD: imported even if never used
from scipy import signal  # BAD: expensive import

COMPONENTS = {
    'kernels': {'MyKernel': '.my_kernel'},
}

# ❌ Top-level expensive computation
LOOKUP_TABLE = generate_huge_lookup_table()  # BAD: runs at import time

COMPONENTS = {'kernels': {'MyKernel': '.my_kernel'}}

# ❌ Import * at top level
from finn.custom_op.fpgadataflow import *  # BAD: imports everything
```

**Review Questions**:
- Are heavy dependencies (torch, numpy) imported lazily?
- Are expensive computations done at import time?
- Does the plugin cause slow startup?

---

## 6. Documentation Requirements Review

### ✅ **GOOD** - Well Documented

**Component docstring**:
```python
@kernel
class MyKernel(HWCustomOp):
    """Hardware-accelerated custom operation.

    Implements [describe what it does] for FPGA deployment.
    Optimized for [specific use case].

    Node Attributes:
        param1: Description of param1
        param2: Description of param2

    Input Tensors:
        - input[0]: Input tensor shape (batch, channels, height, width)

    Output Tensors:
        - output[0]: Output tensor shape (batch, channels, height, width)

    Example:
        >>> # Create node
        >>> node = make_node('MyKernel', ['input'], ['output'],
        ...                  param1=42, param2='value')
    """
```

**Step docstring**:
```python
@step
def my_step(model, cfg):
    """Optimize model for specific platform.

    Applies platform-specific optimizations including [list optimizations].

    Args:
        model: FINN ModelWrapper to optimize
        cfg: Build configuration

    Returns:
        Optimized ModelWrapper

    Config:
        param1: Description (type: int, default: 42)
        param2: Description (type: str, required)
        platform: Target platform (default: 'zynq')

    Example:
        >>> cfg = {'param2': 'value', 'platform': 'alveo'}
        >>> model = my_step(model, cfg)
    """
```

**Plugin README**:
```markdown
# My Plugin

Custom kernels and steps for [purpose].

## Components

### Kernels
- **MyKernel**: Description

### Steps
- **my_step**: Description

## Installation

Add to brainsmith.yaml:
\`\`\`yaml
plugins:
  - path: /path/to/plugin
    name: my_plugin
\`\`\`

## Usage

See examples/ directory.
```

### ❌ **BAD** - Insufficient Documentation

```python
# ❌ No docstring
@kernel
class MyKernel(HWCustomOp):
    op_type = "MyKernel"

# ❌ Minimal docstring
@step
def my_step(model, cfg):
    """Does stuff."""  # BAD: not helpful
    return model

# ❌ No README in plugin
# Missing: README.md explaining what plugin does
```

**Review Questions**:
- Does each component have a comprehensive docstring?
- Are config parameters documented?
- Is there a README explaining the plugin?
- Are there usage examples?

---

## 7. Common Issues Checklist

### Import Issues
- [ ] No circular imports between components
- [ ] All imports are either top-level (type hints) or lazy (inside functions)
- [ ] No `import *` in plugin code

### Registration Issues
- [ ] All components use decorators (not manual registry calls)
- [ ] COMPONENTS dict includes all components
- [ ] Component names in COMPONENTS match class/function names
- [ ] No duplicate registrations

### Naming Issues
- [ ] Backend names follow `{Kernel}_{language}` convention
- [ ] Step names use snake_case
- [ ] Kernel names use PascalCase
- [ ] Namespaced names use `namespace:ComponentName` format

### Configuration Issues
- [ ] Steps read config from `cfg` parameter (not hard-coded)
- [ ] Config parameters have sensible defaults
- [ ] Required config parameters are documented

### Error Handling
- [ ] Components validate inputs
- [ ] Errors have helpful messages
- [ ] Failures don't crash the entire system

---

## 8. Backward Compatibility Review

**Breaking changes to avoid**:
- ❌ Renaming existing components
- ❌ Removing components without deprecation
- ❌ Changing component signatures
- ❌ Changing config parameter names
- ❌ Changing component behavior without migration path

**Safe changes**:
- ✓ Adding new components
- ✓ Adding new optional config parameters
- ✓ Adding new optional methods to components
- ✓ Improving error messages
- ✓ Performance optimizations (same behavior)
- ✓ Documentation improvements

**Review Question**: Does this PR break existing workflows?

---

## 9. Validation Tools

### Manual Validation

```bash
# Check COMPONENTS dict is valid
BRAINSMITH_VALIDATE_PLUGINS=1 python -c "import plugins"

# Verify components appear in discovery
poetry run brainsmith plugins --verbose

# Test component loading
poetry run python -c "
from brainsmith.loader import get_kernel
k = get_kernel('project:MyKernel')
print(f'✓ Loaded: {k}')
"

# Run tests
poetry run pytest plugins/tests/
```

### Automated Validation

Add to CI/CD pipeline:
```yaml
- name: Validate plugins
  run: |
    export BRAINSMITH_VALIDATE_PLUGINS=1
    poetry run python -c "import plugins"
    poetry run brainsmith plugins --verbose
    poetry run pytest plugins/tests/
```

---

## 10. Review Comments Templates

**For COMPONENTS dict issues**:
```
The COMPONENTS dict should use relative imports (starting with '.').

Change:
  'MyKernel': 'plugins.my_kernel'
To:
  'MyKernel': '.my_kernel'

See: docs/plugin-system-guide.md#components-dict
```

**For missing decorators**:
```
Components should use decorators for registration, not manual registry calls.

Change:
  class MyKernel(HWCustomOp):
      pass
  registry.kernel(MyKernel)

To:
  @kernel
  class MyKernel(HWCustomOp):
      pass

See: docs/plugin-system-guide.md#registration
```

**For eager imports**:
```
Plugin __init__.py should use lazy loading, not eager imports.

Remove:
  from .my_kernel import MyKernel

Keep only:
  COMPONENTS = {'kernels': {'MyKernel': '.my_kernel'}}
  __getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)

See: docs/plugin-system-guide.md#lazy-loading
```

**For missing tests**:
```
Please add tests for component discovery and execution.

Required tests:
- Test component appears in list_kernels()/list_steps()
- Test component can be loaded with get_kernel()/get_step()
- Test component executes correctly with sample input

See: docs/plugin-system-guide.md#testing
```

---

## Summary

**Arete Code Review Principles**:

1. **One Clear Pattern** - All plugins use COMPONENTS dict + create_lazy_module
2. **Lazy Loading** - No eager imports in __init__.py
3. **Decorator Registration** - Use @kernel/@step/@backend (not manual registry)
4. **Type Safety** - ComponentsDict annotation on COMPONENTS
5. **Comprehensive Tests** - Discovery, loading, and execution
6. **Good Documentation** - Docstrings with config parameters
7. **Performance** - Lazy imports, cached computations
8. **Backward Compatibility** - No breaking changes without migration

**When in doubt**: Refer to `brainsmith/kernels/__init__.py` as the canonical example.

---

## Quick Reference Links

- **Plugin System Guide**: `docs/plugin-system-guide.md`
- **Plugin Quickstart**: `docs/plugin-quickstart.md`
- **Phase 1 Summary**: `_artifacts/analyses/phase1-foundation-hardening-summary.md`
- **Example Plugin**: `plugins/` directory
- **Core Implementation**: `brainsmith/plugin_helpers.py`

**For questions**: See `brainsmith/kernels/__init__.py` and `brainsmith/steps/__init__.py` as reference implementations.
