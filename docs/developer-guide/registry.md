# Component Registry

The **Component Registry** enables declarative blueprint construction for FPGA accelerators:

```yaml
# blueprint.yaml
design_space:
  kernels: [LayerNorm, Softmax]
  steps: [streamline, infer_kernels]
```

**Component Types**: Steps (pipeline transforms), Kernels (hardware operators), Backends (HLS/RTL code generators)

**Key Benefits**: Parse-time validation (fail fast via `has_step()`), source priority (project overrides core), automatic discovery via decorators

## Component Sources

| Source | Type | Configuration | Caching |
|--------|------|---------------|---------|
| `brainsmith` | Core framework | Automatic (direct import) | Yes |
| `finn` (or custom) | Entry point plugins | `setup.cfg`: `brainsmith.plugins = pkg.module:func` | Yes |
| `project` | Filesystem | Automatic (`kernels/`, `steps/` subdirs) | Yes |
| Custom (e.g., `team`) | Filesystem | `component_sources.team = "/path"` in config | Yes |
| `custom` | Runtime registration | Programmatic (no namespace/domain) | No |

**Source priority** (configurable): `['project', 'brainsmith', 'finn', 'custom']` - first match wins for short names

## Registering Components

**Using components:**
```python
from brainsmith.registry import get_step, get_kernel, list_backends_for_kernel

streamline = get_step('streamline')
LayerNorm = get_kernel('LayerNorm')
backends = list_backends_for_kernel('LayerNorm', language='hls')
```

**Registering components via decorators:**
```python
from brainsmith.registry import kernel, backend, step

@kernel
class MyKernel(HWCustomOp):
    op_type = "MyKernel"

@backend(target_kernel='MyKernel', language='hls')
class MyKernel_hls:
    pass

@step
def my_step(model, **config):
    return model
```

**Registering via entry points** (for pip packages):
```python
# setup.cfg
[options.entry_points]
brainsmith.plugins =
    myplugin = myplugin.registry:register_components
```

**Discovery**: Automatic on first use. Manual refresh: `discover_components(force_refresh=True)`

**Configuration** (`brainsmith.yaml`): `cache_components` (default: true), `components_strict` (default: true), `source_priority`, `component_sources`

---

## API Patterns

All components follow `get_*/has_*/list_*` pattern:
- **get** - Load component (imports module): `get_step()`, `get_kernel()`, `get_backend()`
- **has** - Check existence (no import): `has_step()`, `has_kernel()`
- **list** - Enumerate: `list_steps()`, `list_kernels()`, `list_backends()`

**Name resolution**: Short names use source priority (`LayerNorm` → project > brainsmith > finn). Qualified names force source (`brainsmith:LayerNorm`).

**Special**: `get_kernel_infer()` for ONNX transforms, `list_backends_for_kernel(name, language='hls')` for backend discovery, `get_component_metadata()` for inspection.

---

## Advanced Patterns

**Dynamic backend selection:**
```python
backends = list_backends_for_kernel('LayerNorm', language='hls')
backend_cls = get_backend(backends[0]) if backends else None
```

**Plugin inventory:**
```python
for src in ['brainsmith', 'finn', 'project']:
    print(f"{src}: {len(list_kernels(source=src))} kernels")
```

**Component introspection:** Use `get_component_metadata(name, type)` to inspect without loading.

---

**See also**: [Component Registry API](../api/registry.md), [Settings API](../api/settings.md), [Hardware Kernels Guide](hardware-kernels.md)
