# Implementation Plan: `brainsmith plugins` Performance Optimization

**Goal:** Reduce `brainsmith plugins` execution time from 8.4s to <300ms (96% improvement)

**Strategy:** Implement lazy loading architecture with plugin metadata manifest

**Reference:** See `/tmp/latency_profile_report.md` for detailed performance analysis

---

## Phase 1: Foundation - Lazy Kernel Loading (2.6s savings, 31% improvement)

**Target:** 8.4s → 5.8s

### 1.1 Create Lazy Kernel Registry

**File:** `brainsmith/kernels/__init__.py`

**Current Problem:**
```python
# Eager loading - ALL imports happen immediately
from .crop import *        # 0.43s - imports numpy, onnx, finn
from .layernorm import *   # 1.94s - imports torch
from .shuffle import *
from .softmax import *     # 0.24s - imports scipy
```

**Solution:**
```python
"""Brainsmith Kernels - Lazy Loading Registry"""

# Lazy registry - NO imports until actually needed
_KERNEL_MODULES = {
    'crop': 'brainsmith.kernels.crop',
    'layernorm': 'brainsmith.kernels.layernorm',
    'shuffle': 'brainsmith.kernels.shuffle',
    'softmax': 'brainsmith.kernels.softmax',
}

_loaded_modules = {}

def __getattr__(name):
    """Lazy import kernel modules on attribute access."""
    if name in _KERNEL_MODULES:
        if name not in _loaded_modules:
            import importlib
            module_path = _KERNEL_MODULES[name]
            _loaded_modules[name] = importlib.import_module(module_path)
        return _loaded_modules[name]
    raise AttributeError(f"module 'brainsmith.kernels' has no attribute '{name}'")

def __dir__():
    """Support dir() and tab completion."""
    return list(_KERNEL_MODULES.keys())

# Provide metadata without importing implementations
KERNEL_METADATA = {
    'crop': {'op_types': ['Gather'], 'description': 'Crop operations'},
    'layernorm': {'op_types': ['LayerNormalization'], 'description': 'Layer normalization'},
    'shuffle': {'op_types': ['Reshape', 'Transpose'], 'description': 'Channel shuffle'},
    'softmax': {'op_types': ['Softmax'], 'description': 'Softmax activation'},
}
```

**Changes Required:**
- Modify `brainsmith/kernels/__init__.py` to use lazy loading
- Update any code that does `from brainsmith.kernels import *` to specific imports
- Verify backward compatibility with `__getattr__` hook

**Testing:**
```python
# Test lazy loading
import brainsmith.kernels
assert 'crop' not in sys.modules  # Not loaded yet
crop = brainsmith.kernels.crop    # Loads on access
assert 'brainsmith.kernels.crop' in sys.modules  # Now loaded
```

**Impact:** Saves 2.6s when kernels aren't needed (e.g., `plugins` command)

**Risk:** Low - Python's `__getattr__` is well-supported, maintains API compatibility

---

## Phase 2: Plugin Metadata Manifest (5s savings, 90% total improvement)

**Target:** 5.8s → 0.8s

### 2.1 Design Manifest Schema

**File:** `brainsmith/manifest.py` (new)

```python
"""Plugin metadata manifest for fast discovery without imports."""

from typing import TypedDict, Dict, List, Optional
from pathlib import Path
from datetime import datetime

class ComponentMetadata(TypedDict):
    """Metadata for a single component."""
    name: str
    type: str  # 'step', 'kernel', 'backend'
    module: str  # Full module path for lazy loading
    op_types: List[str]  # For kernels
    target_kernel: Optional[str]  # For backends
    language: Optional[str]  # For backends

class SourceManifest(TypedDict):
    """Manifest for a single plugin source."""
    source: str
    version: str
    steps: List[ComponentMetadata]
    kernels: List[ComponentMetadata]
    backends: List[ComponentMetadata]

class PluginManifest(TypedDict):
    """Complete plugin manifest."""
    version: str  # Manifest format version
    generated_at: str  # ISO timestamp
    sources: Dict[str, SourceManifest]
```

### 2.2 Implement Manifest Generator

**File:** `brainsmith/manifest.py`

```python
MANIFEST_VERSION = "1.0"

def generate_manifest() -> PluginManifest:
    """Generate plugin manifest by doing full discovery.

    This is expensive but only run during:
    - Package build/install
    - Explicit refresh command
    - First run if no manifest exists
    """
    from brainsmith.loader import discover_plugins
    from brainsmith.registry import registry

    # Trigger full discovery (expensive)
    discover_plugins()

    manifest: PluginManifest = {
        'version': MANIFEST_VERSION,
        'generated_at': datetime.utcnow().isoformat(),
        'sources': {}
    }

    # Extract metadata from registry (already loaded)
    for source in registry.list_sources():
        manifest['sources'][source] = {
            'source': source,
            'version': _get_source_version(source),
            'steps': _extract_steps_metadata(source),
            'kernels': _extract_kernels_metadata(source),
            'backends': _extract_backends_metadata(source),
        }

    return manifest

def save_manifest(manifest: PluginManifest, path: Path):
    """Save manifest to JSON file."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(manifest, f, indent=2)

def load_manifest(path: Path) -> Optional[PluginManifest]:
    """Load manifest from JSON file."""
    import json
    if not path.exists():
        return None
    with path.open('r') as f:
        return json.load(f)

def get_manifest_path() -> Path:
    """Get path to manifest file."""
    import brainsmith
    # Store in package directory
    package_dir = Path(brainsmith.__file__).parent
    return package_dir / '.plugin_manifest.json'
```

### 2.3 Integrate Manifest into Plugin Discovery

**File:** `brainsmith/loader.py`

**Modify `discover_plugins()` to use manifest:**

```python
def discover_plugins(force_full: bool = False):
    """Discover and load all plugins from configured sources.

    Args:
        force_full: Force full discovery even if manifest exists
    """
    global _plugins_discovered
    if _plugins_discovered:
        return

    # Try manifest-based fast path first (for 'plugins' command)
    if not force_full:
        manifest = _try_manifest_discovery()
        if manifest:
            _plugins_discovered = True
            logger.info(f"Plugin discovery complete (fast path via manifest)")
            return

    # Fall back to full discovery (for actual execution)
    _full_discovery()
    _plugins_discovered = True

def _try_manifest_discovery() -> Optional[PluginManifest]:
    """Try to load plugins from manifest without imports.

    Returns manifest if successful, None if needs full discovery.
    """
    from brainsmith.manifest import load_manifest, get_manifest_path

    manifest_path = get_manifest_path()
    manifest = load_manifest(manifest_path)

    if not manifest:
        logger.debug("No manifest found, falling back to full discovery")
        return None

    # Validate manifest is current
    if not _is_manifest_valid(manifest):
        logger.debug("Manifest outdated, falling back to full discovery")
        return None

    # Register components from manifest (NO IMPORTS)
    _register_from_manifest(manifest)

    return manifest

def _register_from_manifest(manifest: PluginManifest):
    """Register components from manifest without importing implementations."""
    from brainsmith.registry import registry

    for source_name, source_data in manifest['sources'].items():
        # Register metadata only - implementations loaded lazily on demand
        for step in source_data['steps']:
            registry.register_step_metadata(
                source=source_name,
                name=step['name'],
                module=step['module'],
            )

        for kernel in source_data['kernels']:
            registry.register_kernel_metadata(
                source=source_name,
                name=kernel['name'],
                op_types=kernel.get('op_types', []),
                module=kernel['module'],
            )

        for backend in source_data['backends']:
            registry.register_backend_metadata(
                source=source_name,
                name=backend['name'],
                target_kernel=backend.get('target_kernel'),
                language=backend.get('language'),
                module=backend['module'],
            )

def _full_discovery():
    """Full plugin discovery with imports (original behavior)."""
    logger.info("Discovering plugins (full discovery)...")

    # Original implementation
    import brainsmith.kernels
    import brainsmith.steps
    _load_user_plugins()
    _load_entry_point_plugins()
    _process_deferred_registrations()

def _is_manifest_valid(manifest: PluginManifest) -> bool:
    """Check if manifest is still valid."""
    from brainsmith.manifest import MANIFEST_VERSION

    # Version check
    if manifest.get('version') != MANIFEST_VERSION:
        logger.debug(f"Manifest version mismatch: {manifest.get('version')} != {MANIFEST_VERSION}")
        return False

    # Check plugin package versions
    for source, data in manifest['sources'].items():
        current_version = _get_installed_version(source)
        if current_version and current_version != data.get('version'):
            logger.debug(f"Source '{source}' version mismatch")
            return False

    # Check file mtimes (if editable install)
    if is_editable_install():
        manifest_time = datetime.fromisoformat(manifest['generated_at'])
        if _has_newer_sources(manifest_time):
            logger.debug("Source files newer than manifest")
            return False

    return True
```

### 2.4 Update Registry for Lazy Loading

**File:** `brainsmith/registry.py`

Add support for metadata-only registration:

```python
class Registry:
    def __init__(self):
        self._steps = {}
        self._kernels = {}
        self._backends = {}

        # Lazy loading support
        self._step_modules = {}      # name -> module path
        self._kernel_modules = {}    # name -> module path
        self._backend_modules = {}   # name -> module path
        self._loaded_components = set()  # Track what's been imported

    def register_step_metadata(self, source: str, name: str, module: str):
        """Register step metadata without loading implementation."""
        full_name = f"{source}:{name}"
        self._steps[full_name] = {
            'name': name,
            'source': source,
            'module': module,
            'loaded': False
        }
        self._step_modules[full_name] = module

    def register_kernel_metadata(self, source: str, name: str, module: str, op_types: List[str]):
        """Register kernel metadata without loading implementation."""
        full_name = f"{source}:{name}"
        self._kernels[full_name] = {
            'name': name,
            'source': source,
            'module': module,
            'op_types': op_types,
            'loaded': False
        }
        self._kernel_modules[full_name] = module

    def register_backend_metadata(self, source: str, name: str, module: str,
                                   target_kernel: Optional[str] = None,
                                   language: Optional[str] = None):
        """Register backend metadata without loading implementation."""
        full_name = f"{source}:{name}"
        self._backends[full_name] = {
            'name': name,
            'source': source,
            'module': module,
            'target_kernel': target_kernel,
            'language': language,
            'loaded': False
        }
        self._backend_modules[full_name] = module

    def get_step(self, name: str):
        """Get step, loading implementation if needed."""
        if name not in self._loaded_components:
            self._load_component(name, 'step')
        return self._steps[name]

    def _load_component(self, full_name: str, component_type: str):
        """Lazy-load component implementation when first accessed."""
        if full_name in self._loaded_components:
            return

        if component_type == 'step':
            module_path = self._step_modules.get(full_name)
            meta = self._steps[full_name]
        elif component_type == 'kernel':
            module_path = self._kernel_modules.get(full_name)
            meta = self._kernels[full_name]
        elif component_type == 'backend':
            module_path = self._backend_modules.get(full_name)
            meta = self._backends[full_name]
        else:
            raise ValueError(f"Unknown component type: {component_type}")

        if module_path:
            import importlib
            logger.debug(f"Lazy-loading {component_type} {full_name} from {module_path}")
            importlib.import_module(module_path)
            self._loaded_components.add(full_name)
            meta['loaded'] = True

    def list_step_names(self, source: Optional[str] = None) -> List[str]:
        """List step names without loading implementations."""
        if source:
            return [
                name for name, meta in self._steps.items()
                if meta['source'] == source
            ]
        return list(self._steps.keys())

    def get_component_counts(self) -> Dict[str, Dict[str, int]]:
        """Get counts by source without loading implementations."""
        counts = {}
        for source in self.list_sources():
            counts[source] = {
                'steps': len([s for s, m in self._steps.items() if m['source'] == source]),
                'kernels': len([k for k, m in self._kernels.items() if m['source'] == source]),
                'backends': len([b for b, m in self._backends.items() if m['source'] == source]),
            }
        return counts
```

### 2.5 Manifest Generation Hook

**Add to `brainsmith/__init__.py`:**

```python
def _ensure_manifest():
    """Generate manifest on first import if it doesn't exist."""
    from brainsmith.manifest import get_manifest_path, load_manifest, generate_manifest, save_manifest

    manifest_path = get_manifest_path()

    # Skip if manifest exists and is valid
    manifest = load_manifest(manifest_path)
    if manifest:
        return

    # Generate fresh manifest
    logger.info("Generating plugin manifest (first run)...")
    try:
        manifest = generate_manifest()
        save_manifest(manifest, manifest_path)
        logger.info(f"Manifest saved to {manifest_path}")
    except Exception as e:
        logger.warning(f"Failed to generate manifest: {e}")
        # Continue without manifest - will use full discovery

# Call on import
_ensure_manifest()
```

**Add CLI command:**

```python
# brainsmith/cli/commands/plugins.py

@click.command()
def refresh_manifest():
    """Regenerate the plugin manifest."""
    from brainsmith.manifest import generate_manifest, save_manifest, get_manifest_path

    console.print("[cyan]Regenerating plugin manifest...[/cyan]")
    manifest = generate_manifest()
    manifest_path = get_manifest_path()
    save_manifest(manifest, manifest_path)
    console.print(f"[green]✓[/green] Manifest saved to {manifest_path}")

    # Show summary
    source_count = len(manifest['sources'])
    total_components = sum(
        len(s['steps']) + len(s['kernels']) + len(s['backends'])
        for s in manifest['sources'].values()
    )
    console.print(f"Discovered {total_components} components from {source_count} sources")
```

**Impact:** Saves ~5s by avoiding all plugin imports for listing

**Risk:** Medium - Requires manifest to stay in sync with code

---

## Phase 3: FINN Integration Optimization (0.5s additional savings)

**Target:** 0.8s → 0.3s

### 3.1 Optimize FINN Entry Point

**Problem:** `finn.util.brainsmith_integration:register_all` imports `finn.builder.build_dataflow_steps` which imports `finn.util.test` (5.07s)

**File:** `deps/finn/src/finn/util/brainsmith_integration.py`

**Current:**
```python
def _discover_steps():
    from finn.builder import build_dataflow_steps  # ← 5.19s import!

    steps = []
    for name, func in inspect.getmembers(build_dataflow_steps, inspect.isfunction):
        if name.startswith('step_'):
            steps.append({'name': name[5:], 'func': func})
    return steps
```

**Solution 1: Static metadata (preferred)**
```python
# finn/util/brainsmith_integration.py

# Hardcoded metadata - NO imports needed
_FINN_STEPS = [
    'qonnx_to_finn',
    'tidy_up',
    'streamline',
    'convert_to_hw',
    'create_dataflow_partition',
    'specialize_layers',
    'target_fps_parallelization',
    'apply_folding_config',
    'minimize_bit_width',
    'generate_estimate_reports',
    'hw_codegen',
    'hw_ipgen',
    'set_fifo_depths',
    'create_stitched_ip',
    'measure_rtlsim_performance',
    'out_of_context_synthesis',
    'synthesize_bitfile',
    'make_driver',
    'deployment_package',
]

def _discover_steps():
    """Return FINN step metadata without imports."""
    return [
        {
            'name': step_name,
            'module': 'finn.builder.build_dataflow_steps',
            'func_name': f'step_{step_name}',
        }
        for step_name in _FINN_STEPS
    ]
```

**Solution 2: Lazy AST parsing** (if dynamic discovery needed)
```python
import ast
from pathlib import Path

def _discover_steps_via_ast():
    """Discover steps by parsing AST without importing."""
    steps_file = Path(__file__).parent.parent / 'builder' / 'build_dataflow_steps.py'

    with steps_file.open('r') as f:
        tree = ast.parse(f.read())

    steps = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('step_'):
            steps.append({
                'name': node.name[5:],
                'module': 'finn.builder.build_dataflow_steps',
                'func_name': node.name,
            })

    return steps
```

### 3.2 Fix FINN Test Utils Import

**Investigation:** Find why `finn.util.test` is imported in production code

```bash
# Find the import
grep -rn "from finn.util.test" deps/finn/src/finn/builder/
grep -rn "import finn.util.test" deps/finn/src/finn/builder/
```

**Action:** Submit PR to FINN to:
1. Move test-only imports inside functions
2. Use `TYPE_CHECKING` guards
3. Separate test utilities from production utilities

### 3.3 Similar Optimization for Kernels/Backends

Apply same static metadata pattern to `_register_kernels()` and `_register_backends()`

**Impact:** Avoids 5s of FINN imports, reduces to <0.1s for metadata

**Risk:** Low for static metadata, Medium for AST parsing

---

## Phase 4: Update Plugins Command

**File:** `brainsmith/cli/commands/plugins.py`

```python
@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--verbose', '-v', is_flag=True, help='Show detailed component information')
@click.option('--refresh-manifest', is_flag=True, help='Regenerate plugin manifest')
@click.pass_obj
def plugins(app_ctx: ApplicationContext, verbose: bool, refresh_manifest: bool) -> None:
    """Show plugin information."""

    if refresh_manifest:
        from brainsmith.manifest import generate_manifest, save_manifest, get_manifest_path
        console.print("[cyan]Regenerating plugin manifest...[/cyan]")
        manifest = generate_manifest()
        save_manifest(manifest, get_manifest_path())
        console.print("[green]✓[/green] Manifest regenerated")
        return

    config = app_ctx.get_effective_config()

    if verbose:
        # Full discovery needed for detailed info
        discover_plugins(force_full=True)
        all_steps = list_steps()
        all_kernels = list_kernels()
        all_backends = list_all_backends()
        # ... existing verbose output ...
    else:
        # Fast path - use manifest for summary only
        discover_plugins(force_full=False)

        # Use fast registry methods (no implementation loading)
        from brainsmith.registry import registry
        all_steps = registry.list_step_names()
        all_kernels = registry.list_kernel_names()
        all_backends = registry.list_backend_names()

    # ... rest of display logic remains same ...
```

---

## Implementation Order & Timeline

### Sprint 1: Foundation (Week 1)
**Goal:** Get basic lazy loading working

**Tasks:**
1. ✅ Create latency profile analysis
2. ✅ Document current architecture
3. Implement lazy kernel loading
   - Modify `brainsmith/kernels/__init__.py`
   - Add `__getattr__` hook
   - Add `KERNEL_METADATA`
4. Add tests for lazy loading
5. Measure and verify 2.6s improvement

**Deliverables:**
- `brainsmith plugins` runs in ~5.8s
- All existing tests pass
- Kernels load on-demand when actually used

**Success Criteria:**
- Performance improvement measured
- No test failures
- Backward compatibility verified

### Sprint 2: Manifest Infrastructure (Week 2)
**Goal:** Build manifest system

**Tasks:**
1. Create `brainsmith/manifest.py`
2. Implement manifest schema (TypedDict)
3. Implement `generate_manifest()`
4. Implement `save_manifest()` and `load_manifest()`
5. Add `get_manifest_path()`
6. Add CLI command `brainsmith plugins --refresh-manifest`
7. Unit tests for manifest generation/loading

**Deliverables:**
- Manifest can be generated and loaded
- Manifest contains all plugin metadata
- CLI command works

**Success Criteria:**
- Manifest generates successfully
- Manifest roundtrips through JSON
- All plugin metadata captured

### Sprint 3: Registry Integration (Week 3)
**Goal:** Integrate manifest with discovery

**Tasks:**
1. Add metadata registration to Registry
   - `register_step_metadata()`
   - `register_kernel_metadata()`
   - `register_backend_metadata()`
2. Implement lazy component loading
   - `_load_component()`
   - Track loaded components
3. Modify `loader.py`:
   - Add `_try_manifest_discovery()`
   - Add `_register_from_manifest()`
   - Add `_is_manifest_valid()`
   - Update `discover_plugins(force_full=False)`
4. Update `plugins` command to use fast path
5. Integration tests

**Deliverables:**
- `brainsmith plugins` runs in ~0.8s
- `brainsmith run` still works (full discovery)
- Tests pass

**Success Criteria:**
- Fast path works without imports
- Slow path works for actual execution
- All tests pass

### Sprint 4: FINN Optimization (Week 4)
**Goal:** Optimize FINN entry point

**Tasks:**
1. Investigate FINN import chain
2. Implement static metadata for FINN steps
3. Identify and document `finn.util.test` import
4. Create PR for FINN with fixes
5. Test FINN plugins work correctly
6. Coordinate with FINN team

**Deliverables:**
- `brainsmith plugins` runs in <0.3s
- FINN plugins still work correctly
- PR submitted to FINN repo

**Success Criteria:**
- Target latency achieved
- FINN integration working
- Upstream PR accepted

### Sprint 5: Polish & Documentation (Week 5)
**Goal:** Production-ready

**Tasks:**
1. Comprehensive testing
   - Unit tests
   - Integration tests
   - Performance benchmarks
2. Manifest invalidation logic
   - Version checks
   - File mtime checks
   - Auto-regeneration
3. Error handling and fallbacks
   - Graceful degradation
   - Clear error messages
4. Documentation
   - User guide updates
   - Developer docs
   - Migration guide
5. Performance benchmarking
   - Before/after metrics
   - CI/CD integration

**Deliverables:**
- All tests pass
- Documentation complete
- Performance target achieved (<300ms)
- Backwards compatible

**Success Criteria:**
- 100% test pass rate
- Performance targets met
- Documentation approved
- Ready for merge

---

## Testing Strategy

### Unit Tests

```python
# tests/test_lazy_kernels.py
import sys

def test_lazy_kernel_import():
    """Verify kernels aren't imported until accessed."""
    # Clear cache
    for mod in list(sys.modules.keys()):
        if 'brainsmith.kernels' in mod and mod != 'brainsmith.kernels':
            del sys.modules[mod]

    import brainsmith.kernels

    # Should not be loaded yet
    assert 'brainsmith.kernels.layernorm' not in sys.modules

    # Access triggers load
    _ = brainsmith.kernels.layernorm

    # Now should be loaded
    assert 'brainsmith.kernels.layernorm' in sys.modules

def test_lazy_kernel_getattr():
    """Test __getattr__ hook works."""
    import brainsmith.kernels

    # Should raise for unknown kernel
    with pytest.raises(AttributeError):
        _ = brainsmith.kernels.nonexistent

    # Should work for real kernels
    assert hasattr(brainsmith.kernels, 'crop')

# tests/test_manifest.py
def test_manifest_generation():
    """Verify manifest can be generated."""
    from brainsmith.manifest import generate_manifest

    manifest = generate_manifest()

    assert 'version' in manifest
    assert 'sources' in manifest
    assert 'brainsmith' in manifest['sources']
    assert 'steps' in manifest['sources']['brainsmith']
    assert isinstance(manifest['sources']['brainsmith']['steps'], list)

def test_manifest_roundtrip():
    """Verify manifest can be saved and loaded."""
    from brainsmith.manifest import generate_manifest, save_manifest, load_manifest
    import tempfile

    manifest = generate_manifest()

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        path = Path(f.name)

    save_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded == manifest
    path.unlink()

def test_manifest_discovery():
    """Verify discovery works from manifest."""
    from brainsmith.loader import discover_plugins, _plugins_discovered
    from brainsmith.registry import registry

    # Reset state
    globals()['_plugins_discovered'] = False

    discover_plugins(force_full=False)  # Use manifest

    # Should have components registered
    assert len(registry.list_step_names()) > 0
    assert len(registry.list_kernel_names()) > 0

# tests/test_backward_compat.py
def test_existing_imports_still_work():
    """Verify existing import patterns still work."""
    # These should all still work despite lazy loading
    from brainsmith.kernels import crop
    from brainsmith.kernels.layernorm import LayerNorm
    import brainsmith.kernels

    assert hasattr(brainsmith.kernels, 'crop')
    assert crop is not None

def test_wildcard_imports_still_work():
    """Verify star imports still work (if supported)."""
    # This may need adjustment based on implementation
    # Some lazy loading patterns don't support star imports
    from brainsmith import kernels
    assert hasattr(kernels, 'crop')
```

### Integration Tests

```python
# tests/integration/test_plugins_performance.py
def test_plugins_command_performance():
    """Verify plugins command is fast."""
    import time
    import subprocess

    start = time.time()
    result = subprocess.run(
        ['brainsmith', 'plugins'],
        capture_output=True,
        timeout=2.0,  # Should complete < 2s
        cwd='/home/tafk/dev/brainsmith-2'
    )
    elapsed = time.time() - start

    assert result.returncode == 0
    assert elapsed < 1.0, f"plugins command took {elapsed:.2f}s, expected <1.0s"

def test_run_command_still_works():
    """Verify full discovery works for actual execution."""
    import subprocess

    # Run a simple pipeline (need test fixture)
    result = subprocess.run(
        ['brainsmith', 'run', 'tests/fixtures/simple_config.yaml'],
        capture_output=True,
        cwd='/home/tafk/dev/brainsmith-2'
    )

    assert result.returncode == 0

def test_manifest_refresh_command():
    """Verify manifest refresh works."""
    import subprocess

    result = subprocess.run(
        ['brainsmith', 'plugins', '--refresh-manifest'],
        capture_output=True,
        cwd='/home/tafk/dev/brainsmith-2'
    )

    assert result.returncode == 0
    assert b'regenerat' in result.stdout.lower() or b'refresh' in result.stdout.lower()
```

### Performance Benchmarks

```python
# benchmarks/bench_plugins.py
import time
import statistics
import subprocess

def bench_plugins_command(runs=10):
    """Benchmark plugins command."""
    times = []

    for i in range(runs):
        start = time.time()
        subprocess.run(
            ['brainsmith', 'plugins'],
            capture_output=True,
            cwd='/home/tafk/dev/brainsmith-2'
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Run {i+1}/{runs}: {elapsed:.3f}s")

    avg = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    median = statistics.median(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nResults:")
    print(f"  Average: {avg:.3f}s ± {stdev:.3f}s")
    print(f"  Median:  {median:.3f}s")
    print(f"  Min:     {min_time:.3f}s")
    print(f"  Max:     {max_time:.3f}s")

    assert avg < 0.5, f"Average time {avg:.3f}s exceeds target of 0.5s"

if __name__ == '__main__':
    bench_plugins_command()
```

---

## Success Metrics

### Performance Targets
- ✅ `brainsmith plugins`: <300ms (currently 8400ms) - **96% improvement**
- ✅ `brainsmith plugins --verbose`: <10s (full discovery OK)
- ✅ `brainsmith run`: <1s overhead (currently ~8s)
- ✅ First kernel instantiation: adds <100ms

### Reliability Targets
- ✅ 100% test pass rate
- ✅ Backwards compatible with existing code
- ✅ No regressions in CI/CD
- ✅ Graceful degradation if manifest missing

### Maintainability Targets
- ✅ Manifest auto-regenerates when stale
- ✅ Clear error messages when manifest invalid
- ✅ Documentation for plugin authors
- ✅ Migration guide for existing plugins

---

## Rollback Plan

If issues arise:

1. **Phase 1 issues:**
   - Revert `brainsmith/kernels/__init__.py` to eager imports
   - Tag: `revert-lazy-kernels`

2. **Phase 2 issues:**
   - Add environment variable to disable manifest: `BRAINSMITH_NO_MANIFEST=1`
   - Fallback to full discovery automatically

3. **Phase 3 issues:**
   - FINN changes are isolated in finn repo
   - Can be reverted independently

4. **Critical failure:**
   - Feature flags in settings:
     ```python
     # brainsmith/settings.py
     use_lazy_loading: bool = True  # Can be disabled via config
     use_manifest: bool = True      # Can be disabled via config
     ```

---

## Risks & Mitigations

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Manifest gets out of sync | High | Medium | Auto-invalidation, version checks, file mtimes, fallback to full discovery |
| Breaks existing imports | High | Low | Comprehensive backward compat testing, `__getattr__` hook |
| FINN changes break FINN | Medium | Low | Isolate changes, test thoroughly, coordinate with team |
| Performance regressions | Medium | Low | Comprehensive benchmarking, monitoring, CI checks |
| Complexity for maintainers | Low | Medium | Good documentation, clear error messages, runbook |
| Manifest corruption | Medium | Low | JSON schema validation, checksum, auto-regenerate on error |

---

## Dependencies & Coordination

### Internal Teams
- Core brainsmith team: Review registry changes
- DevOps: CI/CD updates for manifest generation
- Documentation: Update user and developer guides

### External Dependencies
- **FINN team:** Coordinate on entry point optimization PR
- **QONNX team:** Similar optimizations may be beneficial
- **Brevitas team:** Check for similar import issues

### Infrastructure Changes
- Build system changes to generate manifests
- Package manifest in distribution
- Update release process documentation

---

## Documentation Updates

### User Documentation
- Add "Performance Tips" section explaining lazy loading
- Update "Plugin Development" guide
- Add CLI reference for `--refresh-manifest`
- Add troubleshooting section for manifest issues

### Developer Documentation
- Architecture doc explaining manifest system
- Migration guide for plugin authors
- Troubleshooting guide for manifest issues
- Performance optimization guide

### API Documentation
- Document lazy loading behavior
- Document manifest format
- Document registry API changes
- Add docstrings with examples

---

## Future Enhancements

Post-MVP optimizations to consider:

1. **Parallel manifest generation**
   - Generate manifests for multiple sources in parallel
   - Could save additional time during builds

2. **Compressed manifest**
   - Use msgpack or protobuf instead of JSON
   - Faster loading, smaller size

3. **Manifest per source**
   - Each plugin source has its own manifest
   - Allows independent updates

4. **Import-time profiling**
   - Add instrumentation to track slow imports
   - Help plugin authors optimize

5. **Bytecode caching**
   - Ensure `__pycache__` is properly utilized
   - Package pre-compiled bytecode

---

## Appendix

### A. Latency Profile Reference

See `/tmp/latency_profile_report.md` for detailed analysis including:
- Phase-by-phase breakdown
- Import timeline visualization
- Critical path analysis
- Function-level hotspots

### B. Related Patterns

Similar lazy loading patterns in other projects:
- **Django:** App registry with lazy loading
- **Pytest:** Plugin discovery with entry points
- **Webpack:** Module bundling with code splitting
- **Python importlib:** PEP 562 module `__getattr__`

### C. References

- [PEP 562 - Module __getattr__](https://peps.python.org/pep-0562/)
- [PEP 610 - Direct URL](https://peps.python.org/pep-0610/)
- [Python Import System](https://docs.python.org/3/reference/import.html)
- [importlib documentation](https://docs.python.org/3/library/importlib.html)
