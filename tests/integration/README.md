# Integration Tests

Integration tests for the Brainsmith DSE framework, organized by execution time and complexity into a **four-tier test pyramid**.

## Test Pyramid

```
              /\
             /  \  hardware/ - Bitfile + FPGA (hours-days) - Future: See examples/bert
            /----\
           / rtl/ \ RTL Sim + IP Gen (30min-hrs) - Future: See examples/bert
          /--------\
         /  finn/   \ FINN Builds + Estimates (12 tests, ~12s) ✅ IMPLEMENTED
        /------------\
       /    fast/     \ Pure Python DSE Logic (38 tests, ~1s) ✅ IMPLEMENTED
      /________________\
     /      unit/       \ Component Isolation (70 tests, ~2s)
    /____________________\
```

## Current Status: ✅ 50/50 Tests Passing (100%)

**Implemented Tiers:**
- ✅ **fast/** - Pure Python DSE validation (38 tests, ~1s)
- ✅ **finn/** - FINN integration with estimates (12 tests, ~12s)

**Future Tiers (Currently Validated by `examples/bert`):**
- **[Planned]** **rtl/** - RTL simulation and IP generation
- **[Planned]** **hardware/** - Bitfile generation and FPGA validation

---

## Tier 1: Fast Integration Tests (`fast/`)

**Pure Python validation without external dependencies.**

### What It Tests
- ✅ Blueprint YAML parsing, inheritance, and validation
- ✅ Step operations (insert, replace, remove, reordering)
- ✅ Design space utilities (slicing, indexing, combination limits)
- ✅ Tree construction (segment IDs, branching, statistics)
- ✅ Configuration handling (all DSEConfig fields)

❌ **NOT tested**: FINN execution, segment caching, actual pipeline runs

### Test Organization

| Test Suite | Tests | Time | Coverage |
|-------------|-------|------|----------|
| `test_blueprint_parsing.py` | 14 | ~0.3s | Blueprint parsing, inheritance, step operations |
| `test_design_space_validation.py` | 15 | ~0.3s | Step slicing, indexing, combination limits |
| `test_tree_construction.py` | 9 | ~0.3s | Tree building, segment IDs, statistics |
| **Total** | **38** | **~0.9s** | **DSE core logic (57% coverage)** |

### Quick Start
```bash
# Run all fast tests
poetry run pytest tests/integration/fast/ -v

# Run specific test file
poetry run pytest tests/integration/fast/test_blueprint_parsing.py -v

# Run on every save during development
poetry run pytest tests/integration/fast/ -x
```

### Marker
```python
@pytest.mark.fast  # Fast tests (< 1 min, no FINN)
```

---

## Tier 2: FINN Integration Tests (`finn/`)

**FINN transformation pipeline with analytical estimates (no synthesis).**

### What It Tests
- ✅ FINN transformation pipeline with real execution
- ✅ DSE segment execution, caching, and branching
- ✅ Pipeline integration (blueprint → model → estimates)
- ✅ Cache behavior (hit/miss, invalidation, corruption detection)

❌ **NOT tested**: Vivado synthesis, IP generation, bitfile creation, hardware execution

### Test Organization

| Test Suite | Tests | Time | Coverage |
|-------------|-------|------|----------|
| `test_pipeline_integration.py` | 4 | ~2s | End-to-end DSE pipelines |
| `test_segment_execution.py` | 4 | ~1s | Segment mechanics, branching |
| `test_cache_behavior.py` | 4 | ~9s | Cache validation (2 builds per test) |
| **Total** | **12** | **~12s** | **DSE + FINN integration (65% coverage)** |

### FINN Pipeline Used

Tests use **estimate-only pipelines** (analytical, no synthesis):

```python
FINN_PIPELINE_ESTIMATES = [
    'finn:streamline',                    # Model optimization (~5-10s)
    'finn:convert_to_hw',                 # Hardware layer conversion (~5s)
    'finn:create_dataflow_partition',     # Dataflow partitioning (~1s)
    'finn:specialize_layers',             # HLS/RTL specialization (~2s)
    'finn:target_fps_parallelization',    # Auto PE/SIMD config (~1s)
    'finn:generate_estimate_reports',     # Analytical estimates (~5s)
]
```

**Total execution time**: ~20-30 seconds per segment

### Quick Start
```bash
# Run all FINN tests
poetry run pytest tests/integration/finn/ -v

# Run specific test file
poetry run pytest tests/integration/finn/test_pipeline_integration.py -v

# Show test durations
poetry run pytest tests/integration/finn/ -v --durations=10
```

### Requirements
- FINN installed in `deps/finn/`
- Brevitas, QONNX dependencies
- **No Vivado required** (estimates only)

### Marker
```python
@pytest.mark.finn_build  # Real FINN execution (1-30 min)
@pytest.mark.timeout(900)  # 15 min max per test
```

---

## Tier 3: RTL Simulation Tests (`rtl/`)

**RTL generation and simulation (future work).**

### Planned Coverage
- RTL generation and simulation (Vivado XSIM)
- IP generation validation
- Resource utilization estimates
- Timing closure checks

### Current Status
**[Planned]** - Future implementation planned.

**Validation currently provided by**: `examples/bert/` (BERT acceleration with full RTL pipeline)

### Planned Markers
```python
@pytest.mark.rtl_sim  # RTL simulation tests (30min - hours)
@pytest.mark.slow     # Slow tests (hours)
@pytest.mark.timeout(14400)  # 4 hour max
```

### Requirements (Future)
- Vivado HLS installation
- Multi-hour execution times
- CI infrastructure for synthesis

---

## Tier 4: Hardware Validation Tests (`hardware/`)

**Bitfile generation and FPGA validation (future work).**

### Planned Coverage
- Bitfile generation (full Vivado synthesis)
- FPGA programming validation
- Hardware-in-the-loop testing
- Real hardware inference validation

### Current Status
**[Planned]** - Future implementation planned.

**Validation currently provided by**: `examples/bert/` (BERT acceleration with full hardware pipeline)

### Planned Markers
```python
@pytest.mark.bitfile   # Bitfile generation (hours - days)
@pytest.mark.hardware  # Requires actual FPGA hardware
@pytest.mark.timeout(86400)  # 24 hour max
```

### Requirements (Future)
- Vivado full installation
- Multi-day execution times
- Actual FPGA hardware (Pynq-Z1, ZCU102, etc.)
- Hardware test infrastructure

---

## Running Tests

### By Tier
```bash
# Fast tests only (< 1s)
poetry run pytest tests/integration/fast/ -v

# FINN tests only (~12s)
poetry run pytest tests/integration/finn/ -v

# Both fast + FINN (~13s)
poetry run pytest tests/integration/fast/ tests/integration/finn/ -v
```

### By Marker
```bash
# All fast tests
poetry run pytest -m fast -v

# All FINN builds
poetry run pytest -m finn_build -v

# Future: RTL simulation tests
poetry run pytest -m rtl_sim -v

# Future: Hardware tests
poetry run pytest -m hardware -v
```

### Development Workflow
```bash
# Fast feedback loop (< 4s total)
poetry run pytest tests/unit/ tests/integration/fast/ -v

# Full validation before commit (~16s total)
poetry run pytest tests/ -v

# With coverage
poetry run pytest tests/integration/ --cov=brainsmith/dse --cov-report=term-missing
```

---

## Test Execution Time

| Tier | Tests | Time | Frequency | Status |
|------|-------|------|-----------|--------|
| fast/ | 38 | ~1s | Every save | ✅ Implemented |
| finn/ | 12 | ~12s | Pre-commit | ✅ Implemented |
| rtl/ | - | 30min-hrs | Nightly | **[Planned]** See examples/bert |
| hardware/ | - | hrs-days | Weekly | **[Planned]** See examples/bert |

---

## Test Coverage

**Current DSE module coverage**: 79% (from fast + FINN tests)

Key modules:
- `brainsmith/dse/_parser/` - 72-92% (blueprint parsing, step operations)
- `brainsmith/dse/_builder.py` - 94% (tree construction)
- `brainsmith/dse/design_space.py` - 86% (step slicing, utilities)
- `brainsmith/dse/segment.py` - 89% (segment structure)
- `brainsmith/dse/runner.py` - 75% (segment execution)
- `brainsmith/dse/tree.py` - 64% (tree statistics, traversal)

---

## Available Markers

```python
# Implemented
@pytest.mark.fast        # Fast tests (< 1 min, no FINN)
@pytest.mark.finn_build  # Real FINN execution (1-30 min)

# Future (placeholders)
@pytest.mark.rtl_sim     # RTL simulation tests (30min - hours)
@pytest.mark.slow        # Slow tests (hours)
@pytest.mark.bitfile     # Bitfile generation (hours - days)
@pytest.mark.hardware    # Requires actual FPGA hardware

# Utilities
@pytest.mark.timeout(N)  # Test timeout in seconds
```

---

## Design Philosophy

### Test Pyramid Principles

1. **Fast tests at the base** - Run on every save (< 1s)
2. **FINN integration in the middle** - Run before commit (~12s)
3. **RTL/Hardware at the top** - Run periodically, validated by examples

### Why This Structure?

**Fast Feedback Loop:**
- ✅ Catch 80% of bugs immediately with fast tests
- ✅ Validate FINN integration before breaking CI
- ✅ Leave multi-hour synthesis to example projects

**Cost vs. Value:**
- Fast tests: < 1s, unlimited runs
- FINN tests: ~12s, run on every commit
- RTL tests: hours, run nightly (future)
- Hardware tests: days, run weekly (future)

**Real Validation:**
- Tests use **real FINN** (not mocks)
- Tests use **analytical estimates** (fast)
- Examples use **full synthesis** (comprehensive)

---

## Fixtures

### Model Fixtures (`tests/fixtures/models.py`)

- **`simple_onnx_model`** - Basic ONNX model (fast, any test)
- **`quantized_onnx_model`** - QONNX quantized model (FINN-compatible)
- **`brevitas_fc_model`** - Brevitas FC network (session-scoped, ~15s first run)

### Blueprint Fixtures (`tests/fixtures/blueprints.py`)

**Pipeline Presets:**
```python
# Minimal pipeline (cleanup only)
FINN_PIPELINE_MINIMAL = ['finn:streamline', 'finn:tidy_up']

# Full estimate pipeline (default)
FINN_PIPELINE_ESTIMATES = [
    'finn:streamline',
    'finn:convert_to_hw',
    'finn:create_dataflow_partition',
    'finn:specialize_layers',
    'finn:target_fps_parallelization',
    'finn:generate_estimate_reports'
]

# IP generation pipeline (future use)
FINN_PIPELINE_STITCHED_IP = [
    # ... includes hw_codegen, hw_ipgen, create_stitched_ip
]
```

**Usage:**
```python
# Use default estimate pipeline
blueprint = create_finn_blueprint(tmp_path, name="test")

# Use minimal pipeline
blueprint = create_finn_blueprint(
    tmp_path, name="minimal",
    steps=FINN_PIPELINE_MINIMAL,
    target_fps=None
)
```

---

## Troubleshooting

### Tests hang
```python
# Already fixed in brainsmith/_internal/finn/adapter.py:121
finn_config["enable_build_pdb_debug"] = False
```

### Import errors
```bash
# Verify FINN environment
poetry run python -c "from brainsmith import get_config; get_config()"
```

### Need verbose output
```bash
# Very verbose with local variables on failure
poetry run pytest tests/integration/ -vv -l

# Show print statements
poetry run pytest tests/integration/ -v -s

# Stop on first failure
poetry run pytest tests/integration/ -x
```

---

## What's Next?

### Immediate (Phase 5 Complete) ✅
- [x] Fast integration tests (38 tests)
- [x] FINN integration tests (12 tests)
- [x] Test fixtures with validated presets
- [x] Comprehensive documentation

### Future (Phase 6+)
- [ ] RTL simulation tests (`tests/integration/rtl/`)
- [ ] Hardware validation tests (`tests/integration/hardware/`)
- [ ] CI/CD integration with tiered execution
- [ ] Nightly synthesis validation

**For now**: RTL and hardware validation provided by **`examples/bert/`** - a complete BERT acceleration pipeline with full synthesis and hardware deployment.

---

**Related Documentation:**
- `tests/fixtures/blueprints.py` - Blueprint fixture helpers and pipeline presets
- `docs/FINN_QUICK_REFERENCE.md` - FINN pipeline stages and requirements
- `examples/bert/README.md` - Full synthesis and hardware validation example
