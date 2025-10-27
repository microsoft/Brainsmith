# FINN Integration Tests

Integration tests for FINN (Fast Inference Neural Networks) dataflow accelerator generation with the Brainsmith DSE framework.

## What These Tests Validate

✅ **FINN transformation pipeline** - Real FINN execution with analytical estimates
✅ **DSE segment execution** - Segment building, caching, and branching
✅ **Pipeline integration** - End-to-end blueprint → model → estimates
✅ **Cache behavior** - Hit/miss, invalidation, corruption detection

❌ **NOT validated**: Vivado synthesis, IP generation, bitfile creation, hardware execution

## Test Organization

### `test_pipeline_integration.py` (4 tests)
Complete DSE pipelines from model input to final outputs.

- `test_full_pipeline_estimates` - Full estimate pipeline with Brevitas model
- `test_pipeline_with_intermediate_models` - Intermediate model saving
- `test_pipeline_with_step_ranges` - Selective step execution (start_step/stop_step)
- `test_estimate_report_generation` - Estimate report structure validation

### `test_segment_execution.py` (4 tests)
Segment-level execution mechanics and dataflow.

- `test_single_segment_execution` - Basic linear segment execution
- `test_branching_execution` - Branch points and skip operators
- `test_artifact_sharing_at_branches` - Input sharing across branches
- `test_segment_failure_handling` - Graceful failure and error propagation

### `test_cache_behavior.py` (4 tests)
Segment caching with real FINN builds.

- `test_cache_hit_skips_rebuild` - Cache hits avoid expensive rebuilds
- `test_corrupted_cache_triggers_rebuild` - Corruption detection
- `test_cache_invalidation_on_config_change` - Config changes invalidate cache
- `test_invalid_onnx_in_cache` - Invalid ONNX detection

## Quick Start

```bash
# Run all FINN integration tests (~12 seconds)
poetry run pytest tests/integration/finn/ -v

# Run specific test file
poetry run pytest tests/integration/finn/test_pipeline_integration.py -v

# Run single test with verbose output
poetry run pytest tests/integration/finn/test_pipeline_integration.py::TestPipelineIntegration::test_full_pipeline_estimates -xvs

# Show test durations
poetry run pytest tests/integration/finn/ -v --durations=10
```

## FINN Pipeline Stages

Tests use **estimate-only pipelines** (no synthesis):

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

## Test Markers

```python
@pytest.mark.finn_build  # Real FINN execution (estimate-only)
@pytest.mark.timeout(900)  # 15 min max per test
```

## Fixtures

### Model Fixtures (`tests/fixtures/models.py`)

- **`simple_onnx_model`** - Basic ONNX model (fast, any test)
- **`quantized_onnx_model`** - QONNX quantized model (FINN-compatible)
- **`brevitas_fc_model`** - Brevitas FC network (hardware conversion compatible)
  - Session-scoped, cached (~15s first run, instant thereafter)
  - 2-layer FC: 784→64→10, 2-bit quantization
  - Compatible with full `convert_to_hw` pipeline

### Blueprint Fixtures (`tests/fixtures/blueprints.py`)

```python
# Use default estimate pipeline
blueprint = create_finn_blueprint(tmp_path, name="test")

# Use minimal pipeline (cache tests)
blueprint = create_finn_blueprint(
    tmp_path,
    name="minimal",
    steps=FINN_PIPELINE_MINIMAL,
    target_fps=None
)
```

**Available presets**:
- `FINN_PIPELINE_MINIMAL` - Streamline + tidy_up only
- `FINN_PIPELINE_ESTIMATES` - Full estimate pipeline (default)
- `FINN_PIPELINE_STITCHED_IP` - IP generation (future use)

## Requirements

These tests require:
- FINN installed in `deps/finn/`
- FINN environment configured (auto-configured on first import)
- Brevitas, QONNX, and FINN dependencies
- **No Vivado required** (estimates only)

## Test Execution Time

| Test Suite | Tests | Time | What It Does |
|------------|-------|------|--------------|
| Full suite | 12 | ~12s | All integration tests |
| Pipeline tests | 4 | ~2s | End-to-end pipelines |
| Segment tests | 4 | ~1s | Execution mechanics |
| Cache tests | 4 | ~9s | Cache behavior (2 builds per test) |

## Coverage

Current DSE coverage from these tests: **65%**

Key areas covered:
- Segment building and execution
- FINN adapter integration
- Cache hit/miss logic
- Pipeline configuration
- Error handling and recovery

## Synthesis Testing

For **actual Vivado synthesis and IP generation**, see:
- `examples/bert/` - BERT acceleration with full synthesis
- Future: `tests/integration/finn_synthesis/` (Phase 6+)

Synthesis tests would require:
- Vivado HLS installation
- Much longer execution times (hours)
- Hardware infrastructure for CI
- Markers: `@pytest.mark.rtl_sim`, `@pytest.mark.slow`

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

### Deprecation warnings
All FINN dependency warnings are filtered in `tests/pytest.ini`.

## Design Philosophy

These tests follow the **test pyramid**:
- **Fast feedback** - Analytical estimates, not synthesis
- **Real components** - Actual FINN transformations, not mocks
- **Comprehensive coverage** - All DSE mechanics validated
- **Leave slow tests to examples** - Synthesis validated in examples/

This allows **rapid iteration during development** while maintaining confidence that FINN integration works correctly.

---

**Related Documentation**:
- `docs/FINN_QUICK_REFERENCE.md` - FINN cppsim/rtlsim/estimates guide
- `_artifacts/finn_fixture_improvements.md` - Fixture design rationale
- `_artifacts/phase5_tier2_summary.md` - Test implementation details
