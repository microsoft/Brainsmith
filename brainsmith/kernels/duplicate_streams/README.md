# DuplicateStreams Kernel

## Overview

DuplicateStreams duplicates a single input stream to multiple identical output streams (fanout). This is a pure routing operation with minimal hardware cost - essentially wire splitting with appropriate buffering.

**Features**:
- Variable-arity output (2-10+ outputs supported)
- Configurable parallelization (PE parameter)
- Preserves tensor shapes and datatypes exactly
- Minimal hardware cost (no computation, just routing)

**Backends**:
- HLS (C++ synthesis via Vitis HLS)
- Future: RTL backend for ultra-low latency

---

## Testing Strategy

DuplicateStreams uses **parity testing** for FINN equivalence validation, with a significantly smaller and more maintainable test suite compared to traditional manual testing.

### Test Organization

**1. test_duplicatestreams_parity.py** (285 lines, 59 tests)

Primary validation against FINN reference implementation:

- **TestDuplicateStreamsBaseParity** (26 tests)
  - Shape tests (input/output, normal/folded)
  - Datatype tests (input/output inference)
  - Width tests (stream widths, padded widths)
  - Execution test (validates output duplication)
  - Resource estimation (LUT/BRAM/DSP/URAM)
  - Efficiency metrics

- **TestDuplicateStreamsHLSParity** (33 tests)
  - All 26 base tests for HLS backend
  - 7 HLS codegen tests:
    - `test_global_includes_parity` - Include directives
    - `test_defines_parity` - Preprocessor defines
    - `test_pragmas_parity` - HLS synthesis pragmas
    - `test_docompute_parity` - Function call generation
    - `test_blackboxfunction_parity` - Function signature
    - `test_strm_decl_parity` - Stream declarations
    - `test_dataoutstrm_parity` - NPY output generation

**2. test_duplicate_streams.py** (573 lines, 23 tests)

Brainsmith-specific features and edge cases not covered by parity:

- **TestDuplicateStreamsSchema** (8 tests)
  - Dynamic schema building (1-4 outputs)
  - Input/output schema structure
  - DSE dimensions and constraints

- **TestDuplicateStreamsKernelRegistration** (1 test)
  - Plugin system registration

- **TestDuplicateStreamsKernelOp** (3 tests)
  - `get_num_output_streams()` API
  - DesignPoint access patterns

- **TestDuplicateStreamsEdgeCases** (4 tests)
  - Large fanout (10 outputs)
  - Minimal shape ([1,1,1,1])
  - Mixed datatypes (UINT4, INT8, INT16)
  - Extreme PE values (PE=1, PE=128)

- **TestDuplicateStreamsMultiBatch** (2 tests)
  - Batch>1 execution (batch=4)
  - Conv-style multi-dimensional tensors

- **TestDuplicateStreamsIterationCounts** (2 tests)
  - Dict return API validation
  - HLS iteration count correctness

- **TestDuplicateStreamsShapeCompatibility** (1 test)
  - ONNX shape-compatible node generation

- **TestDuplicateStreamsHLSEdgeCases** (2 tests)
  - 5-output HLS generation correctness
  - PE=1 HLS generation (minimal parallelization)

---

## Running Tests

```bash
# Full test suite (82 tests)
PYTHONPATH=/home/tafk/dev/brainsmith-1:$PYTHONPATH poetry run pytest \
  brainsmith/kernels/duplicate_streams/tests/ \
  -v

# Only parity tests (validates FINN equivalence)
PYTHONPATH=/home/tafk/dev/brainsmith-1:$PYTHONPATH poetry run pytest \
  brainsmith/kernels/duplicate_streams/tests/test_duplicatestreams_parity.py \
  -v

# Only Brainsmith-specific tests
PYTHONPATH=/home/tafk/dev/brainsmith-1:$PYTHONPATH poetry run pytest \
  brainsmith/kernels/duplicate_streams/tests/test_duplicate_streams.py \
  -v

# Skip slow tests (cppsim, rtlsim)
PYTHONPATH=/home/tafk/dev/brainsmith-1:$PYTHONPATH poetry run pytest \
  brainsmith/kernels/duplicate_streams/tests/ \
  -v -m "not slow"

# Only cppsim validation (requires Vitis HLS)
PYTHONPATH=/home/tafk/dev/brainsmith-1:$PYTHONPATH poetry run pytest \
  brainsmith/kernels/duplicate_streams/tests/ \
  -v -m cppsim
```

---

## Why Parity Testing?

### Before Migration (Phase 1)
- **1,509 lines** across 3 files
- **52 manual tests** requiring updates on API changes
- **No FINN validation** - tests only verified internal consistency
- **No cppsim tests** - HLS generation untested in practice
- **High maintenance burden** - duplicated logic across base/HLS tests

### After Migration (Phase 2)
- **862 lines** across 2 files (43% reduction)
- **23 manual tests** requiring updates (56% reduction)
- **59 auto-generated parity tests** validating FINN equivalence
- **cppsim validation** included automatically
- **Lower maintenance** - framework evolves, tests get new coverage for free

### Key Benefits

1. **FINN Equivalence Validation**
   - Every test compares Brainsmith vs FINN reference
   - Catches API divergence immediately (Phase 1 caught 2 bugs)
   - Ensures drop-in compatibility with FINN build pipeline

2. **Automatic Coverage**
   - 25 base tests auto-generated per backend
   - 7 HLS codegen tests auto-generated
   - Framework updates provide new coverage for free

3. **Compilation Validation**
   - `test_cppsim_execution_parity` compiles and runs HLS C++
   - Validates generated code actually works (not just syntactically correct)

4. **Standardized Pattern**
   - Same approach used by AddStreams, Thresholding, VVAU, etc.
   - Consistent testing methodology across all kernels
   - Easy to understand for new developers

---

## Known Parity Test Failures (7)

These are expected failures documented in Phase 1 results:

1. **test_make_shape_compatible_op_parity** (base & HLS)
   - **Status**: Intentional divergence
   - **Cause**: Brainsmith uses `Split`, FINN uses `RandomNormal`
   - **Impact**: Low - both are valid ONNX ops for shape inference

2. **test_execute_node_parity** (HLS)
   - **Status**: Test framework issue
   - **Cause**: exec_mode not set before execution
   - **Fix**: Update parity executor to set exec_mode

3. **test_cppsim_execution_parity**
   - **Status**: Same as #2
   - **Fix**: Same as #2

4. **test_lut_estimation_parity**
   - **Status**: Under investigation
   - **Cause**: Different LUT estimation formula
   - **Fix**: Verify if intentional or bug

5. **test_dsp_estimation_parity**
   - **Status**: Under investigation
   - **Cause**: Different DSP estimation formula
   - **Fix**: Verify if intentional or bug

6. **test_rtlsim_execution_parity**
   - **Status**: RTL backend not implemented
   - **Cause**: No RTL backend yet
   - **Fix**: N/A until RTL backend exists

---

## Implementation Details

### Variable-Arity Schema

DuplicateStreams uses a **dynamic schema** that adapts to the number of outputs in the ONNX node:

```python
@classmethod
def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> KernelSchema:
    """Build schema with N outputs based on node structure."""
    num_outputs = len(node.output)

    # Create output schemas for each output
    outputs = [
        OutputSchema(
            name=f"output{i}",
            datatype="input",  # Passthrough from input
            tensor_shape=("input", -1),  # Same shape as input
            stream_tiling=[("input", -1)]  # Same tiling as input
        )
        for i in range(num_outputs)
    ]

    return KernelSchema(
        name="DuplicateStreams",
        inputs=[InputSchema(name="input", stream_tiling=["PE"])],
        outputs=outputs,
        ...
    )
```

### FINN API Compatibility

The `get_number_output_values()` method returns a **dict** for multi-output kernels (FINN API contract):

```python
def get_number_output_values(self):
    """Returns dict mapping output names to iteration counts (FINN API).

    Example: {'out0': 512, 'out1': 512}
    """
    num_outputs = len(self.onnx_node.output)

    if num_outputs == 1:
        # Single-output: Return int (MVAU, Thresholding, etc.)
        folded_shape = self.get_folded_output_shape(ind=0)
        return math.prod(folded_shape[:-1])
    else:
        # Multi-output: Return dict (DuplicateStreams, Split)
        out_val = {}
        for i in range(num_outputs):
            folded_shape = self.get_folded_output_shape(ind=i)
            iteration_count = math.prod(folded_shape[:-1])
            out_val[f"out{i}"] = iteration_count
        return out_val
```

### HLS Generation

The HLS backend generates variable-arity code:

```cpp
// Generated for 2 outputs
void DuplicateStreamsCustom(
    hls::stream<ap_uint<64>> &in0_V,
    hls::stream<ap_uint<64>> &out0_V,
    hls::stream<ap_uint<64>> &out1_V
) {
    for(unsigned int i = 0; i < 512; i++) {
        #pragma HLS PIPELINE II=1
        ap_uint<64> e = in0_V.read();
        out0_V.write(e);
        out1_V.write(e);
    }
}
```

---

## Future Work

1. **Fix remaining parity failures** (exec_mode, resource estimation)
2. **Implement RTL backend** for ultra-low latency
3. **Add more edge case tests** (very large fanout, 100+ outputs)
4. **Performance benchmarking** (latency, throughput)

---

## References

- **FINN Implementation**: `deps/finn/src/finn/custom_op/fpgadataflow/duplicatestreams.py`
- **FINN HLS Backend**: `deps/finn/src/finn/custom_op/fpgadataflow/hls/duplicatestreams_hls.py`
- **Parity Test Framework**: `tests/parity/base_parity_test.py`
- **Phase 1 Results**: `docs/duplicatestreams_parity_phase1_results.md`
- **Phase 2 Coverage Analysis**: `docs/duplicatestreams_phase2_coverage_analysis.md`
- **Migration Plan**: `docs/duplicatestreams_parity_migration_plan.md`
