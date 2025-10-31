# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test infrastructure for MVAU kernel migration.

This test suite validates the Brainsmith MVAU kernel implementation against
golden reference outputs from FINN's MVAU.

Test Matrix (8 configurations from polymorphic_analysis.md):
1. (noActivation=1, mem_mode="internal_embedded", dynamic_input=0, binaryXnorMode=0)  # Phase 1
2. (noActivation=0, mem_mode="internal_embedded", dynamic_input=0, binaryXnorMode=0)  # Phase 2
3. (noActivation=1, mem_mode="internal_decoupled", dynamic_input=0, binaryXnorMode=0) # Phase 3
4. (noActivation=0, mem_mode="internal_decoupled", dynamic_input=0, binaryXnorMode=0) # Phase 3
5. (noActivation=1, mem_mode="external", dynamic_input=0, binaryXnorMode=0)           # Phase 3
6. (noActivation=0, mem_mode="external", dynamic_input=0, binaryXnorMode=0)           # Phase 3
7. (noActivation=0, mem_mode="internal_embedded", dynamic_input=0, binaryXnorMode=1)  # Phase 4
8. (noActivation=0, mem_mode="internal_decoupled", dynamic_input=1, binaryXnorMode=0) # Phase 5
"""

import pytest
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)
import qonnx.custom_op.general.xnorpopcount as xp
from qonnx.custom_op.general.multithreshold import multithreshold

import finn.core.onnx_exec as oxe

# Import MVAU kernel to register it
from brainsmith.kernels.mvau.mvau import MVAU  # noqa: F401


# ============================================================================
# Fixtures and Helpers
# ============================================================================

def make_mvau_model(
    W: np.ndarray,
    pe: int,
    simd: int,
    wdt: DataType,
    idt: DataType,
    odt: DataType,
    T: np.ndarray = None,
    tdt: DataType = None,
    mem_mode: str = "internal_embedded",
    binary_xnor_mode: int = 0,
    no_activation: int = 1,
):
    """Create MVAU model for testing.

    Args:
        W: Weight matrix (MW, MH)
        pe: Processing elements (output parallelism)
        simd: Input parallelism
        wdt: Weight datatype
        idt: Input datatype
        odt: Output datatype
        T: Threshold matrix (MH, n_thres_steps) or None
        tdt: Threshold datatype or None
        mem_mode: Memory mode ("internal_embedded", "internal_decoupled", "external")
        binary_xnor_mode: Use XNOR-popcount (0 or 1)
        no_activation: No activation (1) or with activation (0)

    Returns:
        ModelWrapper with MVAU node
    """
    mw, mh = W.shape
    assert mh % pe == 0, f"MH ({mh}) must be divisible by PE ({pe})"
    assert mw % simd == 0, f"MW ({mw}) must be divisible by SIMD ({simd})"

    # Adjust datatypes for binary XNOR mode
    export_wdt = DataType["BINARY"] if binary_xnor_mode and wdt == DataType["BIPOLAR"] else wdt
    export_idt = DataType["BINARY"] if binary_xnor_mode and idt == DataType["BIPOLAR"] else idt

    # Create ONNX graph
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])

    node_inp_list = ["inp", "weights"]
    if T is not None:
        node_inp_list.append("thresh")

    # Determine ActVal based on output datatype
    if T is not None:
        # ActVal = number of activation bits (bitwidth for quantized output)
        actval = 0 if odt == DataType["BIPOLAR"] else odt.bitwidth()
    else:
        actval = 0

    mvau_node = helper.make_node(
        "MVAU",
        node_inp_list,
        ["outp"],
        domain="brainsmith.kernels",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=export_idt.name,
        weightDataType=export_wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_activation,
        mem_mode=mem_mode,
        dynamic_input=0,
        numInputVectors=[1],
    )

    graph = helper.make_graph(
        nodes=[mvau_node],
        name="mvau_test_graph",
        inputs=[inp],
        outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="mvau-test-model")
    model = ModelWrapper(model)

    # Set datatypes
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)

    # Set initializers
    if binary_xnor_mode:
        # Convert bipolar {-1, +1} to binary {0, 1}
        model.set_initializer("weights", (W + 1) / 2)
    else:
        model.set_initializer("weights", W)

    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)

    return model


def compute_golden_reference(
    x: np.ndarray,
    W: np.ndarray,
    idt: DataType,
    wdt: DataType,
    T: np.ndarray = None,
    odt: DataType = None,
    binary_xnor_mode: bool = False
) -> np.ndarray:
    """Compute golden reference output for MVAU.

    Args:
        x: Input activations (1, MW)
        W: Weight matrix (MW, MH)
        idt: Input datatype
        wdt: Weight datatype
        T: Threshold matrix (MH, n_thres_steps) or None
        odt: Output datatype (required if T is not None)
        binary_xnor_mode: Use XNOR-popcount

    Returns:
        Expected output tensor
    """
    # Matrix multiplication
    if binary_xnor_mode or (wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]):
        # XNOR-popcount for bipolar
        y = xp.xnorpopcountmatmul((x + 1) / 2, (W + 1) / 2)
    else:
        # Regular integer matmul
        y = np.matmul(x, W)

    # Apply thresholding if provided
    if T is not None:
        assert odt is not None, "Output datatype required for thresholding"
        if odt == DataType["BIPOLAR"]:
            # Binary to bipolar: scale=2, bias=-1
            y = multithreshold(y, T, 2, -1)
        else:
            # Signed offset: scale=1, bias=odt.min()
            y = multithreshold(y, T, 1, odt.min())

    return y


# ============================================================================
# Phase 1 Tests: MV Mode (noActivation=1, internal_embedded)
# ============================================================================

@pytest.mark.parametrize("idt", [DataType["INT4"], DataType["INT8"]])
@pytest.mark.parametrize("wdt", [DataType["INT4"], DataType["INT8"]])
@pytest.mark.parametrize("mw", [16, 32])
@pytest.mark.parametrize("mh", [16, 32])
@pytest.mark.parametrize("pe", [1, 2, 4])
@pytest.mark.parametrize("simd", [1, 2, 4])
def test_mvau_phase1_mv_mode_hwop(idt, wdt, mw, mh, pe, simd):
    """Phase 1: Test MV mode (no activation) with internal_embedded.

    This tests the simplest MVAU configuration:
    - noActivation=1 (matmul only, 2 inputs)
    - mem_mode="internal_embedded" (weights in C++ header)
    - binaryXnorMode=0 (regular matmul)
    - dynamic_input=0 (static weights)
    """
    # Skip invalid folding configurations
    if mh % pe != 0 or mw % simd != 0:
        pytest.skip(f"Invalid folding: MH={mh} % PE={pe} or MW={mw} % SIMD={simd}")

    # Generate random weights and inputs
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    x = gen_finn_dt_tensor(idt, (1, mw))

    # Output datatype is accumulator type (no activation)
    odt = DataType["INT32"]

    # Create MVAU model
    model = make_mvau_model(
        W=W,
        pe=pe,
        simd=simd,
        wdt=wdt,
        idt=idt,
        odt=odt,
        T=None,  # No thresholds in Phase 1
        mem_mode="internal_embedded",
        binary_xnor_mode=0,
        no_activation=1,
    )

    # Compute golden reference
    y_expected = compute_golden_reference(x, W, idt, wdt, T=None)
    oshape = model.get_tensor_shape("outp")
    y_expected = y_expected.reshape(oshape)

    # Execute MVAU kernel (HWCustomOp execute_node)
    input_dict = {"inp": x}
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    y_produced = y_produced.reshape(y_expected.shape)

    # Validate
    assert (y_produced == y_expected).all(), \
        f"Phase 1 HW-op execution failed: expected {y_expected}, got {y_produced}"


# ============================================================================
# Phase 2 Tests: MVTU Mode (noActivation=0, internal_embedded)
# ============================================================================

@pytest.mark.parametrize("idt", [DataType["INT4"], DataType["INT8"]])
@pytest.mark.parametrize("wdt", [DataType["INT4"], DataType["INT8"]])
@pytest.mark.parametrize("odt", [DataType["INT2"], DataType["INT4"]])
@pytest.mark.parametrize("mw", [16])
@pytest.mark.parametrize("mh", [16])
@pytest.mark.parametrize("pe", [2])
@pytest.mark.parametrize("simd", [2])
def test_mvau_phase2_mvtu_mode_hwop(idt, wdt, odt, mw, mh, pe, simd):
    """Phase 2: Test MVTU mode (with thresholds) with internal_embedded.

    This tests:
    - noActivation=0 (matmul + threshold, 3 inputs)
    - mem_mode="internal_embedded"
    - binaryXnorMode=0
    - dynamic_input=0
    """
    # Generate weights, inputs, and thresholds
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    x = gen_finn_dt_tensor(idt, (1, mw))

    # Generate thresholds
    (acc_min, acc_max) = calculate_signed_dot_prod_range(idt, wdt, mw)
    n_steps = odt.get_num_possible_values() - 1
    T = np.random.randint(acc_min, acc_max - 1, (mh, n_steps)).astype(np.float32)
    T = np.sort(T, axis=1)  # Ensure non-decreasing
    tdt = DataType["INT32"]

    # Create MVAU model
    model = make_mvau_model(
        W=W,
        pe=pe,
        simd=simd,
        wdt=wdt,
        idt=idt,
        odt=odt,
        T=T,
        tdt=tdt,
        mem_mode="internal_embedded",
        binary_xnor_mode=0,
        no_activation=0,
    )

    # Compute golden reference
    y_expected = compute_golden_reference(x, W, idt, wdt, T=T, odt=odt)
    oshape = model.get_tensor_shape("outp")
    y_expected = y_expected.reshape(oshape)

    # Execute MVAU kernel
    input_dict = {"inp": x}
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    y_produced = y_produced.reshape(y_expected.shape)

    # Validate
    assert (y_produced == y_expected).all(), \
        f"Phase 2 MVTU execution failed"


# ============================================================================
# Phase 3+ Tests (Deferred)
# ============================================================================

@pytest.mark.skip(reason="Phase 3 not yet implemented")
def test_mvau_phase3_internal_decoupled():
    """Phase 3: Test internal_decoupled mem_mode."""
    pass


@pytest.mark.skip(reason="Phase 3 not yet implemented")
def test_mvau_phase3_external():
    """Phase 3: Test external mem_mode."""
    pass


@pytest.mark.skip(reason="Phase 4 not yet implemented")
def test_mvau_phase4_binary_xnor():
    """Phase 4: Test binaryXnorMode=1."""
    pass


@pytest.mark.skip(reason="Phase 5 not yet implemented")
def test_mvau_phase5_dynamic_input():
    """Phase 5: Test dynamic_input=1."""
    pass


# ============================================================================
# Test Matrix Summary
# ============================================================================

def test_configuration_matrix():
    """Document the 8 valid MVAU configurations to be tested.

    Configuration Matrix (from polymorphic_analysis.md):
    | Config | noActivation | mem_mode           | dynamic_input | binaryXnorMode | Phase  |
    |--------|--------------|-------------------|---------------|----------------|--------|
    | 1      | 1 (MV)       | internal_embedded | 0             | 0              | Phase 1|
    | 2      | 0 (MVTU)     | internal_embedded | 0             | 0              | Phase 2|
    | 3      | 1 (MV)       | internal_decoupled| 0             | 0              | Phase 3|
    | 4      | 0 (MVTU)     | internal_decoupled| 0             | 0              | Phase 3|
    | 5      | 1 (MV)       | external          | 0             | 0              | Phase 3|
    | 6      | 0 (MVTU)     | external          | 0             | 0              | Phase 3|
    | 7      | 0 (MVTU)     | internal_embedded | 0             | 1              | Phase 4|
    | 8      | 0 (MVTU)     | internal_decoupled| 1             | 0              | Phase 5|
    """
    # This test just documents the matrix, always passes
    pass
