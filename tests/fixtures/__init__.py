"""Test fixtures for Brainsmith testing.

Organized by purpose:
- Model construction: model_builders.py (OnnxModelBuilder, convenience functions)
- Model annotation: model_annotation.py (DataType annotations, Quant insertion)
- Test data: test_data.py (generate_test_data for all QONNX types)
- ONNX models: models.py (simple_onnx_model, quantized_onnx_model, brevitas_fc_model)
- DSE fixtures: dse/ (blueprints, design_spaces)
- Registry: components/ (test kernels, backends, steps)
- Tests: tests/ (unit tests for fixture utilities)

Usage:
    # Single import for model utilities
    from tests.fixtures import (
        make_binary_op_model,
        annotate_model_datatypes,
        generate_test_data,
    )

    # DSE utilities
    from tests.fixtures.dse import (
        create_finn_blueprint,
        simple_design_space,
    )
"""

# Model construction (from model_builders.py)
# Model annotation (from model_annotation.py)
from .model_annotation import (
    annotate_inputs_and_outputs,
    # v2.3 Direct annotation (recommended)
    annotate_model_datatypes,
    datatype_to_actual_tensorproto,
    datatype_to_numpy_dtype,
    get_tensorproto_name,
    # Legacy Quant insertion (for Quant testing)
    insert_input_quant_nodes,
    # Type conversion utilities
    tensorproto_for_datatype,
)
from .model_builders import (
    OnnxModelBuilder,
    make_binary_op_model,
    make_broadcast_model,
    make_duplicate_streams_model,
    make_funclayernorm_model,
    make_multithreshold_model,
    make_parametric_op_model,
    make_unary_op_model,
    make_vvau_model,
)

# ONNX model fixtures (from models.py)
from .models import (
    brevitas_fc_model,
    create_simple_model,
    quantized_onnx_model,
    simple_onnx_model,
)

# Test data generation (from test_data.py)
from .test_data import (
    generate_onnx_test_data,
    generate_test_data,
)

# DSE fixtures are in dse/ subdirectory
# Import via: from tests.fixtures.dse import create_finn_blueprint, etc.

__all__ = [
    # Model construction
    "OnnxModelBuilder",
    "make_binary_op_model",
    "make_parametric_op_model",
    "make_unary_op_model",
    "make_multithreshold_model",
    "make_funclayernorm_model",
    "make_vvau_model",
    "make_broadcast_model",
    "make_duplicate_streams_model",
    # Model annotation (v2.3)
    "annotate_model_datatypes",
    "annotate_inputs_and_outputs",
    # Model annotation (legacy)
    "insert_input_quant_nodes",
    # Type conversions
    "tensorproto_for_datatype",
    "datatype_to_actual_tensorproto",
    "datatype_to_numpy_dtype",
    "get_tensorproto_name",
    # Test data
    "generate_test_data",
    "generate_onnx_test_data",
    # ONNX models
    "create_simple_model",
    "simple_onnx_model",
    "quantized_onnx_model",
    "brevitas_fc_model",
]
