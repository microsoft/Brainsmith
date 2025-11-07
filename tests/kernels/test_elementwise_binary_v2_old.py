"""Test elementwise arithmetic operations (Add, Sub, Mul).

All operations share identical test matrix since shape/dtype/design handling is operation-agnostic.
"""

import pytest
import numpy as np
import onnx.helper as helper
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.single_kernel_test_v2 import SingleKernelTest
from tests.frameworks.test_config import (
    KernelTestConfig,
    ModelStructure,
    DesignParameters,
    PlatformConfig,
    ValidationConfig,
)
from brainsmith.primitives.transforms.infer_kernels import InferKernels
from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

# ============================================================================
# Reusable Sub-Configs
# ============================================================================

# Platform configurations
ZYNQ_7020 = PlatformConfig(fpgapart="xc7z020clg400-1")

# Validation configurations
STANDARD_VALIDATION = ValidationConfig(
    tolerance_python={"rtol": 1e-7, "atol": 1e-9},
    tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
)
PYTHON_ONLY_VALIDATION = ValidationConfig(
    tolerance_python={"rtol": 1e-7, "atol": 1e-9},
)

# Design configurations
DESIGN_BASELINE = DesignParameters()  # No parallelization
DESIGN_PE8 = DesignParameters(input_streams={0: 8})
DESIGN_PE16 = DesignParameters(input_streams={0: 16})

# ============================================================================
# Compositional Test Patterns - Arete Refactoring
# ============================================================================

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BinaryOpPattern:
    """Reusable test pattern combining shape, design, and validation.

    Composition over repetition - define once, use across all datatypes.
    """
    name: str
    lhs_shape: Tuple[int, ...]
    rhs_shape: Tuple[int, ...]
    design: DesignParameters
    validation: ValidationConfig


# Define all test patterns once - no repetition
_PATTERNS = {
    # Equal-shape patterns (2D)
    "baseline_2d": BinaryOpPattern("baseline_2d", (1, 64), (1, 64), DESIGN_BASELINE, STANDARD_VALIDATION),
    "pe8_2d": BinaryOpPattern("pe8_2d", (4, 128), (4, 128), DESIGN_PE8, STANDARD_VALIDATION),
    "pe16_2d": BinaryOpPattern("pe16_2d", (4, 128), (4, 128), DESIGN_PE16, STANDARD_VALIDATION),

    # Equal-shape patterns (4D) - renamed from misleading "cppsim"
    "multidim_4d": BinaryOpPattern("multidim_4d", (1, 8, 8, 32), (1, 8, 8, 32), DESIGN_PE8, STANDARD_VALIDATION),

    # Broadcasting patterns - dynamic_dynamic
    "broadcast_channel": BinaryOpPattern("broadcast_channel", (1, 8, 8, 32), (32,), DESIGN_BASELINE, PYTHON_ONLY_VALIDATION),
    "broadcast_scalar": BinaryOpPattern("broadcast_scalar", (1, 8, 8, 32), (1,), DESIGN_BASELINE, PYTHON_ONLY_VALIDATION),
    "broadcast_spatial": BinaryOpPattern("broadcast_spatial", (1, 8, 8, 32), (1, 1, 1, 32), DESIGN_PE8, PYTHON_ONLY_VALIDATION),
    "broadcast_bidir": BinaryOpPattern("broadcast_bidir", (1, 8, 1, 32), (1, 1, 8, 1), DESIGN_BASELINE, PYTHON_ONLY_VALIDATION),
    "broadcast_rank": BinaryOpPattern("broadcast_rank", (4, 8, 16), (16,), DESIGN_BASELINE, PYTHON_ONLY_VALIDATION),
}


# Datatype groups - organized by testing needs and coverage level
_DTYPE_GROUPS = {
    # Original types - full coverage with backend testing
    "original": {
        "INT8": ["baseline_2d", "pe8_2d", "multidim_4d"],
        "INT16": ["baseline_2d", "pe16_2d", "multidim_4d"],
        "BIPOLAR": ["baseline_2d", "pe8_2d", "multidim_4d"],
    },

    # New integer types - Python only, full coverage
    "new_int": {
        "UINT8": ["baseline_2d", "pe8_2d", "multidim_4d"],
        "UINT16": ["baseline_2d", "pe16_2d", "multidim_4d"],
        "UINT32": ["baseline_2d", "pe8_2d"],  # Skip 4D for wide types
        "INT32": ["baseline_2d", "pe8_2d"],   # Skip 4D for wide types
    },

    # Float types - full coverage
    "float": {
        "FLOAT32": ["baseline_2d", "pe8_2d", "multidim_4d"],
    },

    # Broadcasting - test critical type representatives
    "broadcast": {
        "INT8": ["broadcast_channel", "broadcast_scalar", "broadcast_spatial", "broadcast_bidir", "broadcast_rank"],
        "FLOAT32": ["broadcast_channel", "broadcast_scalar"],  # Float arithmetic differs
        "UINT8": ["broadcast_channel"],  # Unsigned overflow behavior
    },
}


# Mixed datatype tests - NEW coverage for datatype derivation logic
_MIXED_DTYPE_TESTS = [
    # (lhs_dtype, rhs_dtype, pattern_name, description)
    ("INT8", "UINT8", "baseline_2d", "mixed_sign"),      # Signedness mixing
    ("INT8", "INT16", "baseline_2d", "mixed_width"),     # Width mixing
    ("UINT8", "UINT16", "baseline_2d", "mixed_width_unsigned"),  # Unsigned width
    ("INT8", "UINT16", "baseline_2d", "mixed_both"),     # Both sign and width
]

# Arithmetic operations (all use identical test matrix)
_ARITHMETIC_OPERATIONS = ["Add", "Sub", "Mul"]


def _generate_configs_from_patterns():
    """Generate base configs by composing datatypes × patterns.

    Uses compositional approach - patterns defined once, reused across all types.
    Each config created is operation-agnostic (operation set later).

    Returns:
        List of KernelTestConfig with operation="placeholder" (to be filled by caller)
    """
    base_configs = []

    # Equal-shape tests (original + new types + float)
    for group_name in ["original", "new_int", "float"]:
        dtypes_dict = _DTYPE_GROUPS[group_name]

        for dtype_name, pattern_names in dtypes_dict.items():
            for pattern_name in pattern_names:
                pattern = _PATTERNS[pattern_name]

                # Override validation to PYTHON_ONLY for new types
                validation = PYTHON_ONLY_VALIDATION if group_name in ["new_int", "float"] else pattern.validation

                # Determine platform: only original types with STANDARD_VALIDATION get backend testing
                is_backend_testable = (group_name == "original" and pattern.validation == STANDARD_VALIDATION)
                platform = ZYNQ_7020 if is_backend_testable else PlatformConfig()

                # Add python_only mark for Python-only tests
                marks = [pytest.mark.python_only] if not is_backend_testable else []

                test_id = f"{dtype_name.lower()}_{pattern.name}"

                base_configs.append(
                    KernelTestConfig(
                        test_id=test_id,
                        model=ModelStructure(
                            operation="placeholder",
                            input_shapes={"lhs": pattern.lhs_shape, "rhs": pattern.rhs_shape},
                            input_dtypes={"lhs": DataType[dtype_name], "rhs": DataType[dtype_name]},
                        ),
                        design=pattern.design,
                        platform=platform,
                        validation=validation,
                        marks=marks,
                    )
                )

    # Broadcasting tests - critical type representatives
    for dtype_name, pattern_names in _DTYPE_GROUPS["broadcast"].items():
        for pattern_name in pattern_names:
            pattern = _PATTERNS[pattern_name]

            test_id = f"{dtype_name.lower()}_{pattern.name}"

            base_configs.append(
                KernelTestConfig(
                    test_id=test_id,
                    model=ModelStructure(
                        operation="placeholder",
                        input_shapes={"lhs": pattern.lhs_shape, "rhs": pattern.rhs_shape},
                        input_dtypes={"lhs": DataType[dtype_name], "rhs": DataType[dtype_name]},
                    ),
                    design=pattern.design,
                    platform=PlatformConfig(),  # Broadcasting tests Python-only
                    validation=PYTHON_ONLY_VALIDATION,
                    marks=[pytest.mark.python_only],
                )
            )

    # Mixed datatype tests - NEW coverage for datatype derivation
    for lhs_dtype, rhs_dtype, pattern_name, description in _MIXED_DTYPE_TESTS:
        pattern = _PATTERNS[pattern_name]

        test_id = f"mixed_{description}"

        base_configs.append(
            KernelTestConfig(
                test_id=test_id,
                model=ModelStructure(
                    operation="placeholder",
                    input_shapes={"lhs": pattern.lhs_shape, "rhs": pattern.rhs_shape},
                    input_dtypes={"lhs": DataType[lhs_dtype], "rhs": DataType[rhs_dtype]},
                ),
                design=pattern.design,
                platform=PlatformConfig(),
                validation=PYTHON_ONLY_VALIDATION,
                marks=[pytest.mark.python_only],
            )
        )

    return base_configs


def _generate_all_operation_configs():
    """Generate configs for all operations using compositional patterns.

    Takes base configs (operation-agnostic) and creates final configs
    for each operation in _ARITHMETIC_OPERATIONS.

    Returns:
        List of KernelTestConfig objects for all operations
    """
    base_configs = _generate_configs_from_patterns()
    final_configs = []

    for operation in _ARITHMETIC_OPERATIONS:
        for config in base_configs:
            # Create new config with operation set
            final_configs.append(
                KernelTestConfig(
                    test_id=f"{operation.lower()}_{config.test_id}",
                    model=ModelStructure(
                        operation=operation,
                        input_shapes=config.input_shapes,
                        input_dtypes=config.input_dtypes,
                    ),
                    design=config.design,
                    platform=config.platform,
                    validation=config.validation,
                    marks=config.marks,
                )
            )

    return final_configs


# Generate all configs using compositional approach
_ALL_BINARY_OP_CONFIGS = _generate_all_operation_configs()


# ============================================================================
# Test class - unified for all binary operations
# ============================================================================


class TestElementwiseBinary(SingleKernelTest):
    """Test elementwise arithmetic operations (Add, Sub, Mul).

    Compositional test architecture using reusable patterns:
    - BinaryOpPattern: shape + design + validation combinations
    - Datatype groups: organized by testing needs and coverage level
    - Mixed datatypes: NEW coverage for datatype derivation logic

    Test Coverage Matrix:
    ┌─────────────────────────┬──────────┬─────────────────────────────────────────┐
    │ Category                │ Configs  │ Patterns                                │
    ├─────────────────────────┼──────────┼─────────────────────────────────────────┤
    │ Original types          │ 9        │ INT8/16, BIPOLAR × 3 patterns           │
    │ New integer types       │ 10       │ UINT8/16/32, INT32 × 2-3 patterns       │
    │ Float types             │ 3        │ FLOAT32 × 3 patterns                    │
    │ Broadcasting (INT8)     │ 5        │ channel, scalar, spatial, bidir, rank   │
    │ Broadcasting (FLOAT32)  │ 2        │ channel, scalar (NEW)                   │
    │ Broadcasting (UINT8)    │ 1        │ channel (NEW)                           │
    │ Mixed datatypes         │ 4        │ sign/width mixing (NEW)                 │
    ├─────────────────────────┼──────────┼─────────────────────────────────────────┤
    │ Total base configs      │ 34       │                                         │
    │ × 3 operations          │ × 3      │ Add, Sub, Mul                           │
    │ × 6 inherited tests     │ × 6      │ pipeline + Python + cppsim + rtlsim     │
    ├─────────────────────────┼──────────┼─────────────────────────────────────────┤
    │ **Total tests**         │ **612**  │                                         │
    └─────────────────────────┴──────────┴─────────────────────────────────────────┘

    Patterns (defined in _PATTERNS):
    - baseline_2d: No parallelization, 2D shapes
    - pe8_2d: PE=8 parallelization
    - pe16_2d: PE=16 parallelization
    - multidim_4d: 4D tensor handling (was misleadingly named "cppsim")
    - broadcast_*: 5 broadcasting patterns for dynamic_dynamic

    Arete Improvements:
    ✅ Compositional design - patterns defined once, reused across all types
    ✅ Honest naming - "multidim_4d" not "cppsim" (backend tests controlled by validation)
    ✅ PE16 consistency - used correctly for INT16 and UINT16
    ✅ Float broadcast coverage - FLOAT32 tested with channel+scalar patterns
    ✅ Mixed datatype coverage - NEW tests for datatype derivation logic
    ✅ Reduced code - ~60% less redundancy via composition

    Operations: Add, Sub, Mul (comprehensive)
    Deferred: Div, logical, comparison, bitwise operations
    """

    @pytest.fixture(
        params=[
            pytest.param(cfg, marks=cfg.marks, id=cfg.test_id)
            for cfg in _ALL_BINARY_OP_CONFIGS
        ]
    )
    def kernel_test_config(self, request):
        """Configuration fixture for all binary operations."""
        return request.param

    def make_test_model(self, kernel_test_config):
        """Create binary operation model from config.

        Supports both equal-shape and broadcasting scenarios.
        Output shape is computed via np.broadcast_shapes() to handle broadcasting.
        Operation type is extracted from config.operation (Add/Sub/Mul).
        """
        operation = kernel_test_config.operation
        input_shapes = kernel_test_config.input_shapes

        # Create input tensor infos with lhs/rhs naming
        lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, input_shapes["lhs"])
        rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, input_shapes["rhs"])

        # Compute output shape (handles broadcasting)
        output_shape = tuple(np.broadcast_shapes(input_shapes["lhs"], input_shapes["rhs"]))
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Create node and graph
        node = helper.make_node(operation, ["lhs", "rhs"], ["output"], name=f"{operation}_0")
        graph = helper.make_graph([node], f"test_{operation.lower()}", [lhs, rhs], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        return model, ["lhs", "rhs"]

    def get_kernel_inference_transform(self):
        """Convert ONNX binary op → ElementwiseBinaryOp kernel."""
        return lambda: InferKernels([ElementwiseBinaryOp])
