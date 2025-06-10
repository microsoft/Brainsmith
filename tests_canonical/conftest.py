"""
Shared fixtures and configuration for canonical test suite.

This module provides common test fixtures, utilities, and configuration
for testing HWKG and dataflow modeling systems based on current implementation.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from brainsmith.dataflow.core.dataflow_interface import (
    DataflowInterface, 
    DataflowInterfaceType, 
    DataflowDataType,
    DataTypeConstraint
)
from brainsmith.dataflow.core.dataflow_model import DataflowModel
from brainsmith.dataflow.core.block_chunking import TensorChunking
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata


@pytest.fixture
def basic_datatype():
    """Standard 8-bit signed integer datatype."""
    return DataflowDataType(
        base_type="INT",
        bitwidth=8,
        signed=True,
        finn_type="INT8"
    )


@pytest.fixture 
def float_datatype():
    """32-bit floating point datatype."""
    return DataflowDataType(
        base_type="FLOAT",
        bitwidth=32,
        signed=True,
        finn_type="FLOAT32"
    )


@pytest.fixture
def basic_datatype_constraint():
    """Basic datatype constraint allowing INT/UINT 1-32 bits."""
    return DataTypeConstraint(
        base_types=["INT", "UINT"],
        min_bitwidth=1,
        max_bitwidth=32,
        signed_allowed=True,
        unsigned_allowed=True
    )


@pytest.fixture
def cnn_input_interface(basic_datatype):
    """CNN input interface: 64×56×56 with 1×56×56 blocks, 1×1×8 streams."""
    return DataflowInterface(
        name="input0",
        interface_type=DataflowInterfaceType.INPUT,
        tensor_dims=[64, 56, 56],
        block_dims=[1, 56, 56],
        stream_dims=[1, 1, 8],
        dtype=basic_datatype
    )


@pytest.fixture
def cnn_weight_interface(basic_datatype):
    """CNN weight interface: 128×64×3×3 with 32×1×3×3 blocks."""
    return DataflowInterface(
        name="weights0",
        interface_type=DataflowInterfaceType.WEIGHT,
        tensor_dims=[128, 64, 3, 3],
        block_dims=[32, 1, 3, 3],
        stream_dims=[8, 1, 1, 1],
        dtype=basic_datatype
    )


@pytest.fixture
def cnn_output_interface(basic_datatype):
    """CNN output interface: 128×54×54 with 32×54×54 blocks."""
    return DataflowInterface(
        name="output0", 
        interface_type=DataflowInterfaceType.OUTPUT,
        tensor_dims=[128, 54, 54],
        block_dims=[32, 54, 54],
        stream_dims=[8, 1, 1],
        dtype=basic_datatype
    )


@pytest.fixture
def transformer_input_interface(basic_datatype):
    """Transformer input interface: 512×768 with 1×768 blocks."""
    return DataflowInterface(
        name="input_tokens",
        interface_type=DataflowInterfaceType.INPUT,
        tensor_dims=[512, 768],
        block_dims=[1, 768],
        stream_dims=[1, 64],
        dtype=basic_datatype
    )


@pytest.fixture
def transformer_weight_interface(basic_datatype):
    """Transformer weight interface: 768×768 with 768×1 blocks."""
    return DataflowInterface(
        name="attention_weights",
        interface_type=DataflowInterfaceType.WEIGHT,
        tensor_dims=[768, 768],
        block_dims=[768, 1],
        stream_dims=[64, 1],
        dtype=basic_datatype
    )


@pytest.fixture
def simple_interfaces(basic_datatype):
    """Simple set of interfaces for basic testing."""
    input_if = DataflowInterface(
        name="in0",
        interface_type=DataflowInterfaceType.INPUT,
        tensor_dims=[64],
        block_dims=[16],
        stream_dims=[4],
        dtype=basic_datatype
    )
    
    output_if = DataflowInterface(
        name="out0", 
        interface_type=DataflowInterfaceType.OUTPUT,
        tensor_dims=[64],
        block_dims=[16],
        stream_dims=[4],
        dtype=basic_datatype
    )
    
    return {"input": input_if, "output": output_if}


@pytest.fixture
def cnn_interfaces(cnn_input_interface, cnn_weight_interface, cnn_output_interface):
    """Complete CNN interface set."""
    return {
        "input": cnn_input_interface,
        "weights": cnn_weight_interface,
        "output": cnn_output_interface
    }


@pytest.fixture
def transformer_interfaces(transformer_input_interface, transformer_weight_interface):
    """Transformer interface set."""
    output_if = DataflowInterface(
        name="output_tokens",
        interface_type=DataflowInterfaceType.OUTPUT,
        tensor_dims=[512, 768],
        block_dims=[1, 768],
        stream_dims=[1, 64],
        dtype=transformer_input_interface.dtype
    )
    
    return {
        "input": transformer_input_interface,
        "weights": transformer_weight_interface,
        "output": output_if
    }


@pytest.fixture
def basic_dataflow_model(simple_interfaces):
    """Basic dataflow model with simple interfaces."""
    interfaces = list(simple_interfaces.values())
    return DataflowModel(interfaces, {})


@pytest.fixture
def cnn_dataflow_model(cnn_interfaces):
    """CNN dataflow model."""
    interfaces = list(cnn_interfaces.values())
    return DataflowModel(interfaces, {})


@pytest.fixture
def mock_rtl_interface():
    """Mock RTL interface for testing RTL conversion."""
    mock_interface = Mock()
    mock_interface.name = "test_input"
    mock_interface.type = "AXI_STREAM"
    mock_interface.metadata = {
        "tdim_override": [32, 32],
        "qdim_override": [64, 64],
        "datatype_constraints": {
            "base_types": ["INT"],
            "min_bitwidth": 8,
            "max_bitwidth": 8
        }
    }
    return mock_interface


@pytest.fixture
def mock_onnx_model():
    """Mock ONNX model for testing model integration."""
    mock_model = Mock()
    mock_model.get_tensor_shape.return_value = [1, 64, 56, 56]
    mock_model.get_tensor_datatype.return_value = "INT8"
    return mock_model


@pytest.fixture
def sample_systemverilog_code():
    """Sample SystemVerilog code for parser testing."""
    return """
    module test_kernel (
        input wire clk,
        input wire rst,
        
        // @brainsmith INTERFACE_TYPE=AXI_STREAM
        // @brainsmith TDIM=[32,32]
        input wire [127:0] s_axis_input_tdata,
        input wire s_axis_input_tvalid,
        output wire s_axis_input_tready,
        
        // @brainsmith INTERFACE_TYPE=AXI_STREAM  
        output wire [127:0] m_axis_output_tdata,
        output wire m_axis_output_tvalid,
        input wire m_axis_output_tready
    );
    
    // Implementation here
    
    endmodule
    """


@pytest.fixture
def axiom_test_data():
    """Test data for validating Interface-Wise Dataflow Modeling axioms."""
    return {
        # Axiom 1: Data Hierarchy examples
        "hierarchy_examples": [
            {
                "tensor_dims": [64, 56, 56],
                "block_dims": [1, 56, 56], 
                "stream_dims": [1, 1, 8],
                "element_bitwidth": 8
            },
            {
                "tensor_dims": [512, 768],
                "block_dims": [1, 768],
                "stream_dims": [1, 64],
                "element_bitwidth": 16
            }
        ],
        
        # Axiom 9: Layout-driven chunking examples
        "layout_examples": [
            {"layout": "[N, C, H, W]", "shape": [1, 64, 56, 56], "chunk_dim": "C"},
            {"layout": "[N, H, W, C]", "shape": [1, 56, 56, 64], "chunk_dim": "H×W"},
            {"layout": "[N, L, C]", "shape": [1, 512, 768], "chunk_dim": "L"},
        ],
        
        # Parallelism bounds for testing
        "parallelism_bounds": [
            {"iPar": 1, "wPar": 1, "expected_efficiency": 100},
            {"iPar": 4, "wPar": 2, "expected_efficiency": 100},
            {"iPar": 128, "wPar": 1, "expected_efficiency": 50}  # Wasteful
        ]
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark expectations."""
    return {
        "chunking_strategies": {
            "max_computation_time_ms": 100,
            "max_memory_usage_mb": 50
        },
        "code_generation": {
            "max_template_render_time_ms": 500,
            "max_generated_code_size_kb": 100
        },
        "rtl_parsing": {
            "max_parse_time_ms": 200,
            "max_interfaces_detected": 20
        }
    }


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def validate_axiom_compliance(interface: DataflowInterface) -> Dict[str, bool]:
        """Validate interface compliance with core axioms."""
        results = {}
        
        # Axiom 2: Core relationship validation
        try:
            num_blocks = interface.get_num_blocks()
            total_elements_tensor = np.prod(interface.tensor_dims)
            total_elements_blocks = np.prod(interface.block_dims) * np.prod(num_blocks)
            results["axiom_2_core_relationship"] = (total_elements_tensor == total_elements_blocks)
        except:
            results["axiom_2_core_relationship"] = False
            
        # Axiom 8: Tiling constraint
        try:
            # Check stream dims tile into block dims
            stream_tiles = all(
                bd % sd == 0 for bd, sd in zip(interface.block_dims, interface.stream_dims)
                if len(interface.stream_dims) <= len(interface.block_dims)
            )
            results["axiom_8_tiling_constraint"] = stream_tiles
        except:
            results["axiom_8_tiling_constraint"] = False
            
        return results
    
    @staticmethod
    def create_test_interface(tensor_dims: List[int], block_dims: List[int], 
                            stream_dims: List[int], interface_type: DataflowInterfaceType,
                            name: str = "test_interface") -> DataflowInterface:
        """Create a test interface with specified dimensions."""
        dtype = DataflowDataType("INT", 8, True, "INT8")
        return DataflowInterface(
            name=name,
            interface_type=interface_type,
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            stream_dims=stream_dims,
            dtype=dtype
        )


@pytest.fixture
def test_utils():
    """Test utilities fixture."""
    return TestUtils


# Test markers for categorizing tests
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "axiom: Tests that validate specific axioms")
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests across components")
    config.addinivalue_line("markers", "performance: Performance and regression tests")
    config.addinivalue_line("markers", "golden: Golden reference validation tests")
    config.addinivalue_line("markers", "slow: Slow tests that may take significant time")