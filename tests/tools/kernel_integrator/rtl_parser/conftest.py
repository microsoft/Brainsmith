############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Pytest configuration and shared fixtures for RTL Parser tests.

This module provides common fixtures, utilities, and configuration for testing
the RTL Parser pipeline. All fixtures follow PD-3 (Concrete Tests) by using
real implementations without mocks.
"""

import pytest
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil

# Import RTL parser components
from brainsmith.tools.kernel_integrator.rtl_parser import RTLParser, parse_rtl_file
from brainsmith.tools.kernel_integrator.rtl_parser.ast_parser import ASTParser
from brainsmith.tools.kernel_integrator.rtl_parser.module_extractor import ModuleExtractor
from brainsmith.tools.kernel_integrator.rtl_parser.protocol_validator import ProtocolValidator
from brainsmith.tools.kernel_integrator.rtl_parser.interface_builder import InterfaceBuilder
from brainsmith.tools.kernel_integrator.rtl_parser.pragma import PragmaHandler
from brainsmith.tools.kernel_integrator.rtl_parser.parameter_linker import ParameterLinker

# Import data structures
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata, InterfaceMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Parameter, Port, PortDirection
from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def rtl_parser() -> RTLParser:
    """Create a fresh RTL parser instance with strict=False for testing."""
    return RTLParser(strict=False)


@pytest.fixture
def strict_rtl_parser() -> RTLParser:
    """Create a fresh RTL parser instance with strict=True for validation testing."""
    return RTLParser(strict=True)


@pytest.fixture
def ast_parser() -> ASTParser:
    """Create an AST parser instance."""
    return ASTParser()


@pytest.fixture
def module_extractor(ast_parser) -> ModuleExtractor:
    """Create a module extractor instance."""
    return ModuleExtractor(ast_parser)


@pytest.fixture
def protocol_validator() -> ProtocolValidator:
    """Create a protocol validator instance."""
    return ProtocolValidator()


@pytest.fixture
def interface_builder() -> InterfaceBuilder:
    """Create an interface builder instance."""
    return InterfaceBuilder()


@pytest.fixture
def pragma_handler() -> PragmaHandler:
    """Create a pragma handler instance."""
    return PragmaHandler()


@pytest.fixture
def parameter_linker() -> ParameterLinker:
    """Create a parameter linker instance."""
    return ParameterLinker()


@pytest.fixture
def minimal_rtl() -> str:
    """Minimal valid SystemVerilog module."""
    return """
module minimal (
    input wire clk,
    input wire rst,
    output wire done
);
    // Minimal implementation
    assign done = 1'b0;
endmodule
"""


@pytest.fixture
def parametric_rtl() -> str:
    """Module with parameters."""
    return """
module parametric #(
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 16,
    parameter SIGNED = 0
) (
    input wire clk,
    input wire [WIDTH-1:0] data_in,
    output wire [WIDTH-1:0] data_out
);
    // Implementation
endmodule
"""


@pytest.fixture
def axi_stream_rtl() -> str:
    """Module with AXI-Stream interfaces."""
    return """
module axi_stream #(
    parameter integer DATA_WIDTH = 32
) (
    input wire clk,
    input wire rst,
    
    // AXI-Stream input
    input wire [DATA_WIDTH-1:0] s_axis_tdata,
    input wire s_axis_tvalid,
    output wire s_axis_tready,
    input wire s_axis_tlast,
    
    // AXI-Stream output
    output wire [DATA_WIDTH-1:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input wire m_axis_tready,
    output wire m_axis_tlast
);
    // Implementation
endmodule
"""


@pytest.fixture
def pragma_rtl() -> str:
    """Module with various pragmas."""
    return """
// @brainsmith TOP_MODULE pragma_example
module pragma_example #(
    parameter integer INPUT_WIDTH = 16,
    parameter integer OUTPUT_WIDTH = 32,
    parameter integer WEIGHT_WIDTH = 8
) (
    input wire clk,
    input wire rst,
    
    // @brainsmith DATATYPE s_axis_input UINT 8 32
    input wire [INPUT_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    // @brainsmith WEIGHT s_axis_weights
    input wire [WEIGHT_WIDTH-1:0] s_axis_weights_tdata,
    input wire s_axis_weights_tvalid,
    output wire s_axis_weights_tready,
    
    output wire [OUTPUT_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);
    // @brainsmith ALIAS INPUT_WIDTH input_width
    // @brainsmith DERIVED_PARAMETER total_width INPUT_WIDTH + OUTPUT_WIDTH
    // Implementation
endmodule
"""


@pytest.fixture
def sample_kernel_metadata() -> KernelMetadata:
    """Create a sample KernelMetadata for testing."""
    # Create interfaces
    input_interface = InterfaceMetadata(
        name="s_axis_input",
        interface_type=InterfaceType.INPUT,
        compiler_name="input0"
    )
    
    output_interface = InterfaceMetadata(
        name="m_axis_output",
        interface_type=InterfaceType.OUTPUT,
        compiler_name="output0"
    )
    
    # Create kernel metadata
    return KernelMetadata(
        module_name="test_kernel",
        exposed_parameters=["WIDTH", "DEPTH"],
        interfaces=[input_interface, output_interface]
    )


@pytest.fixture
def sample_parameters() -> list[Parameter]:
    """Create sample RTL parameters."""
    return [
        Parameter(name="WIDTH", value="32", param_type="integer"),
        Parameter(name="DEPTH", value="16", param_type="integer"),
        Parameter(name="SIGNED", value="0", param_type=None)
    ]


@pytest.fixture
def sample_ports() -> list[Port]:
    """Create sample RTL ports."""
    return [
        Port(name="clk", direction="input", width="1"),
        Port(name="rst", direction="input", width="1"),
        Port(name="s_axis_tdata", direction="input", width="DATA_WIDTH"),
        Port(name="s_axis_tvalid", direction="input", width="1"),
        Port(name="s_axis_tready", direction="output", width="1"),
        Port(name="m_axis_tdata", direction="output", width="DATA_WIDTH"),
        Port(name="m_axis_tvalid", direction="output", width="1"),
        Port(name="m_axis_tready", direction="input", width="1")
    ]


def create_rtl_file(content: str, filename: str, temp_dir: Path) -> Path:
    """Helper to create RTL file in temp directory."""
    file_path = temp_dir / filename
    file_path.write_text(content)
    return file_path


def assert_no_warnings(result: Any) -> None:
    """Assert that no warnings were generated during parsing."""
    if hasattr(result, 'warnings'):
        assert len(result.warnings) == 0, f"Unexpected warnings: {result.warnings}"


def assert_has_warnings(result: Any, expected_count: int = None) -> None:
    """Assert that warnings were generated during parsing."""
    assert hasattr(result, 'warnings'), "Result should have warnings attribute"
    if expected_count is not None:
        assert len(result.warnings) == expected_count, \
            f"Expected {expected_count} warnings, got {len(result.warnings)}: {result.warnings}"
    else:
        assert len(result.warnings) > 0, "Expected warnings but none were generated"


# Performance testing helpers
@pytest.fixture
def performance_timer():
    """Context manager for timing operations."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, *args):
            self.end_time = time.time()
            self.elapsed = self.end_time - self.start_time
            
        def assert_under(self, seconds: float):
            assert self.elapsed < seconds, \
                f"Operation took {self.elapsed:.3f}s, expected under {seconds}s"
    
    return Timer


# Test data validation helpers
def validate_interface_metadata(interface: InterfaceMetadata) -> None:
    """Validate that InterfaceMetadata is well-formed."""
    assert interface.name, "Interface must have a name"
    assert interface.interface_type is not None, "Interface must have a type"
    assert interface.compiler_name, "Interface must have a compiler name"
    
    # Validate compiler name format
    if interface.interface_type == InterfaceType.INPUT:
        assert interface.compiler_name.startswith("input"), \
            f"Input interface compiler name should start with 'input', got {interface.compiler_name}"
    elif interface.interface_type == InterfaceType.OUTPUT:
        assert interface.compiler_name.startswith("output"), \
            f"Output interface compiler name should start with 'output', got {interface.compiler_name}"
    elif interface.interface_type == InterfaceType.WEIGHT:
        assert interface.compiler_name.startswith("weight"), \
            f"Weight interface compiler name should start with 'weight', got {interface.compiler_name}"


def validate_kernel_metadata(km: KernelMetadata) -> None:
    """Validate that KernelMetadata is well-formed."""
    assert km.module_name, "Kernel must have a module name"
    assert isinstance(km.interfaces, list), "Interfaces must be a list"
    assert isinstance(km.exposed_parameters, list), "Exposed parameters must be a list"
    
    # Validate each interface
    for interface in km.interfaces:
        validate_interface_metadata(interface)
    
    # Check for unique interface names
    interface_names = [i.name for i in km.interfaces]
    assert len(interface_names) == len(set(interface_names)), \
        f"Duplicate interface names found: {interface_names}"
    
    # Check for unique compiler names
    compiler_names = [i.compiler_name for i in km.interfaces]
    assert len(compiler_names) == len(set(compiler_names)), \
        f"Duplicate compiler names found: {compiler_names}"