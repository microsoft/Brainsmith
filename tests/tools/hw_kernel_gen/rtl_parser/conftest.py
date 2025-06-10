############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Pytest configuration file for RTL Parser tests."""

import pytest
import logging
import tempfile
import os
import shutil
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import ProtocolValidator
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, InterfaceType, PortGroup

"""Pytest configuration file for RTL Parser tests."""

import pytest
import logging
import tempfile
import os
import shutil
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import ProtocolValidator, AXI_LITE_SUFFIXES
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, InterfaceType, PortGroup

logger = logging.getLogger(__name__)

# =============================================================================
# CORE FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def parser():
    """Provides a configured RTLParser instance for tests.
    
    This fixture has module scope to improve test performance by reusing the
    parser instance across multiple tests.
    """
    logger.info("Setting up RTLParser fixture (module scope)")
    try:
        parser_instance = RTLParser(debug=False)
        logger.info("RTLParser fixture created successfully.")
    except Exception as e:
        logger.error(f"Failed to create RTLParser fixture: {e}", exc_info=True)
        pytest.fail(f"Failed to create RTLParser fixture: {e}")
    return parser_instance

@pytest.fixture(scope="function")
def parser_debug():
    """Provides an RTLParser instance with debug enabled for detailed testing."""
    logger.info("Setting up debug RTLParser fixture")
    try:
        return RTLParser(debug=True)
    except Exception as e:
        logger.error(f"Failed to create debug RTLParser fixture: {e}", exc_info=True)
        pytest.fail(f"Failed to create debug RTLParser fixture: {e}")

@pytest.fixture(scope="function")
def scanner():
    """Provides an InterfaceScanner instance for tests."""
    logger.info("Setting up InterfaceScanner fixture")
    try:
        scanner_instance = InterfaceScanner()
        logger.info("InterfaceScanner fixture created successfully.")
    except Exception as e:
        logger.error(f"Failed to create InterfaceScanner fixture: {e}", exc_info=True)
        pytest.fail(f"Failed to create InterfaceScanner fixture: {e}")
    return scanner_instance

@pytest.fixture(scope="function")
def validator():
    """Provides a ProtocolValidator instance for tests."""
    return ProtocolValidator()

@pytest.fixture(scope="function")
def interface_builder():
    """Provides an InterfaceBuilder instance for tests."""
    return InterfaceBuilder()

@pytest.fixture(scope="function")
def interface_builder_debug():
    """Provides an InterfaceBuilder instance with debug enabled."""
    return InterfaceBuilder(debug=True)

@pytest.fixture(scope="function")
def temp_sv_file():
    """Creates a temporary directory and a helper function to write SystemVerilog files.
    
    Returns:
        A function that accepts content and optional filename, writes the content to a file,
        and returns the absolute path to the created file.
        
    Example:
        def test_something(temp_sv_file):
            path = temp_sv_file("module test; endmodule", "test_module.sv")
            # use path...
    """
    temp_dir = tempfile.mkdtemp()
    files_created = []

    def _create_file(content: str, filename: str = "test.sv") -> str:
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        files_created.append(file_path)
        return file_path

    yield _create_file

    # Cleanup: Remove the temporary directory and its contents
    for path in files_created:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

# =============================================================================
# MODULE CONTENT FIXTURES
# =============================================================================

# Constants for SystemVerilog module components
VALID_HEADER_PARAMS_PORTSOPEN = """\
    module valid_module #(
        parameter WIDTH = 32
    ) ("""

HEADER_PARAMS_PLACEHOLDER = """\
    module valid_module #(
        <PLACEHOLDER>
    ) ("""

VALID_GLOBAL_SIGNALS = """\
        // Global control signals
        input logic ap_clk,
        input logic ap_rst_n,"""

VALID_AXI_STREAM_IN_INTERFACE = """\
        // AXI-Stream input
        input logic [WIDTH-1:0] in0_TDATA,
        input logic in0_TVALID,
        output logic in0_TREADY,"""

VALID_AXI_STREAM_OUT_INTERFACE = """\
        // AXI-Stream output
        output logic [WIDTH-1:0] out0_TDATA,
        output logic out0_TVALID,
        input logic out0_TREADY"""

VALID_MIN_INTERFACES = f"""\
{VALID_GLOBAL_SIGNALS}
{VALID_AXI_STREAM_IN_INTERFACE}
{VALID_AXI_STREAM_OUT_INTERFACE}
"""

VALID_PORTS_CLOSE = """\
    );"""

VALID_MODULE_BODY_CONTENT = """\
        // Module body content"""

VALID_ENDMODULE_STATEMENT = """\
    endmodule"""

VALID_MODULE_BODY = f"""\
{VALID_PORTS_CLOSE}
{VALID_MODULE_BODY_CONTENT}
{VALID_ENDMODULE_STATEMENT}
"""

@pytest.fixture
def valid_module_content():
    """Returns SystemVerilog content for a valid module by assembling predefined parts."""
    return f"""\
{VALID_HEADER_PARAMS_PORTSOPEN}
{VALID_MIN_INTERFACES}
{VALID_MODULE_BODY}
    """

@pytest.fixture
def valid_module_placeholder_params():
    """Returns SystemVerilog content for a valid module with parameter placeholders."""
    return f"""\
{HEADER_PARAMS_PLACEHOLDER}
{VALID_MIN_INTERFACES}
{VALID_MODULE_BODY}
    """

# =============================================================================
# PORT FIXTURES - GLOBAL CONTROL
# =============================================================================

@pytest.fixture
def global_ports():
    """Returns a list of standard global control ports."""
    return [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        Port(name="ap_clk2x", direction=Direction.INPUT, width="1")  # Optional signal
    ]

@pytest.fixture  
def global_ports_minimal():
    """Returns minimal required global control ports."""
    return [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1")
    ]

# =============================================================================
# PORT FIXTURES - AXI STREAM
# =============================================================================

@pytest.fixture
def axi_stream_in_ports():
    """Returns a list of standard AXI-Stream input ports."""
    return [
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="in0_TLAST", direction=Direction.INPUT, width="1")
    ]

@pytest.fixture
def axi_stream_out_ports():
    """Returns a list of standard AXI-Stream output ports."""
    return [
        Port(name="out1_TDATA", direction=Direction.OUTPUT, width="32"),
        Port(name="out1_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out1_TREADY", direction=Direction.INPUT, width="1")
    ]

@pytest.fixture
def axis_in_ports_with_widths():
    """AXI-Stream input ports with parametric widths for metadata testing."""
    return [
        Port(name="data_in_TDATA", direction=Direction.INPUT, width="[AXIS_WIDTH-1:0]"),
        Port(name="data_in_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="data_in_TREADY", direction=Direction.OUTPUT, width="1"),
    ]

# =============================================================================
# PORT FIXTURES - AXI LITE
# =============================================================================

@pytest.fixture
def axilite_config_ports():
    """Returns a list of complete AXI-Lite config ports (read + write channels)."""
    return [
        # Write Address Channel
        Port(name="config_AWADDR", direction=Direction.INPUT, width="32"),
        Port(name="config_AWVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        # Write Data Channel  
        Port(name="config_WDATA", direction=Direction.INPUT, width="32"),
        Port(name="config_WSTRB", direction=Direction.INPUT, width="4"),
        Port(name="config_WVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_WREADY", direction=Direction.OUTPUT, width="1"),
        # Write Response Channel
        Port(name="config_BRESP", direction=Direction.OUTPUT, width="2"),
        Port(name="config_BVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BREADY", direction=Direction.INPUT, width="1"),
        # Read Address Channel
        Port(name="config_ARADDR", direction=Direction.INPUT, width="32"),
        Port(name="config_ARVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_ARREADY", direction=Direction.OUTPUT, width="1"),
        # Read Data Channel
        Port(name="config_RDATA", direction=Direction.OUTPUT, width="32"),
        Port(name="config_RRESP", direction=Direction.OUTPUT, width="2"), 
        Port(name="config_RVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_RREADY", direction=Direction.INPUT, width="1")
    ]

@pytest.fixture
def axilite_write_ports():
    """Returns AXI-Lite write-only channel ports."""
    return [
        Port(name="config_AWADDR", direction=Direction.INPUT, width="6"),
        Port(name="config_AWVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_WDATA", direction=Direction.INPUT, width="32"),
        Port(name="config_WSTRB", direction=Direction.INPUT, width="4"),
        Port(name="config_WVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_WREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BRESP", direction=Direction.OUTPUT, width="2"),
        Port(name="config_BVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BREADY", direction=Direction.INPUT, width="1")
    ]

@pytest.fixture
def axilite_read_ports():
    """Returns AXI-Lite read-only channel ports."""
    return [
        Port(name="config_ARADDR", direction=Direction.INPUT, width="6"),
        Port(name="config_ARVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_ARREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_RDATA", direction=Direction.OUTPUT, width="32"),
        Port(name="config_RRESP", direction=Direction.OUTPUT, width="2"),
        Port(name="config_RVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_RREADY", direction=Direction.INPUT, width="1")
    ]

@pytest.fixture
def axilite_write_ports_with_widths():
    """AXI-Lite write-only ports with parametric widths for metadata testing."""
    return [
        Port(name="config_AWADDR", direction=Direction.INPUT, width="[ADDR_WIDTH-1:0]"),
        Port(name="config_AWVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_WDATA", direction=Direction.INPUT, width="[DATA_WIDTH-1:0]"),
        Port(name="config_WSTRB", direction=Direction.INPUT, width="[DATA_WIDTH/8-1:0]"),
        Port(name="config_WVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_WREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BRESP", direction=Direction.OUTPUT, width="2"),
        Port(name="config_BVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BREADY", direction=Direction.INPUT, width="1")
    ]

# =============================================================================
# MIXED PORT FIXTURES
# =============================================================================

@pytest.fixture
def unassigned_ports_list():
    """Returns a list of ports that don't belong to any standard interface."""
    return [
        Port(name="custom_signal", direction=Direction.INPUT, width="1"),
        Port(name="debug_out", direction=Direction.OUTPUT, width="8")
    ]

@pytest.fixture
def ports_all_valid_mixed():
    """Returns a comprehensive list of ports for mixed interface testing."""
    return [
        # Global control
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        # AXI-Stream input
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
        # AXI-Stream output
        Port(name="out1_V_TDATA", direction=Direction.OUTPUT, width="32"),
        Port(name="out1_V_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out1_V_TREADY", direction=Direction.INPUT, width="1"),
        # AXI-Lite config (write-only)
        Port(name="config_AWADDR", direction=Direction.INPUT, width="6"),
        Port(name="config_AWVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_AWREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_WDATA", direction=Direction.INPUT, width="32"),
        Port(name="config_WSTRB", direction=Direction.INPUT, width="4"),
        Port(name="config_WVALID", direction=Direction.INPUT, width="1"),
        Port(name="config_WREADY", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BRESP", direction=Direction.OUTPUT, width="2"),
        Port(name="config_BVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="config_BREADY", direction=Direction.INPUT, width="1")
    ]

@pytest.fixture 
def ports_with_invalid_axis():
    """Returns ports where AXI-Stream interface is missing required signals."""
    return [
        # Valid global
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        # Invalid AXI-Stream (missing TREADY)
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        # Valid AXI-Stream
        Port(name="out1_TDATA", direction=Direction.OUTPUT, width="32"),
        Port(name="out1_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out1_TREADY", direction=Direction.INPUT, width="1")
    ]

@pytest.fixture
def ports_with_unassigned():
    """Returns a mix of valid interfaces and unassigned ports."""
    return [
        # Valid global
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        # Valid AXI-Stream
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1"),
        # Unassigned ports
        Port(name="custom_enable", direction=Direction.INPUT, width="1"),
        Port(name="debug_counter", direction=Direction.OUTPUT, width="16")
    ]

# =============================================================================
# HELPER FUNCTIONS 
# =============================================================================

def create_port_group(interface_type: InterfaceType, prefix: str, ports: list[Port]) -> PortGroup:
    """Helper function to create a PortGroup from a list of ports."""
    port_dict = {}
    for port in ports:
        # Extract suffix from port name (remove prefix + underscore)
        if port.name.startswith(f"{prefix}_"):
            suffix = port.name[len(prefix)+1:].upper()
            port_dict[suffix] = port
        else:
            # Handle global ports that don't follow prefix_suffix pattern
            if interface_type == InterfaceType.GLOBAL_CONTROL:
                if port.name.startswith("ap_"):
                    suffix = port.name[3:].lower()  # Remove 'ap_' and make lowercase
                    port_dict[suffix] = port
    
    return PortGroup(
        interface_type=interface_type,
        name=prefix,
        ports=port_dict
    )
