############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       GitHub Copilot <thomaskeller@microsoft.com>
############################################################################

"""Common test fixtures for RTL Parser tests."""

import os
import pytest
import tempfile
import shutil
import logging

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, InterfaceType, PortGroup
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_scanner import InterfaceScanner

logger = logging.getLogger(__name__)

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
        path = os.path.join(temp_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        files_created.append(path)
        logger.debug(f"Created temp file: {path}")
        return path

    yield _create_file

    # Cleanup: Remove the temporary directory and its contents
    for path in files_created:
        try:
            os.remove(path)
            logger.debug(f"Removed temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove temp file {path}: {e}")
    try:
        shutil.rmtree(temp_dir)
        logger.debug(f"Removed temp directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to remove temp directory {temp_dir}: {e}")

@pytest.fixture(scope="function")
def parser_debug():
    """Provides a configured RTLParser instance with debug enabled."""
    logger.info("Setting up debug RTLParser fixture (function scope)")
    try:
        parser_instance = RTLParser(debug=True)
        logger.info("Debug RTLParser fixture created successfully.")
    except Exception as e:
        logger.error(f"Failed to create debug RTLParser fixture: {e}", exc_info=True)
        pytest.fail(f"Failed to create debug RTLParser fixture: {e}")

    return parser_instance

# Constants for valid_module_content parts
# These constants represent logical blocks of a SystemVerilog module definition.
# They can be used individually or combined by the valid_module_content fixture.

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
        // Module body..."""

VALID_ENDMODULE_STATEMENT = """\
    endmodule"""

VALID_MODULE_BODY = f"""\
{VALID_PORTS_CLOSE}
{VALID_MODULE_BODY_CONTENT}
{VALID_ENDMODULE_STATEMENT}
"""

@pytest.fixture
def valid_module_content():
    """Returns SystemVerilog content for a valid module by assembling predefined parts.
    
    This approach allows tests to import and use individual VALID_* constants 
    for more flexible test case generation.
    """
    # The f-string structure below, including newlines and explicit indented blank lines,
    # is designed to precisely replicate the original multi-line string's formatting.
    return f"""\
{VALID_HEADER_PARAMS_PORTSOPEN}
{VALID_MIN_INTERFACES}
{VALID_MODULE_BODY}
    """


@pytest.fixture
def valid_module_placeholder_params():
    """Returns SystemVerilog content for a valid module by assembling predefined parts.
    
    This approach allows tests to import and use individual VALID_* constants 
    for more flexible test case generation. Uses placeholders for parameters.
    """
    # The f-string structure below, including newlines and explicit indented blank lines,
    # is designed to precisely replicate the original multi-line string's formatting.
    return f"""\
{HEADER_PARAMS_PLACEHOLDER}
{VALID_MIN_INTERFACES}
{VALID_MODULE_BODY}
    """

# Common port fixtures that can be reused across tests
@pytest.fixture
def global_ports():
    """Returns a list of standard global control ports."""
    return [
        Port(name="ap_clk", direction=Direction.INPUT, width="1"),
        Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
        Port(name="ap_clk2x", direction=Direction.INPUT, width="1")  # Optional signal
    ]

@pytest.fixture
def axi_stream_in_ports():
    """Returns a list of standard AXI-Stream input ports."""
    return [
        Port(name="in0_TDATA", direction=Direction.INPUT, width="32"),
        Port(name="in0_TVALID", direction=Direction.INPUT, width="1"),
        Port(name="in0_TREADY", direction=Direction.OUTPUT, width="1")
    ]

@pytest.fixture
def axi_stream_out_ports():
    """Returns a list of standard AXI-Stream output ports."""
    return [
        Port(name="out0_TDATA", direction=Direction.OUTPUT, width="32"),
        Port(name="out0_TVALID", direction=Direction.OUTPUT, width="1"),
        Port(name="out0_TREADY", direction=Direction.INPUT, width="1")
    ]

@pytest.fixture
def axilite_config_ports():
    """Returns a list of standard AXI-Lite config ports."""
    return [
        Port(name="config_AWADDR", direction=Direction.INPUT, width="32"),
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
def scanner():
    """Provides an InterfaceScanner instance for tests in this directory."""
    logger.info("Setting up InterfaceScanner fixture")
    try:
        scanner_instance = InterfaceScanner()
        logger.info("InterfaceScanner fixture created successfully.")
    except Exception as e:
        logger.error(f"Failed to create InterfaceScanner fixture: {e}", exc_info=True)
        pytest.fail(f"Failed to create InterfaceScanner fixture: {e}")

    return scanner_instance
