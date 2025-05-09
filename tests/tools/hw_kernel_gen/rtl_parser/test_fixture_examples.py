############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       GitHub Copilot <thomaskeller@microsoft.com>
############################################################################

"""Examples of how to use fixtures defined in test_fixtures.py."""

import pytest
import os
import logging

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction

logger = logging.getLogger(__name__)

# Example for temp_sv_file fixture
def test_example_temp_sv_file(temp_sv_file):
    """Demonstrates the use of the temp_sv_file fixture."""
    logger.info("Testing temp_sv_file fixture")
    content = "module example_module; endmodule"
    filename = "example.sv"
    
    file_path = temp_sv_file(content, filename)
    
    assert os.path.exists(file_path)
    assert os.path.basename(file_path) == filename
    with open(file_path, "r") as f:
        assert f.read() == content
    logger.info(f"Successfully tested temp_sv_file: {file_path}")

# Example for parser_debug fixture
def test_example_parser_debug(parser_debug):
    """Demonstrates the use of the parser_debug fixture."""
    logger.info("Testing parser_debug fixture")
    assert isinstance(parser_debug, RTLParser)
    assert parser_debug.debug is True
    logger.info(f"Successfully tested parser_debug: {type(parser_debug)}")

# Example for global_ports fixture
def test_example_global_ports(global_ports):
    """Demonstrates the use of the global_ports fixture."""
    logger.info("Testing global_ports fixture")
    assert isinstance(global_ports, list)
    assert len(global_ports) > 0
    for port in global_ports:
        assert isinstance(port, Port)
        logger.debug(f"Global port: {port.name}, Direction: {port.direction}, Width: {port.width}")
    
    # Example specific checks
    ap_clk_present = any(p.name == "ap_clk" and p.direction == Direction.INPUT for p in global_ports)
    ap_rst_n_present = any(p.name == "ap_rst_n" and p.direction == Direction.INPUT for p in global_ports)
    assert ap_clk_present
    assert ap_rst_n_present
    logger.info(f"Successfully tested global_ports: found {len(global_ports)} ports.")

# Example for axi_stream_in_ports fixture
def test_example_axi_stream_in_ports(axi_stream_in_ports):
    """Demonstrates the use of the axi_stream_in_ports fixture."""
    logger.info("Testing axi_stream_in_ports fixture")
    assert isinstance(axi_stream_in_ports, list)
    assert len(axi_stream_in_ports) > 0
    for port in axi_stream_in_ports:
        assert isinstance(port, Port)
        logger.debug(f"AXI Stream In port: {port.name}, Direction: {port.direction}, Width: {port.width}")
    
    # Example specific checks (e.g., TDATA, TVALID, TREADY)
    tdata_present = any("TDATA" in p.name and p.direction == Direction.INPUT for p in axi_stream_in_ports)
    tvalid_present = any("TVALID" in p.name and p.direction == Direction.INPUT for p in axi_stream_in_ports)
    tready_present = any("TREADY" in p.name and p.direction == Direction.OUTPUT for p in axi_stream_in_ports)
    assert tdata_present
    assert tvalid_present
    assert tready_present
    logger.info(f"Successfully tested axi_stream_in_ports: found {len(axi_stream_in_ports)} ports.")

# Example for axi_stream_out_ports fixture
def test_example_axi_stream_out_ports(axi_stream_out_ports):
    """Demonstrates the use of the axi_stream_out_ports fixture."""
    logger.info("Testing axi_stream_out_ports fixture")
    assert isinstance(axi_stream_out_ports, list)
    assert len(axi_stream_out_ports) > 0
    for port in axi_stream_out_ports:
        assert isinstance(port, Port)
        logger.debug(f"AXI Stream Out port: {port.name}, Direction: {port.direction}, Width: {port.width}")

    # Example specific checks
    tdata_present = any("TDATA" in p.name and p.direction == Direction.OUTPUT for p in axi_stream_out_ports)
    tvalid_present = any("TVALID" in p.name and p.direction == Direction.OUTPUT for p in axi_stream_out_ports)
    tready_present = any("TREADY" in p.name and p.direction == Direction.INPUT for p in axi_stream_out_ports)
    assert tdata_present
    assert tvalid_present
    assert tready_present
    logger.info(f"Successfully tested axi_stream_out_ports: found {len(axi_stream_out_ports)} ports.")

# Example for axilite_config_ports fixture
def test_example_axilite_config_ports(axilite_config_ports):
    """Demonstrates the use of the axilite_config_ports fixture."""
    logger.info("Testing axilite_config_ports fixture")
    assert isinstance(axilite_config_ports, list)
    assert len(axilite_config_ports) > 0
    for port in axilite_config_ports:
        assert isinstance(port, Port)
        logger.debug(f"AXI-Lite Config port: {port.name}, Direction: {port.direction}, Width: {port.width}")
    
    # Example specific checks (e.g., AWADDR, WDATA)
    awaddr_present = any("AWADDR" in p.name and p.direction == Direction.INPUT for p in axilite_config_ports)
    wdata_present = any("WDATA" in p.name and p.direction == Direction.INPUT for p in axilite_config_ports)
    assert awaddr_present
    assert wdata_present
    logger.info(f"Successfully tested axilite_config_ports: found {len(axilite_config_ports)} ports.")

# Example for valid_module_content fixture
def test_example_valid_module_content(valid_module_content, temp_sv_file, parser_debug):
    """Demonstrates the use of the valid_module_content fixture."""
    logger.info("Testing valid_module_content fixture")
    assert isinstance(valid_module_content, str)
    assert "module valid_test" in valid_module_content # Check for module name
    assert "ap_clk" in valid_module_content           # Check for a global signal
    assert "in0_TDATA" in valid_module_content        # Check for an AXI-S input
    assert "out0_TDATA" in valid_module_content       # Check for an AXI-S output
    
    # Optionally, try parsing it
    file_path = temp_sv_file(valid_module_content, "valid_module_example.sv")
    try:
        kernel = parser_debug.parse_file(file_path)
        assert kernel is not None
        assert kernel.name == "valid_test"
        logger.info(f"Successfully parsed valid_module_content: Kernel name '{kernel.name}'")
    except Exception as e:
        logger.error(f"Failed to parse valid_module_content: {e}")
        pytest.fail(f"Parsing valid_module_content failed: {e}")
    logger.info("Successfully tested valid_module_content.")

# Example for modifying valid_module_content at runtime
def test_example_modifying_valid_module_content(valid_module_content, temp_sv_file, parser_debug):
    """Demonstrates modifying the valid_module_content fixture at runtime."""
    logger.info("Testing runtime modification of valid_module_content fixture")
    original_content = valid_module_content
    
    # 1. Replace a substring (e.g., change a port name or module name)
    # Let's change the module name from 'valid_test' to 'modified_test'
    modified_content_v1 = original_content.replace("module valid_test", "module modified_test")
    assert "module modified_test" in modified_content_v1
    assert "module valid_test" not in modified_content_v1
    
    # Try parsing the first modification
    file_path_v1 = temp_sv_file(modified_content_v1, "modified_module_v1.sv")
    try:
        kernel_v1 = parser_debug.parse_file(file_path_v1)
        assert kernel_v1 is not None
        assert kernel_v1.name == "modified_test"
        logger.info(f"Successfully parsed modified_content_v1: Kernel name '{kernel_v1.name}'")
    except Exception as e:
        logger.error(f"Failed to parse modified_content_v1: {e}")
        pytest.fail(f"Parsing modified_content_v1 failed: {e}")

    # 2. Remove a line (e.g., remove a specific port declaration)
    # Let's remove the 'ap_clk' port declaration.
    # This is a bit more fragile and depends on the exact formatting.
    lines = original_content.splitlines()
    lines_without_ap_clk = [line for line in lines if "input logic ap_clk," not in line and "input logic ap_clk " not in line] # Handle with or without trailing comma
    modified_content_v2 = "\\n".join(lines_without_ap_clk)
    assert "ap_clk" not in modified_content_v2 # This might be too broad if "ap_clk" appears elsewhere

    # To be more precise, check if the specific line is gone:
    assert "input logic ap_clk," not in modified_content_v2
    assert "input logic ap_clk " not in modified_content_v2


    # Try parsing the second modification (this might fail if ap_clk is essential for the parser's logic)
    # For demonstration, we'll assume removing ap_clk might make it unparsable or change its properties.
    # Depending on the parser's strictness, this could raise an error or result in a kernel without an ap_clk port.
    file_path_v2 = temp_sv_file(modified_content_v2, "modified_module_v2.sv")
    try:
        kernel_v2 = parser_debug.parse_file(file_path_v2)
        assert kernel_v2 is not None
        # Check that ap_clk is no longer in the parsed kernel's ports
        assert not any(p.name == "ap_clk" for p in kernel_v2.ports)
        logger.info(f"Successfully parsed modified_content_v2: Kernel name '{kernel_v2.name}', ap_clk port removed.")
    except Exception as e:
        # This might be expected if ap_clk is mandatory for the parser logic used in valid_module_content
        logger.warning(f"Parsing modified_content_v2 (ap_clk removed) resulted in an error (as might be expected): {e}")
        # If removal should lead to a specific error, you can assert that here.
        # For this example, we'll just log it.

    # 3. Example: Removing a specific interface (e.g. in0)
    # This requires more careful manipulation, potentially using regex or more complex string processing
    # if the structure is consistent.
    # For instance, removing all lines related to 'in0_'
    lines_v3 = original_content.splitlines()
    modified_content_v3_lines = []
    for line in lines_v3:
        if "in0_" not in line:
            modified_content_v3_lines.append(line)
    modified_content_v3 = "\\n".join(modified_content_v3_lines)
    
    # Remove trailing commas from port lists if an in0 port was the last one in a group
    modified_content_v3 = modified_content_v3.replace(",\\n);", "\\n);") # Comma before closing parenthesis
    modified_content_v3 = modified_content_v3.replace(",\\n    // AXI-Lite slave interface for control", "\\n    // AXI-Lite slave interface for control") # Example if in0 was before s_axi_control

    file_path_v3 = temp_sv_file(modified_content_v3, "modified_module_v3.sv")
    try:
        kernel_v3 = parser_debug.parse_file(file_path_v3)
        assert kernel_v3 is not None
        assert not any("in0_" in p.name for p in kernel_v3.ports)
        logger.info(f"Successfully parsed modified_content_v3: Kernel name '{kernel_v3.name}', 'in0_' related ports removed.")
    except Exception as e:
        logger.error(f"Failed to parse modified_content_v3: {e}")
        pytest.fail(f"Parsing modified_content_v3 (in0 ports removed) failed: {e}")

    logger.info("Successfully demonstrated runtime modification of valid_module_content.")

# Example combining multiple fixtures
def test_example_combined_fixtures(parser_debug, temp_sv_file, global_ports, axi_stream_in_ports):
    """Demonstrates combining multiple fixtures in one test."""
    logger.info("Testing combined fixtures")
    
    # Use parser_debug
    assert isinstance(parser_debug, RTLParser)
    
    # Create a simple module using temp_sv_file
    module_name = "combined_test_module"
    ports_str_parts = []
    for port in global_ports:
        ports_str_parts.append(f"{port.direction.to_sv_keyword()} logic [{int(port.width)-1}:0] {port.name}" if port.width != "1" 
                               else f"{port.direction.to_sv_keyword()} logic {port.name}")
    for port in axi_stream_in_ports:
         ports_str_parts.append(f"{port.direction.to_sv_keyword()} logic [{int(port.width)-1}:0] {port.name}" if port.width != "1" 
                               else f"{port.direction.to_sv_keyword()} logic {port.name}")

    ports_str = ",\\n    ".join(ports_str_parts)
    
    content = f"""
module {module_name} (
    {ports_str}
);
    // Minimal body
    assign in0_TREADY = 1'b1; 
endmodule
"""
    file_path = temp_sv_file(content, "combined_module.sv")
    assert os.path.exists(file_path)
    
    # Parse the created file
    try:
        kernel = parser_debug.parse_file(file_path)
        assert kernel.name == module_name
        assert len(kernel.ports) == len(global_ports) + len(axi_stream_in_ports)
        logger.info(f"Successfully parsed combined module: {kernel.name} with {len(kernel.ports)} ports.")
    except Exception as e:
        logger.error(f"Failed to parse combined module: {e}")
        pytest.fail(f"Parsing combined module failed: {e}")
        
    logger.info("Successfully tested combined fixtures.")

# Example for using individual module part constants
from .test_fixtures import (
    VALID_HEADER_PARAMS_PORTSOPEN,
    VALID_GLOBAL_SIGNALS,
    VALID_AXI_STREAM_IN_INTERFACE,
    # VALID_AXI_STREAM_OUT_INTERFACE, # Let's omit this for a custom example
    VALID_PORTS_CLOSE,
    VALID_MODULE_BODY_CONTENT,
    VALID_ENDMODULE_STATEMENT,
    HEADER_PARAMS_PLACEHOLDER,
    VALID_MIN_INTERFACES, # For comparison or alternative construction
    VALID_MODULE_BODY
)

def test_example_using_module_part_constants(temp_sv_file, parser_debug):
    """Demonstrates using individual VALID_* constants to build module content."""
    logger.info("Testing usage of individual module part constants")

    # Scenario 1: Build a module with only global signals and an AXI stream input
    # We'll use a custom module name and omit the AXI stream output for this example.
    custom_module_name = "my_custom_module"
    custom_header = VALID_HEADER_PARAMS_PORTSOPEN.replace("valid_module", custom_module_name)
    
    # Constructing ports part by part
    # VALID_GLOBAL_SIGNALS ends with a comma.
    # VALID_AXI_STREAM_IN_INTERFACE ends with a comma.
    # If this is the last interface, the final comma needs to be removed.
    ports_content_parts = [
        VALID_GLOBAL_SIGNALS,
        VALID_AXI_STREAM_IN_INTERFACE.strip()
    ]
    # Join and remove trailing comma if present from the last element before joining
    if ports_content_parts[-1].endswith(','):
      ports_content_parts[-1] = ports_content_parts[-1][:-1]
    
    ports_content = "\\n".join(ports_content_parts)

    custom_module_content_v1 = f"""\\
{custom_header}
{ports_content}
{VALID_PORTS_CLOSE}
{VALID_MODULE_BODY_CONTENT}
{VALID_ENDMODULE_STATEMENT}
"""
    logger.debug(f"Custom module content v1:\\n{custom_module_content_v1}")

    assert custom_module_name in custom_module_content_v1
    assert "in0_TDATA" in custom_module_content_v1
    assert "out0_TDATA" not in custom_module_content_v1 # We omitted this

    file_path_v1 = temp_sv_file(custom_module_content_v1, "custom_module_v1.sv")
    try:
        kernel_v1 = parser_debug.parse_file(file_path_v1)
        assert kernel_v1 is not None
        assert kernel_v1.name == custom_module_name
        assert any(p.name == "in0_TDATA" for p in kernel_v1.ports)
        assert not any(p.name == "out0_TDATA" for p in kernel_v1.ports)
        logger.info(f"Successfully parsed custom_module_content_v1: Kernel '{kernel_v1.name}'")
    except Exception as e:
        logger.error(f"Failed to parse custom_module_content_v1: {e}")
        pytest.fail(f"Parsing custom_module_content_v1 failed: {e}")

    # Scenario 2: Using HEADER_PARAMS_PLACEHOLDER and VALID_MIN_INTERFACES, VALID_MODULE_BODY
    parameter_definition = "parameter DATA_WIDTH = 64"
    header_with_custom_params = HEADER_PARAMS_PLACEHOLDER.replace("<PLACEHOLDER>", parameter_definition)
    
    # VALID_MIN_INTERFACES is a pre-combined string of global, axi-s in, and axi-s out.
    # The last port in VALID_MIN_INTERFACES (from VALID_AXI_STREAM_OUT_INTERFACE) does not have a trailing comma.
    # VALID_MODULE_BODY starts with VALID_PORTS_CLOSE.
    # So, the direct concatenation should be syntactically correct.
    custom_module_content_v2 = f"""\\
{header_with_custom_params}
{VALID_MIN_INTERFACES}
{VALID_MODULE_BODY}
"""

    logger.debug(f"Custom module content v2:\\n{custom_module_content_v2}")
    assert "parameter DATA_WIDTH = 64" in custom_module_content_v2
    assert "in0_TDATA" in custom_module_content_v2
    assert "out0_TDATA" in custom_module_content_v2 # From VALID_MIN_INTERFACES

    file_path_v2 = temp_sv_file(custom_module_content_v2, "custom_module_v2.sv")
    try:
        kernel_v2 = parser_debug.parse_file(file_path_v2)
        assert kernel_v2 is not None
        assert kernel_v2.name == "valid_module" # Default name from HEADER_PARAMS_PLACEHOLDER
        assert any(p.name == "DATA_WIDTH" for p in kernel_v2.parameters)
        logger.info(f"Successfully parsed custom_module_content_v2: Kernel '{kernel_v2.name}'")
    except Exception as e:
        logger.error(f"Failed to parse custom_module_content_v2: {e}")
        pytest.fail(f"Parsing custom_module_content_v2 failed: {e}")

    logger.info("Successfully demonstrated using individual module part constants.")
