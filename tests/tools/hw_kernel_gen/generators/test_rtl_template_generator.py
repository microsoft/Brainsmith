import pytest
from pathlib import Path
import jinja2
import shutil

# Assuming the test execution is from the root directory
# Adjust the path if necessary based on your test runner configuration
from brainsmith.tools.hw_kernel_gen.generators.rtl_template_generator import generate_rtl_template
from brainsmith.tools.hw_kernel_gen.rtl_parser import (
    HWKernel, Parameter, Port, PortGroup, InterfaceType, Direction, Interface, ValidationResult # Added Interface, ValidationResult
)

# Fixture to create a mock HWKernel object for testing
@pytest.fixture
def mock_hw_kernel():
    # Define some parameters
    params = [
        Parameter(name="DATA_WIDTH", param_type="int", default_value="32"),
        Parameter(name="ADDR_WIDTH", param_type="int", default_value="16"),
    ]

    # Define some ports and interfaces (as PortGroups first)
    clk_port = Port(name="ap_clk", direction=Direction.INPUT, width=1)
    rst_port = Port(name="ap_rst_n", direction=Direction.INPUT, width=1)
    global_control_pg = PortGroup(
        name="global",
        interface_type=InterfaceType.GLOBAL_CONTROL,
        ports={"ap_clk": clk_port, "ap_rst_n": rst_port},
        metadata={}
    )

    s_axis_tdata = Port(name="s_axis_input_TDATA", direction=Direction.INPUT, width="DATA_WIDTH")
    s_axis_tvalid = Port(name="s_axis_input_TVALID", direction=Direction.INPUT, width=1)
    s_axis_tready = Port(name="s_axis_input_TREADY", direction=Direction.OUTPUT, width=1)
    s_axis_tlast = Port(name="s_axis_input_TLAST", direction=Direction.INPUT, width=1)
    stream_in_pg = PortGroup(
        name="s_axis_input",
        interface_type=InterfaceType.AXI_STREAM,
        ports={
            "TDATA": s_axis_tdata,
            "TVALID": s_axis_tvalid,
            "TREADY": s_axis_tready,
            "TLAST": s_axis_tlast,
        },
        metadata={"data_width_expr": "DATA_WIDTH-1:0"}
    )

    m_axis_tdata = Port(name="m_axis_output_TDATA", direction=Direction.OUTPUT, width="DATA_WIDTH")
    m_axis_tvalid = Port(name="m_axis_output_TVALID", direction=Direction.OUTPUT, width=1)
    m_axis_tready = Port(name="m_axis_output_TREADY", direction=Direction.INPUT, width=1)
    m_axis_tlast = Port(name="m_axis_output_TLAST", direction=Direction.OUTPUT, width=1)
    stream_out_pg = PortGroup(
        name="m_axis_output",
        interface_type=InterfaceType.AXI_STREAM,
        ports={
            "TDATA": m_axis_tdata,
            "TVALID": m_axis_tvalid,
            "TREADY": m_axis_tready,
            "TLAST": m_axis_tlast,
        },
        metadata={"data_width_expr": "DATA_WIDTH-1:0"}
    )

    s_axil_awaddr = Port(name="s_axil_control_AWADDR", direction=Direction.INPUT, width="ADDR_WIDTH")
    s_axil_awvalid = Port(name="s_axil_control_AWVALID", direction=Direction.INPUT, width=1)
    # Add more AXI-Lite ports as needed for a more complete test...
    axilite_control_pg = PortGroup(
        name="s_axil_control",
        interface_type=InterfaceType.AXI_LITE,
        ports={
            "AWADDR": s_axil_awaddr,
            "AWVALID": s_axil_awvalid,
            # ... other AXI-Lite ports
        },
        metadata={
            "addr_width_expr": "ADDR_WIDTH-1:0",
            "data_width_expr": "31:0"
        }
    )

    # Create dummy ValidationResult
    valid_result = ValidationResult(valid=True)

    # Convert PortGroups to Interface objects and create the dictionary
    interfaces_dict = {
        iface.name: Interface(
            name=iface.name,
            type=iface.interface_type, # Use 'type' for Interface object
            ports=iface.ports,
            validation_result=valid_result,
            metadata=iface.metadata
        )
        for iface in [global_control_pg, stream_in_pg, stream_out_pg, axilite_control_pg]
    }

    return HWKernel(
        name="test_kernel",
        parameters=params,
        interfaces=interfaces_dict, # Pass the dictionary of Interface objects
    )

# Test the basic successful generation of the wrapper file
def test_generate_rtl_template_success(mock_hw_kernel, tmp_path):
    """Test that the generator creates the output file successfully."""
    output_dir = tmp_path / "generated_rtl"
    expected_output_file = output_dir / f"{mock_hw_kernel.name}_wrapper.v" # Corrected: module_name -> name

    # Ensure the output directory doesn't exist initially (handled by tmp_path)
    assert not output_dir.exists()

    # Run the generator
    generated_file_path = generate_rtl_template(mock_hw_kernel, output_dir)

    # Assertions
    assert generated_file_path == expected_output_file
    assert output_dir.exists()
    assert expected_output_file.exists()
    assert expected_output_file.is_file()

    # Basic content check (can be expanded)
    content = expected_output_file.read_text()
    assert f"module {mock_hw_kernel.name}_wrapper #(" in content # Corrected: module_name -> name
    assert f"parameter DATA_WIDTH = $DATA_WIDTH$" in content
    assert f"parameter ADDR_WIDTH = $ADDR_WIDTH$" in content
    assert "input ap_clk," in content
    assert "input ap_rst_n," in content
    assert "input [DATA_WIDTH-1:0] s_axis_input_TDATA," in content
    assert "output s_axis_input_TREADY," in content
    assert "output [DATA_WIDTH-1:0] m_axis_output_TDATA," in content
    assert "input m_axis_output_TREADY," in content
    assert "input [ADDR_WIDTH-1:0] s_axil_control_AWADDR," in content
    assert f") {mock_hw_kernel.name}_inst (" in content # Check instance name
    assert ".clk( clk )," in content
    assert ".rst_n( rst_n )," in content
    assert ".s_axis_input_TDATA( s_axis_input_TDATA )," in content
    assert ".m_axis_output_TREADY( m_axis_output_TREADY )" in content # Last connection, no comma
    assert "endmodule" in content

# Test case for when the template file is not found
def test_generate_rtl_template_not_found(mock_hw_kernel, tmp_path, monkeypatch):
    """Test FileNotFoundError when the template is missing."""
    output_dir = tmp_path / "generated_rtl_notfound"

    # Mock the template file path to ensure it doesn't exist
    # Use monkeypatch to temporarily change the internal variable
    monkeypatch.setattr("brainsmith.tools.hw_kernel_gen.generators.rtl_template_generator._TEMPLATE_FILE", "non_existent_template.j2")

    with pytest.raises(jinja2.TemplateNotFound, match="non_existent_template.j2"): # Expect TemplateNotFound
        generate_rtl_template(mock_hw_kernel, output_dir)

    # Check that the output directory might have been created, but the file wasn't
    assert output_dir.exists()
    assert not list(output_dir.glob('*')) # No files generated

# Test case for template rendering errors (e.g., undefined variable)
# This requires a way to inject a faulty template or context.
# Option 1: Mock the template object itself (more complex)
# Option 2: Temporarily modify the real template (simpler for this test)
def test_generate_rtl_template_render_error(mock_hw_kernel, tmp_path, monkeypatch):
    """Test jinja2.TemplateError during rendering."""
    output_dir = tmp_path / "generated_rtl_render_error"

    # Determine project root dynamically (assuming test is run from within the project)
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent # Use resolve() for robustness
    template_dir = project_root / "brainsmith/tools/hw_kernel_gen/templates"

    original_template_path = template_dir / "rtl_wrapper.v.j2"
    backup_template_path = template_dir / "rtl_wrapper.v.j2.bak"

    # Create a faulty template content
    faulty_content = "module {{ kernel.name }}_wrapper (\\n {{ undefined_variable }}\\n);\\nendmodule" # Corrected: kernel.module_name -> kernel.name

    try:
        # Backup original template
        if original_template_path.exists():
            shutil.move(str(original_template_path), str(backup_template_path))

        # Write faulty template
        original_template_path.write_text(faulty_content)

        # Expect a TemplateError (specifically UndefinedError is caught and re-raised)
        with pytest.raises(jinja2.UndefinedError, match="'undefined_variable' is undefined"): # Expect UndefinedError and specific message
            generate_rtl_template(mock_hw_kernel, output_dir)

        # Check that the output directory might have been created, but the file wasn't
        assert output_dir.exists()
        assert not list(output_dir.glob('*')) # No files generated

    finally:
        # Restore original template
        if backup_template_path.exists():
            shutil.move(str(backup_template_path), str(original_template_path))
        elif original_template_path.exists() and original_template_path.read_text() == faulty_content:
             # Clean up the faulty file if backup didn't exist or restore failed
             original_template_path.unlink()

