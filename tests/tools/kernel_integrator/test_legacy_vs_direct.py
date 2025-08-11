"""Test comparing legacy vs direct generation outputs."""
import pytest
from pathlib import Path
from brainsmith.tools.kernel_integrator.kernel_integrator import KernelIntegrator
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata, InterfaceMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Parameter, ParameterCategory
from brainsmith.core.dataflow.types import InterfaceType


def create_test_metadata():
    """Create comprehensive test metadata."""
    # Need to import Port and PortDirection for control interface
    from brainsmith.tools.kernel_integrator.types.rtl import Port, PortDirection
    
    # Create interfaces
    interfaces = [
        InterfaceMetadata(
            name="input",
            interface_type=InterfaceType.INPUT,
            bdim_shape=["B", "M"],
            sdim_shape=["K", "N"]
        ),
        InterfaceMetadata(
            name="output", 
            interface_type=InterfaceType.OUTPUT,
            bdim_shape=["B", "P"]
        ),
        InterfaceMetadata(
            name="weights",
            interface_type=InterfaceType.WEIGHT
        ),
        # Add control interface for legacy compatibility
        InterfaceMetadata(
            name="ap_ctrl",
            interface_type=InterfaceType.CONTROL,
            ports=[
                Port(name="ap_clk", width="1", direction=PortDirection.INPUT),
                Port(name="ap_rst_n", width="1", direction=PortDirection.INPUT),
                Port(name="ap_start", width="1", direction=PortDirection.INPUT),
                Port(name="ap_done", width="1", direction=PortDirection.OUTPUT),
                Port(name="ap_idle", width="1", direction=PortDirection.OUTPUT),
                Port(name="ap_ready", width="1", direction=PortDirection.OUTPUT)
            ]
        )
    ]
    
    # Create parameters
    parameters = [
        # Shape parameters
        Parameter(name="B", category=ParameterCategory.SHAPE),
        Parameter(name="M", category=ParameterCategory.SHAPE),
        Parameter(name="K", category=ParameterCategory.SHAPE),
        Parameter(name="N", category=ParameterCategory.SHAPE),
        Parameter(name="P", category=ParameterCategory.SHAPE),
        # Algorithm parameters
        Parameter(name="algo_param", category=ParameterCategory.ALGORITHM),
        # Datatype parameter
        Parameter(
            name="input_dtype",
            category=ParameterCategory.DATATYPE,
            interface_name="input"
        )
    ]
    
    return KernelMetadata(
        name="test_kernel",
        interfaces=interfaces,
        parameters=parameters,
        source_file="test.v",
        exposed_parameters=["B", "M"],
        internal_datatypes=[],
        linked_parameters={}
    )


def test_generation_comparison():
    """Compare legacy vs direct generation outputs."""
    metadata = create_test_metadata()
    
    # Generate with legacy system
    legacy_integrator = KernelIntegrator(use_direct_generators=False)
    legacy_result = legacy_integrator.generate_and_write(
        metadata, 
        write_files=False
    )
    
    # Generate with direct system
    direct_integrator = KernelIntegrator(use_direct_generators=True)
    direct_result = direct_integrator.generate_and_write(
        metadata,
        write_files=False
    )
    
    # Both should succeed
    if legacy_result.errors:
        print(f"Legacy errors: {legacy_result.errors}")
    if direct_result.errors:
        print(f"Direct errors: {direct_result.errors}")
    assert len(legacy_result.errors) == 0
    assert len(direct_result.errors) == 0
    
    # Should generate same number of files
    assert len(legacy_result.generated_files) == len(direct_result.generated_files)
    
    # Check that all expected files are present
    legacy_files = {f.path.name for f in legacy_result.generated_files}
    direct_files = {f.path.name for f in direct_result.generated_files}
    
    # Legacy uses different naming for HWCustomOp
    expected_legacy_files = {
        'test_kernel_hw_custom_op.py',
        'test_kernel_rtl.py', 
        'test_kernel_wrapper.v'
    }
    expected_direct_files = {
        'test_kernel.py',
        'test_kernel_rtl.py', 
        'test_kernel_wrapper.v'
    }
    
    assert legacy_files == expected_legacy_files
    assert direct_files == expected_direct_files
    
    # Compare file sizes (should be similar)
    # Map legacy names to direct names
    file_mapping = {
        'test_kernel_hw_custom_op.py': 'test_kernel.py',
        'test_kernel_rtl.py': 'test_kernel_rtl.py',
        'test_kernel_wrapper.v': 'test_kernel_wrapper.v'
    }
    
    for legacy_file in legacy_result.generated_files:
        direct_name = file_mapping.get(legacy_file.path.name, legacy_file.path.name)
        direct_file = next(f for f in direct_result.generated_files 
                          if f.path.name == direct_name)
        
        # Content should be non-empty
        assert len(legacy_file.content) > 0
        assert len(direct_file.content) > 0
        
        # Sizes should be within reasonable range
        # (allowing for template differences)
        size_ratio = len(direct_file.content) / len(legacy_file.content)
        assert 0.5 < size_ratio < 2.0, f"Size difference too large for {legacy_file.path.name}"
        
        print(f"\n{legacy_file.path.name}:")
        print(f"  Legacy: {len(legacy_file.content)} chars")
        print(f"  Direct: {len(direct_file.content)} chars")
        print(f"  Ratio: {size_ratio:.2f}")


def test_direct_generation_content():
    """Test that direct generation produces valid content."""
    metadata = create_test_metadata()
    
    integrator = KernelIntegrator(use_direct_generators=True)
    result = integrator.generate_and_write(metadata, write_files=False)
    
    assert len(result.errors) == 0
    assert len(result.generated_files) == 3
    
    # Check HWCustomOp content
    hw_file = next(f for f in result.generated_files if f.path.name == 'test_kernel.py')
    assert 'class TestKernel(AutoHWCustomOp):' in hw_file.content
    assert 'def define_kernel' in hw_file.content
    assert 'def get_nodeattr_types' in hw_file.content
    
    # Check RTL Backend content
    rtl_file = next(f for f in result.generated_files if f.path.name == 'test_kernel_rtl.py')
    assert 'class test_kernel_rtl' in rtl_file.content
    assert 'AutoRTLBackend' in rtl_file.content
    assert 'def get_template_params' in rtl_file.content
    
    # Check RTL Wrapper content
    wrapper_file = next(f for f in result.generated_files if f.path.name == 'test_kernel_wrapper.v')
    assert 'module test_kernel_wrapper' in wrapper_file.content
    assert 'parameter B =' in wrapper_file.content
    assert 'input wire ap_clk' in wrapper_file.content


if __name__ == "__main__":
    # Run comparison when executed directly
    test_generation_comparison()
    test_direct_generation_content()
    print("\nAll tests passed!")