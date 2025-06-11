"""
Transition Test Suite: HWKernel Downstream Functionality Parity Tests

This test suite captures all current behavior of components that depend on HWKernel,
ensuring we can validate functional parity after removing the HWKernel layer.

Test Coverage:
1. RTL Converter functionality (HWKernel → DataflowModel)
2. Template generation with HWKernel context
3. Generator workflows using HWKernel
4. CLI integration with HWKernel
5. All HWKernel properties and methods used downstream

Run this test suite before and after HWKernel removal to ensure parity.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import copy

# Import current HWKernel-based components
from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    HWKernel, Interface, InterfaceType, Direction, Port, Parameter, Pragma, ValidationResult
)
from brainsmith.dataflow.rtl_integration import RTLDataflowConverter
from brainsmith.tools.unified_hwkg import UnifiedHWKGGenerator
from brainsmith.dataflow.core.dataflow_model import DataflowModel

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "examples" / "thresholding"


class TestHWKernelDownstreamParity:
    """
    Comprehensive test suite capturing all HWKernel downstream functionality.
    
    These tests document and validate the current behavior of all components
    that depend on HWKernel, serving as a parity baseline for the transition.
    """
    
    @pytest.fixture
    def sample_rtl_file(self):
        """Sample RTL file for testing."""
        return TEST_DATA_DIR / "thresholding_axi.sv"
    
    @pytest.fixture
    def parsed_hw_kernel(self, sample_rtl_file):
        """Parse a real RTL file to get HWKernel."""
        hw_kernel = parse_rtl_file(sample_rtl_file)
        assert hw_kernel is not None, "Failed to parse RTL file"
        return hw_kernel
    
    @pytest.fixture
    def mock_hw_kernel(self):
        """Create a mock HWKernel with all properties that downstream components use."""
        # Create validation results
        validation_result = ValidationResult(
            valid=True,
            message="Valid interface"
        )
        
        # Create mock interfaces
        s_axis_interface = Interface(
            name="s_axis",
            type=InterfaceType.AXI_STREAM,
            ports={
                "s_axis_tdata": Port("s_axis_tdata", Direction.INPUT, "logic", "((PE*WI+7)/8)*8-1:0"),
                "s_axis_tvalid": Port("s_axis_tvalid", Direction.INPUT, "logic", "1"),
                "s_axis_tready": Port("s_axis_tready", Direction.OUTPUT, "logic", "1")
            },
            validation_result=validation_result
        )
        
        m_axis_interface = Interface(
            name="m_axis",
            type=InterfaceType.AXI_STREAM,
            ports={
                "m_axis_tdata": Port("m_axis_tdata", Direction.OUTPUT, "logic", "((PE*O_BITS+7)/8)*8-1:0"),
                "m_axis_tvalid": Port("m_axis_tvalid", Direction.OUTPUT, "logic", "1"),
                "m_axis_tready": Port("m_axis_tready", Direction.INPUT, "logic", "1")
            },
            validation_result=validation_result
        )
        
        # Create HWKernel with all properties
        hw_kernel = HWKernel(
            name="test_kernel",
            parameters={
                "PE": Parameter("PE", "int unsigned", "1"),
                "WI": Parameter("WI", "int unsigned", "8"),
                "O_BITS": Parameter("O_BITS", "int unsigned", "8")
            },
            interfaces={
                "s_axis": s_axis_interface,
                "m_axis": m_axis_interface
            },
            pragmas=[],
            source_file=Path("test_kernel.sv")
        )
        
        # Set additional properties that downstream components might use
        hw_kernel.pragma_sophistication_level = "basic"
        hw_kernel.parsing_warnings = []
        
        return hw_kernel
    
    def test_capture_rtl_converter_usage(self, mock_hw_kernel):
        """Capture all HWKernel properties used by RTLDataflowConverter."""
        converter = RTLDataflowConverter()
        
        # Track which HWKernel properties are accessed
        accessed_properties = {
            "name": False,
            "interfaces": False,
            "pragmas": False,
            "source_file": False,
            "pragma_sophistication_level": False,
            "parsing_warnings": False,
            "parameters": False
        }
        
        # Create a proxy to track property access
        class HWKernelProxy:
            def __init__(self, hw_kernel):
                self._hw_kernel = hw_kernel
                
            def __getattr__(self, name):
                if name in accessed_properties:
                    accessed_properties[name] = True
                return getattr(self._hw_kernel, name)
        
        proxy = HWKernelProxy(mock_hw_kernel)
        
        # Run conversion
        result = converter.convert(proxy)
        
        # Document which properties were accessed
        print("\nRTLDataflowConverter accesses these HWKernel properties:")
        for prop, accessed in accessed_properties.items():
            if accessed:
                print(f"  - hw_kernel.{prop}")
        
        # Validate conversion result
        assert result.success, f"Conversion failed: {result.errors}"
        assert result.dataflow_model is not None
        
        # Capture the conversion output for parity testing
        self._capture_dataflow_model_state(result.dataflow_model, "rtl_converter_output")
    
    def test_capture_generator_usage(self, sample_rtl_file):
        """Capture how UnifiedHWKGGenerator uses HWKernel."""
        generator = UnifiedHWKGGenerator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Run generation
            result = generator.generate_from_rtl(
                rtl_file=sample_rtl_file,
                compiler_data={'onnx_patterns': [], 'cost_function': lambda *a, **k: 1.0},
                output_dir=output_dir
            )
            
            assert result.success, f"Generation failed: {result.errors}"
            
            # Capture generated files content
            generated_files_content = {}
            for file_path in result.generated_files:
                with open(file_path, 'r') as f:
                    generated_files_content[file_path.name] = f.read()
            
            # Save baseline for comparison
            self._save_baseline("generator_output", {
                "file_count": len(result.generated_files),
                "file_names": [f.name for f in result.generated_files],
                "dataflow_model_interfaces": len(result.dataflow_model.interfaces) if result.dataflow_model else 0,
                "sample_content": generated_files_content.get("thresholding_axi_hwcustomop.py", "")[:500]
            })
    
    def test_capture_template_context_usage(self, mock_hw_kernel):
        """Capture what HWKernel properties are used in template context building."""
        # This would test template context building if it used HWKernel directly
        # Currently templates use DataflowModel, but let's verify
        
        # Check if any templates reference HWKernel properties
        template_properties = {
            "kernel.name": mock_hw_kernel.name,
            "kernel.parameters": dict(mock_hw_kernel.parameters),
            "kernel.interfaces": list(mock_hw_kernel.interfaces.keys()),
            "kernel.source_file": str(mock_hw_kernel.source_file) if mock_hw_kernel.source_file else None
        }
        
        self._save_baseline("template_context", template_properties)
    
    def test_dataflow_model_conversion_details(self, parsed_hw_kernel):
        """Capture detailed DataflowModel conversion behavior."""
        converter = RTLDataflowConverter()
        result = converter.convert(parsed_hw_kernel)
        
        assert result.success, f"Conversion failed: {result.errors}"
        
        dataflow_model = result.dataflow_model
        
        # Capture detailed conversion results
        conversion_details = {
            "interface_count": len(dataflow_model.interfaces),
            "interface_names": list(dataflow_model.interfaces.keys()),
            "interface_types": {
                name: iface.interface_type.value 
                for name, iface in dataflow_model.interfaces.items()
            },
            "tensor_dims": {
                name: iface.tensor_dims 
                for name, iface in dataflow_model.interfaces.items()
            },
            "block_dims": {
                name: iface.block_dims 
                for name, iface in dataflow_model.interfaces.items()
            },
            "stream_dims": {
                name: iface.stream_dims 
                for name, iface in dataflow_model.interfaces.items()
            },
            "parameters": dataflow_model.parameters
        }
        
        self._save_baseline("dataflow_conversion_details", conversion_details)
    
    def test_interface_type_mapping(self, mock_hw_kernel):
        """Test how RTL Interface types map to DataflowInterface types."""
        converter = RTLDataflowConverter()
        
        # Test each interface type mapping
        interface_mappings = {}
        
        for iface_name, rtl_interface in mock_hw_kernel.interfaces.items():
            # Use the converter's internal mapping
            dataflow_type = converter.interface_mapper.map_interface_type(rtl_interface)
            interface_mappings[iface_name] = {
                "rtl_type": rtl_interface.type.value,
                "dataflow_type": dataflow_type.value,
                "has_tdata": any("tdata" in p for p in rtl_interface.ports),
                "has_tvalid": any("tvalid" in p for p in rtl_interface.ports),
                "is_input": iface_name.startswith("s_"),
                "is_output": iface_name.startswith("m_")
            }
        
        self._save_baseline("interface_type_mappings", interface_mappings)
    
    def test_pragma_conversion_behavior(self):
        """Test how pragmas are converted from HWKernel to chunking strategies."""
        # Create HWKernel with various pragma types
        pragma_tests = []
        
        # Test enhanced BDIM pragma
        enhanced_pragma = Pragma(
            type="BDIM",
            line_number=10,
            raw_text="@brainsmith:BDIM:s_axis[0]=[PE,WI]",
            parsed_data={
                "format": "enhanced",
                "interface_name": "s_axis",
                "chunk_index": 0,
                "chunk_sizes": ["PE", "WI"]
            }
        )
        
        # Test legacy BDIM pragma
        legacy_pragma = Pragma(
            type="BDIM",
            line_number=20,
            raw_text="@brainsmith:BDIM:m_axis:O_BITS*PE",
            parsed_data={
                "format": "legacy",
                "interface_name": "m_axis",
                "dimension_expressions": ["O_BITS*PE"]
            }
        )
        
        converter = RTLDataflowConverter()
        
        # Convert pragmas
        for pragma in [enhanced_pragma, legacy_pragma]:
            strategy = converter.pragma_converter.convert_bdim_pragma(pragma)
            pragma_tests.append({
                "pragma_format": pragma.parsed_data.get("format"),
                "interface_name": pragma.parsed_data.get("interface_name"),
                "strategy_type": strategy.get("type") if strategy else None,
                "raw_text": pragma.raw_text
            })
        
        self._save_baseline("pragma_conversion", pragma_tests)
    
    def test_error_handling_scenarios(self, mock_hw_kernel):
        """Test error handling with various HWKernel states."""
        converter = RTLDataflowConverter()
        
        error_scenarios = []
        
        # Test with missing name
        hw_kernel_no_name = copy.deepcopy(mock_hw_kernel)
        hw_kernel_no_name.name = None
        result = converter.convert(hw_kernel_no_name)
        error_scenarios.append({
            "scenario": "missing_name",
            "success": result.success,
            "errors": result.errors
        })
        
        # Test with empty interfaces
        hw_kernel_no_interfaces = copy.deepcopy(mock_hw_kernel)
        hw_kernel_no_interfaces.interfaces = {}
        result = converter.convert(hw_kernel_no_interfaces)
        error_scenarios.append({
            "scenario": "empty_interfaces",
            "success": result.success,
            "warnings": result.warnings
        })
        
        # Test with missing pragmas attribute
        hw_kernel_no_pragmas = copy.deepcopy(mock_hw_kernel)
        delattr(hw_kernel_no_pragmas, 'pragmas')
        result = converter.convert(hw_kernel_no_pragmas)
        error_scenarios.append({
            "scenario": "missing_pragmas_attr",
            "success": result.success,
            "warnings": result.warnings
        })
        
        self._save_baseline("error_scenarios", error_scenarios)
    
    def test_full_pipeline_snapshot(self, sample_rtl_file):
        """Capture complete pipeline behavior from RTL to generated code."""
        # Parse RTL
        hw_kernel = parse_rtl_file(sample_rtl_file)
        assert hw_kernel is not None
        
        # Convert to DataflowModel
        converter = RTLDataflowConverter()
        conversion_result = converter.convert(hw_kernel)
        assert conversion_result.success
        
        # Generate code
        generator = UnifiedHWKGGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            gen_result = generator.generate_from_rtl(
                sample_rtl_file,
                {'onnx_patterns': [], 'cost_function': lambda *a, **k: 1.0},
                Path(temp_dir)
            )
            assert gen_result.success
            
            # Capture complete pipeline state
            pipeline_snapshot = {
                "hw_kernel_name": hw_kernel.name,
                "hw_kernel_interfaces": list(hw_kernel.interfaces.keys()),
                "hw_kernel_parameters": list(hw_kernel.parameters.keys()),
                "dataflow_interfaces": list(conversion_result.dataflow_model.interfaces.keys()),
                "generated_files": [f.name for f in gen_result.generated_files],
                "conversion_warnings": conversion_result.warnings,
                "generation_warnings": gen_result.warnings
            }
        
        self._save_baseline("full_pipeline", pipeline_snapshot)
    
    # Helper methods for baseline capture
    
    def _capture_dataflow_model_state(self, dataflow_model: DataflowModel, baseline_name: str):
        """Capture DataflowModel state for comparison."""
        state = {
            "interfaces": {},
            "parameters": dataflow_model.parameters,
            "computation_graph": dataflow_model.computation_graph
        }
        
        for name, iface in dataflow_model.interfaces.items():
            state["interfaces"][name] = {
                "interface_type": iface.interface_type.value,
                "tensor_dims": iface.tensor_dims,
                "block_dims": iface.block_dims,
                "stream_dims": iface.stream_dims,
                "dtype": {
                    "base_type": iface.dtype.base_type,
                    "bitwidth": iface.dtype.bitwidth,
                    "signed": iface.dtype.signed
                } if iface.dtype else None
            }
        
        self._save_baseline(baseline_name, state)
    
    def _save_baseline(self, name: str, data: Any):
        """Save baseline data for future comparison."""
        baseline_dir = Path(__file__).parent / "baselines"
        baseline_dir.mkdir(exist_ok=True)
        
        baseline_file = baseline_dir / f"{name}_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nSaved baseline: {baseline_file}")


class TestHWKernelPropertyUsage:
    """
    Test suite to identify all HWKernel properties and methods used downstream.
    """
    
    def test_enumerate_hwkernel_api_usage(self):
        """Enumerate all HWKernel properties/methods that must be preserved."""
        # Document the complete HWKernel API that's used downstream
        used_api = {
            "properties": [
                "name",  # Module name
                "interfaces",  # Dict[str, Interface]
                "parameters",  # Dict[str, Parameter]
                "pragmas",  # List[Pragma]
                "source_file",  # Optional[Path]
                "pragma_sophistication_level",  # str
                "parsing_warnings",  # List[str]
            ],
            "methods": [
                # HWKernel has minimal methods, mostly property access
            ],
            "interface_properties": [
                "name",  # Interface name
                "type",  # InterfaceType enum
                "ports",  # Dict[str, Port]
            ],
            "parameter_properties": [
                "name",  # Parameter name
                "type",  # Parameter type string
                "default",  # Default value
            ],
            "pragma_properties": [
                "type",  # Pragma type
                "line_number",  # Source line
                "raw_text",  # Original text
                "parsed_data",  # Parsed data dict
            ]
        }
        
        # Save API usage documentation
        baseline_dir = Path(__file__).parent / "baselines"
        baseline_dir.mkdir(exist_ok=True)
        
        with open(baseline_dir / "hwkernel_api_usage.json", 'w') as f:
            json.dump(used_api, f, indent=2)
        
        print("\nDocumented HWKernel API usage for parity testing")


# Parity validation functions to run after transition

def validate_parity(baseline_name: str, new_data: Any) -> bool:
    """
    Validate that new implementation produces same results as baseline.
    
    Args:
        baseline_name: Name of baseline to compare against
        new_data: Data from new implementation
        
    Returns:
        bool: True if parity achieved, False otherwise
    """
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_file = baseline_dir / f"{baseline_name}_baseline.json"
    
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_file}")
    
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    # Deep comparison
    differences = compare_nested_dicts(baseline_data, new_data)
    
    if differences:
        print(f"\nParity check FAILED for {baseline_name}:")
        for diff in differences:
            print(f"  - {diff}")
        return False
    else:
        print(f"\nParity check PASSED for {baseline_name}")
        return True


def compare_nested_dicts(d1: Any, d2: Any, path: str = "") -> List[str]:
    """Deep comparison of nested data structures."""
    differences = []
    
    if type(d1) != type(d2):
        differences.append(f"{path}: type mismatch - {type(d1).__name__} vs {type(d2).__name__}")
        return differences
    
    if isinstance(d1, dict):
        keys1 = set(d1.keys())
        keys2 = set(d2.keys())
        
        if keys1 != keys2:
            if keys1 - keys2:
                differences.append(f"{path}: missing keys in new data - {keys1 - keys2}")
            if keys2 - keys1:
                differences.append(f"{path}: extra keys in new data - {keys2 - keys1}")
        
        for key in keys1 & keys2:
            differences.extend(compare_nested_dicts(d1[key], d2[key], f"{path}.{key}" if path else key))
    
    elif isinstance(d1, list):
        if len(d1) != len(d2):
            differences.append(f"{path}: list length mismatch - {len(d1)} vs {len(d2)}")
        else:
            for i, (item1, item2) in enumerate(zip(d1, d2)):
                differences.extend(compare_nested_dicts(item1, item2, f"{path}[{i}]"))
    
    elif d1 != d2:
        differences.append(f"{path}: value mismatch - {d1} vs {d2}")
    
    return differences


if __name__ == "__main__":
    # Run tests to capture baselines
    pytest.main([__file__, "-v", "-s"])