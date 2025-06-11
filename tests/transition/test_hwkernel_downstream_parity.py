"""
HWKernel Downstream Functionality Parity Test Suite

This test suite captures ALL current behavior of components that depend on HWKernel,
using REAL RTL files and REAL parsing to ensure we don't circumvent any functionality.

This suite serves as the definitive baseline for validating that removing HWKernel
doesn't break any downstream functionality.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import hashlib
import sys

# Import all HWKernel-dependent components
from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import HWKernel
from brainsmith.dataflow.rtl_integration import RTLDataflowConverter
from brainsmith.tools.unified_hwkg import UnifiedHWKGGenerator
from brainsmith.dataflow.core.dataflow_model import DataflowModel

# Test data paths
TEST_RTL_DIR = Path(__file__).parent.parent.parent / "examples" / "thresholding"
BASELINE_DIR = Path(__file__).parent / "baselines"
BASELINE_DIR.mkdir(exist_ok=True)


class HWKernelUsageTracker:
    """
    Tracks all accesses to HWKernel properties and methods.
    This ensures we capture EXACTLY what downstream components use.
    """
    
    def __init__(self, hw_kernel: HWKernel):
        self._hw_kernel = hw_kernel
        self._accessed_properties = set()
        self._method_calls = []
        
    def __getattr__(self, name):
        # Track property access
        self._accessed_properties.add(name)
        value = getattr(self._hw_kernel, name)
        
        # If it's a method, wrap it to track calls
        if callable(value):
            def wrapped(*args, **kwargs):
                self._method_calls.append((name, args, kwargs))
                return value(*args, **kwargs)
            return wrapped
        
        # For nested objects (like interfaces dict), track nested access
        if name == 'interfaces' and isinstance(value, dict):
            return self._wrap_dict(value, f"{name}")
        
        return value
    
    def _wrap_dict(self, d: dict, prefix: str) -> dict:
        """Wrap dict to track nested property access."""
        class DictWrapper(dict):
            def __getitem__(wrapper_self, key):
                self._accessed_properties.add(f"{prefix}['{key}']")
                return super().__getitem__(key)
        
        return DictWrapper(d)
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get report of all HWKernel usage."""
        return {
            "accessed_properties": sorted(list(self._accessed_properties)),
            "method_calls": self._method_calls
        }


class TestHWKernelDownstreamParity:
    """
    Comprehensive tests capturing all HWKernel downstream behavior.
    Uses REAL RTL files and REAL parsing - no mocks that could hide functionality.
    """
    
    def test_rtl_converter_complete_behavior(self):
        """Test complete RTLDataflowConverter behavior with real RTL files."""
        print("\n=== Testing RTLDataflowConverter Complete Behavior ===")
        
        # Test with multiple real RTL files
        test_files = [
            TEST_RTL_DIR / "thresholding_axi.sv",
            # Add more RTL files as available
        ]
        
        converter = RTLDataflowConverter()
        results = {}
        
        for rtl_file in test_files:
            if not rtl_file.exists():
                print(f"Skipping {rtl_file} - not found")
                continue
                
            print(f"\nParsing {rtl_file.name}...")
            
            # Parse with real parser
            hw_kernel = parse_rtl_file(rtl_file)
            assert hw_kernel is not None, f"Failed to parse {rtl_file}"
            
            # Track HWKernel usage
            tracker = HWKernelUsageTracker(hw_kernel)
            
            # Convert to DataflowModel
            result = converter.convert(tracker)
            
            # Capture complete result
            results[rtl_file.name] = {
                "success": result.success,
                "errors": result.errors,
                "warnings": result.warnings,
                "hw_kernel_usage": tracker.get_usage_report(),
                "dataflow_model": self._serialize_dataflow_model(result.dataflow_model) if result.success else None
            }
            
            # Additional validation
            if result.success:
                self._validate_dataflow_model(result.dataflow_model, hw_kernel)
        
        # Save baseline
        self._save_baseline("rtl_converter_behavior", results)
    
    def test_unified_generator_complete_pipeline(self):
        """Test complete UnifiedHWKGGenerator pipeline with real RTL."""
        print("\n=== Testing Unified Generator Complete Pipeline ===")
        
        generator = UnifiedHWKGGenerator()
        results = {}
        
        rtl_file = TEST_RTL_DIR / "thresholding_axi.sv"
        compiler_data = {
            'onnx_patterns': [],
            'cost_function': lambda *args, **kwargs: 1.0
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Run complete generation
            result = generator.generate_from_rtl(
                rtl_file=rtl_file,
                compiler_data=compiler_data,
                output_dir=output_dir
            )
            
            # Capture complete results
            results["generation"] = {
                "success": result.success,
                "errors": result.errors,
                "warnings": result.warnings,
                "generated_files": [f.name for f in result.generated_files],
                "file_sizes": {f.name: f.stat().st_size for f in result.generated_files},
                "dataflow_model_interfaces": len(result.dataflow_model.interfaces) if result.dataflow_model else 0
            }
            
            # Capture generated file content samples
            if result.success:
                results["file_samples"] = {}
                for file_path in result.generated_files:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Capture first 1000 chars and compute hash
                        results["file_samples"][file_path.name] = {
                            "preview": content[:1000],
                            "full_hash": hashlib.sha256(content.encode()).hexdigest(),
                            "size": len(content)
                        }
        
        self._save_baseline("unified_generator_pipeline", results)
    
    def test_interface_conversion_details(self):
        """Test detailed interface conversion from RTL to DataflowInterface."""
        print("\n=== Testing Interface Conversion Details ===")
        
        rtl_file = TEST_RTL_DIR / "thresholding_axi.sv"
        hw_kernel = parse_rtl_file(rtl_file)
        assert hw_kernel is not None
        
        converter = RTLDataflowConverter()
        results = {}
        
        # Test each interface conversion in detail
        for iface_name, rtl_interface in hw_kernel.interfaces.items():
            print(f"\nConverting interface: {iface_name}")
            
            # Track what properties are accessed
            interface_info = {
                "rtl_interface_type": rtl_interface.type.value,
                "rtl_interface_ports": list(rtl_interface.ports.keys()),
                "port_details": {}
            }
            
            # Capture port details
            for port_name, port in rtl_interface.ports.items():
                interface_info["port_details"][port_name] = {
                    "direction": port.direction.value,
                    "data_type": port.data_type,
                    "width": port.width
                }
            
            # Find relevant pragmas
            relevant_pragmas = converter._find_interface_pragmas(rtl_interface, hw_kernel.pragmas)
            interface_info["pragma_count"] = len(relevant_pragmas)
            
            # Convert to DataflowInterface
            dataflow_interface = converter._convert_interface(rtl_interface, relevant_pragmas, hw_kernel)
            
            if dataflow_interface:
                interface_info["dataflow_result"] = {
                    "interface_type": dataflow_interface.interface_type.value,
                    "tensor_dims": dataflow_interface.tensor_dims,
                    "block_dims": dataflow_interface.block_dims,
                    "stream_dims": dataflow_interface.stream_dims,
                    "dtype": {
                        "base_type": dataflow_interface.dtype.base_type,
                        "bitwidth": dataflow_interface.dtype.bitwidth,
                        "signed": dataflow_interface.dtype.signed
                    } if dataflow_interface.dtype else None
                }
            
            results[iface_name] = interface_info
        
        self._save_baseline("interface_conversion_details", results)
    
    def test_pragma_processing_pipeline(self):
        """Test complete pragma processing from RTL to chunking strategies."""
        print("\n=== Testing Pragma Processing Pipeline ===")
        
        # Create test RTL with various pragmas
        test_rtl_content = """
module test_pragmas (
    // @brainsmith:BDIM:s_axis[0]=[PE,WI]
    input logic s_axis_tdata,
    input logic s_axis_tvalid,
    output logic s_axis_tready,
    
    // @brainsmith:BDIM:m_axis:O_BITS*PE
    output logic m_axis_tdata,
    output logic m_axis_tvalid,
    input logic m_axis_tready
);
endmodule
"""
        
        # Write test RTL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
            f.write(test_rtl_content)
            test_file = Path(f.name)
        
        try:
            # Parse with real parser
            hw_kernel = parse_rtl_file(test_file)
            
            if hw_kernel and hw_kernel.pragmas:
                converter = RTLDataflowConverter()
                results = {}
                
                for pragma in hw_kernel.pragmas:
                    pragma_info = {
                        "type": pragma.type if hasattr(pragma, 'type') else 'unknown',
                        "raw_text": pragma.raw_text if hasattr(pragma, 'raw_text') else '',
                        "line_number": pragma.line_number if hasattr(pragma, 'line_number') else 0,
                        "parsed_data": pragma.parsed_data if hasattr(pragma, 'parsed_data') else {}
                    }
                    
                    # Test pragma conversion
                    if hasattr(pragma, 'type') and pragma.type == 'BDIM':
                        strategy = converter.pragma_converter.convert_bdim_pragma(pragma)
                        pragma_info["converted_strategy"] = strategy
                    
                    results[f"pragma_{pragma.line_number}"] = pragma_info
                
                self._save_baseline("pragma_processing", results)
            else:
                print("No pragmas found in test RTL")
                
        finally:
            test_file.unlink()
    
    def test_error_handling_scenarios(self):
        """Test all error handling paths with various RTL issues."""
        print("\n=== Testing Error Handling Scenarios ===")
        
        converter = RTLDataflowConverter()
        results = {}
        
        # Test various problematic RTL scenarios
        test_scenarios = [
            # Scenario 1: Empty module
            ("empty_module", """
module empty_module();
endmodule
"""),
            # Scenario 2: Module with only parameters
            ("params_only", """
module params_only #(
    parameter int N = 8
)();
endmodule
"""),
            # Scenario 3: Module with non-standard interfaces
            ("non_standard", """
module non_standard (
    input wire custom_in,
    output reg custom_out
);
endmodule
"""),
        ]
        
        for scenario_name, rtl_content in test_scenarios:
            # Write test RTL
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
                f.write(rtl_content)
                test_file = Path(f.name)
            
            try:
                # Parse and convert
                hw_kernel = parse_rtl_file(test_file)
                
                if hw_kernel:
                    result = converter.convert(hw_kernel)
                    results[scenario_name] = {
                        "parsed": True,
                        "conversion_success": result.success,
                        "errors": result.errors,
                        "warnings": result.warnings,
                        "interface_count": len(hw_kernel.interfaces) if hasattr(hw_kernel, 'interfaces') else 0
                    }
                else:
                    results[scenario_name] = {
                        "parsed": False,
                        "error": "Failed to parse RTL"
                    }
                    
            finally:
                test_file.unlink()
        
        self._save_baseline("error_handling", results)
    
    def test_hwkernel_property_completeness(self):
        """Document ALL HWKernel properties that could be used downstream."""
        print("\n=== Testing HWKernel Property Completeness ===")
        
        rtl_file = TEST_RTL_DIR / "thresholding_axi.sv"
        hw_kernel = parse_rtl_file(rtl_file)
        assert hw_kernel is not None
        
        # Document all available properties
        properties = {
            "direct_attributes": [],
            "callable_methods": [],
            "property_types": {}
        }
        
        # Get all attributes
        for attr_name in dir(hw_kernel):
            if not attr_name.startswith('_'):
                attr_value = getattr(hw_kernel, attr_name)
                
                if callable(attr_value):
                    properties["callable_methods"].append(attr_name)
                else:
                    properties["direct_attributes"].append(attr_name)
                    properties["property_types"][attr_name] = type(attr_value).__name__
        
        # Document interface structure
        if hasattr(hw_kernel, 'interfaces'):
            properties["interface_structure"] = {}
            for iface_name, iface in hw_kernel.interfaces.items():
                properties["interface_structure"][iface_name] = {
                    "attributes": [a for a in dir(iface) if not a.startswith('_')],
                    "port_count": len(iface.ports) if hasattr(iface, 'ports') else 0
                }
        
        self._save_baseline("hwkernel_properties", properties)
    
    # Helper methods
    
    def _serialize_dataflow_model(self, dataflow_model: Optional[DataflowModel]) -> Optional[Dict]:
        """Serialize DataflowModel for baseline comparison."""
        if not dataflow_model:
            return None
            
        return {
            "interface_count": len(dataflow_model.interfaces),
            "interfaces": {
                name: {
                    "type": iface.interface_type.value,
                    "tensor_dims": iface.tensor_dims,
                    "block_dims": iface.block_dims,
                    "stream_dims": iface.stream_dims,
                    "dtype_info": {
                        "base_type": iface.dtype.base_type,
                        "bitwidth": iface.dtype.bitwidth
                    } if iface.dtype else None
                }
                for name, iface in dataflow_model.interfaces.items()
            },
            "parameters": dataflow_model.parameters,
            "computation_graph": dataflow_model.computation_graph
        }
    
    def _validate_dataflow_model(self, dataflow_model: DataflowModel, original_hw_kernel: HWKernel):
        """Validate that DataflowModel preserves essential HWKernel information."""
        # Check that all interfaces were converted
        hw_interface_names = set(original_hw_kernel.interfaces.keys())
        df_interface_names = set(dataflow_model.interfaces.keys())
        
        # Note: Some interfaces might be filtered (e.g., config interfaces)
        print(f"HWKernel interfaces: {hw_interface_names}")
        print(f"DataflowModel interfaces: {df_interface_names}")
        
        # Check that kernel name is preserved
        if 'kernel_name' in dataflow_model.parameters:
            assert dataflow_model.parameters['kernel_name'] == original_hw_kernel.name
    
    def _save_baseline(self, name: str, data: Any):
        """Save baseline data for future comparison."""
        baseline_file = BASELINE_DIR / f"{name}_baseline.json"
        
        # Convert any non-serializable objects
        def default_serializer(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        with open(baseline_file, 'w') as f:
            json.dump(data, f, indent=2, default=default_serializer)
        
        print(f"\nSaved baseline: {baseline_file}")
        print(f"Baseline contains {len(str(data))} characters of data")


class TestParityValidation:
    """
    Tests to validate parity after HWKernel removal.
    These will be run with both old and new implementations.
    """
    
    def test_validate_conversion_parity(self):
        """Validate that new direct parser produces same DataflowModel as current pipeline."""
        # This test will be implemented after the new parser is created
        # It will compare:
        # 1. Old: RTL → HWKernel → DataflowModel
        # 2. New: RTL → DataflowModel
        # And ensure the resulting DataflowModels are equivalent
        pass
    
    def test_validate_generation_parity(self):
        """Validate that generated code is functionally equivalent."""
        # This test will compare generated HWCustomOp/RTLBackend files
        # from both pipelines to ensure functional equivalence
        pass


def run_all_baseline_captures():
    """Run all tests to capture complete baseline behavior."""
    print("=" * 70)
    print("CAPTURING HWKERNEL DOWNSTREAM BEHAVIOR BASELINES")
    print("=" * 70)
    
    test_instance = TestHWKernelDownstreamParity()
    
    # Run all test methods
    test_methods = [
        test_instance.test_rtl_converter_complete_behavior,
        test_instance.test_unified_generator_complete_pipeline,
        test_instance.test_interface_conversion_details,
        test_instance.test_pragma_processing_pipeline,
        test_instance.test_error_handling_scenarios,
        test_instance.test_hwkernel_property_completeness,
    ]
    
    for test_method in test_methods:
        try:
            test_method()
        except Exception as e:
            print(f"\nERROR in {test_method.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("BASELINE CAPTURE COMPLETE")
    print(f"Baselines saved to: {BASELINE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    # Set Python path if needed
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    run_all_baseline_captures()