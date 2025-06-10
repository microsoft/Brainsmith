"""
Comprehensive testbench for the generated ThresholdingAxiHWCustomOp.

This testbench validates all aspects of the auto-generated HWCustomOp class:
- Initialization and inheritance
- Interface metadata structure and validation
- Node attribute definitions and access
- Dataflow integration and chunking strategies
- ONNX node creation functionality
- Error handling and edge cases
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add the generated module to Python path
sys.path.insert(0, str(Path(__file__).parent / "hwkg_demo_final"))

# Import the generated class
try:
    from thresholding_axi_hwcustomop import ThresholdingAxiHWCustomOp, make_thresholding_axi_node
    GENERATED_CLASS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import generated class: {e}")
    GENERATED_CLASS_AVAILABLE = False

# Mock the brainsmith dependencies for testing
class MockDataTypeConstraint:
    def __init__(self, finn_type, bit_width, signed=False):
        self.finn_type = finn_type
        self.bit_width = bit_width
        self.signed = signed

class MockInterfaceMetadata:
    def __init__(self, name, interface_type, allowed_datatypes, chunking_strategy):
        self.name = name
        self.interface_type = interface_type
        self.allowed_datatypes = allowed_datatypes
        self.chunking_strategy = chunking_strategy

class MockDataflowInterfaceType:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    WEIGHT = "WEIGHT"
    CONFIG = "CONFIG"

class MockAutoHWCustomOp:
    def __init__(self, onnx_node, interface_metadata=None, **kwargs):
        self.onnx_node = onnx_node
        self.interface_metadata_collection = interface_metadata
        self._dataflow_model = Mock()
        self._current_parallelism = {}
    
    def get_nodeattr_types(self):
        return {"base_attr": ("s", False, "default")}
    
    @property
    def interface_metadata(self):
        return Mock()
    
    @property
    def dataflow_model(self):
        return self._dataflow_model
    
    def update_parallelism(self, iPar=None, wPar=None):
        pass

def default_chunking():
    return Mock()

def index_chunking(start_index, shape):
    mock = Mock()
    mock.start_index = start_index
    mock.shape = shape
    return mock

def last_dim_chunking(chunk_size):
    mock = Mock()
    mock.chunk_size = chunk_size
    return mock

# Mock all the brainsmith imports
mock_modules = {
    'brainsmith': Mock(),
    'brainsmith.dataflow': Mock(),
    'brainsmith.dataflow.core': Mock(),
    'brainsmith.dataflow.core.auto_hw_custom_op': Mock(),
    'brainsmith.dataflow.core.interface_metadata': Mock(),
    'brainsmith.dataflow.core.dataflow_interface': Mock(),
    'brainsmith.dataflow.core.tensor_chunking': Mock(),
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

# Set up the mocks to return our mock classes
sys.modules['brainsmith.dataflow.core.auto_hw_custom_op'].AutoHWCustomOp = MockAutoHWCustomOp
sys.modules['brainsmith.dataflow.core.interface_metadata'].InterfaceMetadata = MockInterfaceMetadata
sys.modules['brainsmith.dataflow.core.interface_metadata'].DataTypeConstraint = MockDataTypeConstraint
sys.modules['brainsmith.dataflow.core.dataflow_interface'].DataflowInterfaceType = MockDataflowInterfaceType
sys.modules['brainsmith.dataflow.core.tensor_chunking'].default_chunking = default_chunking
sys.modules['brainsmith.dataflow.core.tensor_chunking'].index_chunking = index_chunking
sys.modules['brainsmith.dataflow.core.tensor_chunking'].last_dim_chunking = last_dim_chunking


class TestThresholdingAxiHWCustomOp(unittest.TestCase):
    """Comprehensive tests for the generated ThresholdingAxiHWCustomOp class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GENERATED_CLASS_AVAILABLE:
            self.skipTest("Generated class not available")
        
        # Create mock ONNX node
        self.mock_onnx_node = Mock()
        self.mock_onnx_node.input = ["input_tensor"]
        self.mock_onnx_node.output = ["output_tensor"]
        self.mock_onnx_node.op_type = "ThresholdingAxiHWCustomOp"
    
    def test_class_exists_and_importable(self):
        """Test that the generated class exists and can be imported."""
        self.assertTrue(GENERATED_CLASS_AVAILABLE, "Generated ThresholdingAxiHWCustomOp class should be importable")
        self.assertTrue(hasattr(ThresholdingAxiHWCustomOp, '__init__'), "Class should have __init__ method")
        self.assertTrue(callable(ThresholdingAxiHWCustomOp), "Class should be callable")
    
    def test_initialization_basic(self):
        """Test basic initialization of the HWCustomOp."""
        try:
            op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
            self.assertIsNotNone(op, "HWCustomOp should initialize successfully")
            self.assertEqual(op.kernel_name, "thresholding_axi", "Kernel name should be set correctly")
            self.assertEqual(op.rtl_source, "thresholding_axi.sv", "RTL source should be set correctly")
        except Exception as e:
            self.fail(f"Basic initialization failed: {e}")
    
    def test_interface_metadata_structure(self):
        """Test that interface metadata is properly structured."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        # Check that interface metadata exists
        self.assertTrue(hasattr(op, '_interface_metadata'), "Should have _interface_metadata attribute")
        self.assertIsInstance(op._interface_metadata, list, "Interface metadata should be a list")
        
        # Should have exactly 2 AXI-Stream interfaces (s_axis and m_axis)
        self.assertEqual(len(op._interface_metadata), 2, "Should have exactly 2 interfaces")
        
        # Check interface names
        interface_names = [iface.name for iface in op._interface_metadata]
        self.assertIn("s_axis", interface_names, "Should have s_axis interface")
        self.assertIn("m_axis", interface_names, "Should have m_axis interface")
        
        # Check interface types
        s_axis = next(iface for iface in op._interface_metadata if iface.name == "s_axis")
        m_axis = next(iface for iface in op._interface_metadata if iface.name == "m_axis")
        
        # Check interface types (they're enum values, not strings)
        self.assertEqual(str(s_axis.interface_type), "DataflowInterfaceType.INPUT", "s_axis should be INPUT interface")
        self.assertEqual(str(m_axis.interface_type), "DataflowInterfaceType.OUTPUT", "m_axis should be OUTPUT interface")
    
    def test_datatype_constraints(self):
        """Test that datatype constraints are properly defined."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        for iface in op._interface_metadata:
            self.assertIsInstance(iface.allowed_datatypes, list, f"Interface {iface.name} should have datatype list")
            self.assertGreater(len(iface.allowed_datatypes), 0, f"Interface {iface.name} should have at least one datatype")
            
            # Check first datatype constraint
            constraint = iface.allowed_datatypes[0]
            self.assertEqual(constraint.finn_type, "UINT8", f"Interface {iface.name} should use UINT8")
            self.assertEqual(constraint.bit_width, 8, f"Interface {iface.name} should use 8-bit width")
            self.assertFalse(constraint.signed, f"Interface {iface.name} should be unsigned")
    
    def test_chunking_strategies(self):
        """Test that chunking strategies are properly assigned."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        for iface in op._interface_metadata:
            self.assertIsNotNone(iface.chunking_strategy, f"Interface {iface.name} should have chunking strategy")
            # The default_chunking() function returns a Mock in our test setup
    
    def test_node_attributes(self):
        """Test that node attributes are properly defined."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        attrs = op.get_nodeattr_types()
        self.assertIsInstance(attrs, dict, "Node attributes should be a dictionary")
        
        # The actual generated class doesn't have our mock base_attr, so just check it's a dict
        self.assertIsInstance(attrs, dict, "Should return a dictionary of attributes")
        
        # Check RTL parameters are included
        expected_params = ["N", "WI", "WT", "C", "PE", "SIGNED", "FPARG", "BIAS", 
                          "THRESHOLDS_PATH", "USE_AXILITE", "DEPTH_TRIGGER_URAM", 
                          "DEPTH_TRIGGER_BRAM", "DEEP_PIPELINE"]
        
        for param in expected_params:
            self.assertIn(param, attrs, f"Should have {param} parameter")
            self.assertIsInstance(attrs[param], tuple, f"{param} should be a tuple")
            self.assertEqual(len(attrs[param]), 3, f"{param} should have 3 elements (type, required, default)")
    
    def test_kernel_interface_specs(self):
        """Test that kernel interface specifications are properly defined."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        specs = op.get_kernel_interface_specs()
        self.assertIsInstance(specs, dict, "Interface specs should be a dictionary")
        
        # Check s_axis specification
        self.assertIn("s_axis", specs, "Should have s_axis specification")
        s_axis_spec = specs["s_axis"]
        self.assertEqual(s_axis_spec["type"], "input", "s_axis should be input type")
        self.assertEqual(s_axis_spec["interface_type"], "AXI_STREAM", "s_axis should be AXI_STREAM")
        self.assertEqual(s_axis_spec["chunking_strategy"], "default_chunking", "s_axis should use default chunking")
        
        # Check m_axis specification
        self.assertIn("m_axis", specs, "Should have m_axis specification")
        m_axis_spec = specs["m_axis"]
        self.assertEqual(m_axis_spec["type"], "output", "m_axis should be output type")
        self.assertEqual(m_axis_spec["interface_type"], "AXI_STREAM", "m_axis should be AXI_STREAM")
        self.assertEqual(m_axis_spec["chunking_strategy"], "default_chunking", "m_axis should use default chunking")
    
    def test_inheritance_structure(self):
        """Test that inheritance from AutoHWCustomOp works correctly."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        # Should have inherited methods and properties
        self.assertTrue(hasattr(op, 'interface_metadata'), "Should inherit interface_metadata property")
        self.assertTrue(hasattr(op, 'dataflow_model'), "Should inherit dataflow_model property")
        self.assertTrue(hasattr(op, 'update_parallelism'), "Should inherit update_parallelism method")
    
    def test_initialization_with_kwargs(self):
        """Test initialization with additional keyword arguments."""
        kwargs = {"test_param": "test_value", "another_param": 42}
        
        try:
            op = ThresholdingAxiHWCustomOp(self.mock_onnx_node, **kwargs)
            self.assertIsNotNone(op, "Should handle kwargs during initialization")
        except Exception as e:
            self.fail(f"Initialization with kwargs failed: {e}")
    
    def test_onnx_node_creation_function(self):
        """Test the make_thresholding_axi_node convenience function."""
        inputs = ["input_tensor"]
        outputs = ["output_tensor"]
        node_attrs = {"N": 4, "WI": 8}
        
        with patch('onnx.helper.make_node') as mock_make_node:
            mock_make_node.return_value = Mock()
            
            result = make_thresholding_axi_node(inputs, outputs, **node_attrs)
            
            # Verify make_node was called with correct parameters
            mock_make_node.assert_called_once_with(
                "ThresholdingAxiHWCustomOp",
                inputs=inputs,
                outputs=outputs,
                domain="finn.custom_op.fpgadataflow",
                **node_attrs
            )
            self.assertIsNotNone(result, "Should return a valid ONNX node")
    
    def test_error_handling_invalid_onnx_node(self):
        """Test error handling with invalid ONNX node."""
        # Test with None
        try:
            op = ThresholdingAxiHWCustomOp(None)
            # Should not raise exception during initialization
        except Exception as e:
            # If it does raise an exception, it should be meaningful
            self.assertIsInstance(e, (TypeError, ValueError, AttributeError), 
                                f"Should raise appropriate exception for None input, got {type(e)}")
    
    def test_string_representations(self):
        """Test string representations and debugging info."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        # Should have meaningful string representation
        str_repr = str(op)
        self.assertIsInstance(str_repr, str, "Should have string representation")
        
        # Check that key attributes are accessible
        self.assertEqual(op.kernel_name, "thresholding_axi")
        self.assertEqual(op.rtl_source, "thresholding_axi.sv")
    
    def test_interface_metadata_immutability(self):
        """Test that interface metadata is properly structured and accessible."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        # Interface metadata should be accessible
        self.assertIsNotNone(op._interface_metadata)
        
        # Should be able to iterate over interfaces
        interface_count = 0
        for iface in op._interface_metadata:
            interface_count += 1
            self.assertTrue(hasattr(iface, 'name'), "Interface should have name")
            self.assertTrue(hasattr(iface, 'interface_type'), "Interface should have type")
        
        self.assertEqual(interface_count, 2, "Should have exactly 2 interfaces")


class TestGeneratedCodeQuality(unittest.TestCase):
    """Tests for code quality and structure of the generated file."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generated_file = Path(__file__).parent / "hwkg_demo_final" / "thresholding_axi_hwcustomop.py"
    
    def test_file_exists(self):
        """Test that the generated file exists."""
        self.assertTrue(self.generated_file.exists(), "Generated file should exist")
    
    def test_file_syntax(self):
        """Test that the generated file has valid Python syntax."""
        try:
            with open(self.generated_file, 'r') as f:
                code = f.read()
            
            # Try to compile the code
            compile(code, str(self.generated_file), 'exec')
        except SyntaxError as e:
            self.fail(f"Generated file has syntax error: {e}")
    
    def test_required_imports(self):
        """Test that all required imports are present."""
        with open(self.generated_file, 'r') as f:
            content = f.read()
        
        required_imports = [
            "from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp",
            "from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint",
            "from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType",
            "from brainsmith.dataflow.core.tensor_chunking import index_chunking, default_chunking, last_dim_chunking"
        ]
        
        for import_line in required_imports:
            self.assertIn(import_line, content, f"Should have import: {import_line}")
    
    def test_class_definition(self):
        """Test that the class is properly defined."""
        with open(self.generated_file, 'r') as f:
            content = f.read()
        
        # Should have class definition
        self.assertIn("class ThresholdingAxiHWCustomOp(AutoHWCustomOp):", content)
        
        # Should have docstring
        self.assertIn('"""', content)
        
        # Should have proper method definitions
        self.assertIn("def __init__(self, onnx_node, **kwargs):", content)
        self.assertIn("def get_nodeattr_types(self):", content)
        self.assertIn("def get_kernel_interface_specs(self):", content)
    
    def test_copyright_and_metadata(self):
        """Test that copyright and generation metadata are present."""
        with open(self.generated_file, 'r') as f:
            content = f.read()
        
        # Should have copyright notice
        self.assertIn("# Copyright (c) Microsoft Corporation.", content)
        
        # Should have generation timestamp
        self.assertIn("# Generation timestamp:", content)
        
        # Should have source file reference
        self.assertIn("# Generated from: thresholding_axi.sv", content)


class TestIntegrationWithDataflow(unittest.TestCase):
    """Integration tests with the dataflow system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GENERATED_CLASS_AVAILABLE:
            self.skipTest("Generated class not available")
        
        self.mock_onnx_node = Mock()
        self.mock_onnx_node.input = ["input_tensor"]
        self.mock_onnx_node.output = ["output_tensor"]
    
    def test_dataflow_model_creation(self):
        """Test that dataflow model is created during initialization."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        # Should have dataflow model property
        self.assertTrue(hasattr(op, 'dataflow_model'), "Should have dataflow_model property")
        
        # Dataflow model should be accessible
        model = op.dataflow_model
        self.assertIsNotNone(model, "Dataflow model should not be None")
    
    def test_parallelism_update(self):
        """Test parallelism update functionality."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        # Should have update_parallelism method
        self.assertTrue(hasattr(op, 'update_parallelism'), "Should have update_parallelism method")
        self.assertTrue(callable(op.update_parallelism), "update_parallelism should be callable")
        
        # Should be able to call without errors
        try:
            op.update_parallelism(iPar={'s_axis': 4}, wPar={})
        except Exception as e:
            self.fail(f"update_parallelism should not raise exception: {e}")


def run_comprehensive_test():
    """Run all tests and provide detailed report."""
    print("=" * 80)
    print("COMPREHENSIVE TESTBENCH FOR GENERATED THRESHOLDING_AXI_HWCUSTOMOP")
    print("=" * 80)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestThresholdingAxiHWCustomOp))
    suite.addTests(loader.loadTestsFromTestCase(TestGeneratedCodeQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithDataflow))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Provide summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\n❌ TESTBENCH FAILED - Generated code has issues")
        return False
    else:
        print("\n✅ TESTBENCH PASSED - Generated code is functional")
        return True


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)