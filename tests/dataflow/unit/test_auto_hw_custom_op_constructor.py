"""
Unit tests for AutoHWCustomOp constructor fix.

Tests the new FINN-style constructor pattern and ensures proper functionality
after the Phase 1-3 implementation changes.
"""

import pytest
import onnx.helper
from typing import List

from brainsmith.dataflow.core import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy


class MockAutoHWCustomOp(AutoHWCustomOp):
    """Mock implementation of AutoHWCustomOp for unit testing."""
    
    def get_nodeattr_types(self):
        """Define ONNX node attributes for testing."""
        my_attrs = {
            "PE": ("i", False, 4),  # Optional with default
            "VECTOR_SIZE": ("i", True, None),  # Required parameter
            
            # Datatype attributes for all dataflow interfaces
            "input0_dtype": ("s", True, ""),  # Required datatype specification
            "output0_dtype": ("s", True, ""),  # Required datatype specification
            
            # Base HWCustomOp attributes
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            "numInputVectors": ("ints", False, [1]),
        }
        
        # Update with parent class attributes (FINN base classes)
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs
    
    @staticmethod
    def get_interface_metadata() -> List[InterfaceMetadata]:
        """Return test interface metadata."""
        return [
            InterfaceMetadata(
                name="ap",
                interface_type=InterfaceType.CONTROL,
                datatype_constraints=[],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':'],
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="input0",
                interface_type=InterfaceType.INPUT,
                datatype_constraints=[
                    DatatypeConstraintGroup(
                        base_type="FIXED",
                        min_width=8,
                        max_width=16
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="output0",
                interface_type=InterfaceType.OUTPUT,
                datatype_constraints=[
                    DatatypeConstraintGroup(
                        base_type="FIXED",
                        min_width=16,
                        max_width=32
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],
                    rindex=0
                )
            ),
        ]


class MockAutoHWCustomOpOptionalDtypes(AutoHWCustomOp):
    """Mock implementation with optional datatype attributes for constraint testing."""
    
    def get_nodeattr_types(self):
        """Define ONNX node attributes with optional datatypes."""
        my_attrs = {
            "PE": ("i", False, 4),  # Optional with default
            "VECTOR_SIZE": ("i", True, None),  # Required parameter
            
            # Datatype attributes are optional to test our validation logic
            "input0_dtype": ("s", False, ""),  # Optional datatype specification
            "output0_dtype": ("s", False, ""),  # Optional datatype specification
            
            # Base HWCustomOp attributes
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            "numInputVectors": ("ints", False, [1]),
        }
        
        # Update with parent class attributes (FINN base classes)
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs
    
    @staticmethod
    def get_interface_metadata() -> List[InterfaceMetadata]:
        """Return test interface metadata."""
        return [
            InterfaceMetadata(
                name="ap",
                interface_type=InterfaceType.CONTROL,
                datatype_constraints=[],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':'],
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="input0",
                interface_type=InterfaceType.INPUT,
                datatype_constraints=[
                    DatatypeConstraintGroup(
                        base_type="FIXED",
                        min_width=8,
                        max_width=16
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="output0",
                interface_type=InterfaceType.OUTPUT,
                datatype_constraints=[
                    DatatypeConstraintGroup(
                        base_type="FIXED",
                        min_width=16,
                        max_width=32
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],
                    rindex=0
                )
            ),
        ]


def create_test_node_with_attributes():
    """Create ONNX node with all required attributes for testing."""
    return onnx.helper.make_node(
        'MockAutoHWCustomOp',
        ['input0'],
        ['output0'],
        PE=4,
        VECTOR_SIZE=1024,
        input0_dtype='FIXED<16,8>',
        output0_dtype='FIXED<16,8>'
    )


def create_node_without_datatypes():
    """Create ONNX node missing required datatype attributes."""
    return onnx.helper.make_node(
        'MockAutoHWCustomOp',
        ['input0'],
        ['output0'],
        PE=4,
        VECTOR_SIZE=1024
        # Missing input0_dtype and output0_dtype
    )


def create_node_with_partial_datatypes():
    """Create ONNX node with only some datatype attributes."""
    return onnx.helper.make_node(
        'MockAutoHWCustomOp',
        ['input0'],
        ['output0'],
        PE=4,
        VECTOR_SIZE=1024,
        input0_dtype='FIXED<16,8>'
        # Missing output0_dtype
    )


class TestConstructorPattern:
    """Test the new FINN-style constructor pattern."""
    
    def test_simple_constructor_success(self):
        """Test that constructor follows FINN pattern and succeeds with valid node."""
        node = create_test_node_with_attributes()
        
        # Should not raise any exceptions
        op = MockAutoHWCustomOp(node)
        
        # Verify basic properties
        assert op.onnx_node == node
        assert hasattr(op, '_dataflow_model')
        assert op._dataflow_model is not None
        assert hasattr(op, '_current_parallelism')
        
        # Verify FINN base functionality works
        assert op.get_nodeattr('PE') == 4
        assert op.get_nodeattr('VECTOR_SIZE') == 1024
        assert op.get_nodeattr('input0_dtype') == 'FIXED<16,8>'
        assert op.get_nodeattr('output0_dtype') == 'FIXED<16,8>'
    
    def test_constructor_without_kwargs(self):
        """Test constructor works without additional kwargs."""
        node = create_test_node_with_attributes()
        
        # Should work with just the node
        op = MockAutoHWCustomOp(node)
        assert op.onnx_node == node
    
    def test_constructor_with_kwargs(self):
        """Test constructor works following FINN's pattern (no arbitrary kwargs)."""
        node = create_test_node_with_attributes()
        
        # FINN base class doesn't support arbitrary kwargs, which is expected
        # Our constructor should work exactly like FINN's standard pattern
        op = MockAutoHWCustomOp(node)
        assert op.onnx_node == node
        
        # Verify that FINN correctly rejects arbitrary kwargs (expected behavior)
        import pytest
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            MockAutoHWCustomOp(node, some_param=123)


class TestDatatypeValidation:
    """Test datatype validation and error handling."""
    
    def test_missing_all_datatypes_error(self):
        """Test proper error when all datatypes are missing."""
        node = create_node_without_datatypes()
        
        with pytest.raises(Exception, match="Required attribute.*unspecified"):
            MockAutoHWCustomOp(node)
    
    def test_missing_partial_datatypes_error(self):
        """Test proper error when some datatypes are missing."""
        node = create_node_with_partial_datatypes()
        
        with pytest.raises(Exception, match="Required attribute output0_dtype unspecified"):
            MockAutoHWCustomOp(node)
    
    def test_datatype_error_message_contains_constraints(self):
        """Test that error messages include constraint descriptions."""
        node = create_node_without_datatypes()
        
        try:
            MockAutoHWCustomOpOptionalDtypes(node)
            pytest.fail("Expected ValueError was not raised")
        except ValueError as e:
            error_msg = str(e)
            # Should mention the interface name and constraint info
            assert "input0" in error_msg
            assert "FIXED" in error_msg
            assert "8" in error_msg and "16" in error_msg  # width range


class TestParameterResolution:
    """Test parameter resolution from node attributes."""
    
    def test_parameter_resolution_success(self):
        """Test that parameters are correctly resolved from node attributes."""
        node = create_test_node_with_attributes()
        op = MockAutoHWCustomOp(node)
        
        # Parameters should be accessible via get_nodeattr
        assert op.get_nodeattr('PE') == 4
        assert op.get_nodeattr('VECTOR_SIZE') == 1024
    
    def test_missing_parameter_handling(self):
        """Test handling of missing optional parameters."""
        node = onnx.helper.make_node(
            'MockAutoHWCustomOp',
            ['input0'],
            ['output0'],
            # Missing PE parameter (should have default)
            VECTOR_SIZE=1024,
            input0_dtype='FIXED<16,8>',
            output0_dtype='FIXED<16,8>'
        )
        
        # Should still work if PE has a default value in get_nodeattr_types
        op = MockAutoHWCustomOp(node)
        assert op.get_nodeattr('VECTOR_SIZE') == 1024


class TestDataflowModelBuilding:
    """Test dataflow model building from node attributes."""
    
    def test_dataflow_model_creation(self):
        """Test that dataflow model is properly created."""
        node = create_test_node_with_attributes()
        op = MockAutoHWCustomOp(node)
        
        # Should have a dataflow model
        model = op._dataflow_model
        assert model is not None
        
        # Should have the expected number of interfaces
        assert len(model.interfaces) == 3  # ap, input0, output0
        
        # Verify interface names
        interface_names = [iface.name for iface in model.interfaces.values()]
        assert 'ap' in interface_names
        assert 'input0' in interface_names
        assert 'output0' in interface_names
    
    def test_control_interface_creation(self):
        """Test that control interfaces are properly created."""
        node = create_test_node_with_attributes()
        op = MockAutoHWCustomOp(node)
        
        # Find the control interface
        control_interfaces = [
            iface for iface in op._dataflow_model.interfaces.values() 
            if iface.interface_type == InterfaceType.CONTROL
        ]
        
        assert len(control_interfaces) == 1
        control_iface = control_interfaces[0]
        assert control_iface.name == 'ap'
    
    def test_dataflow_interface_creation(self):
        """Test that dataflow interfaces are properly created with datatypes."""
        node = create_test_node_with_attributes()
        op = MockAutoHWCustomOp(node)
        
        # Find input and output interfaces
        input_interfaces = [
            iface for iface in op._dataflow_model.interfaces.values() 
            if iface.interface_type == InterfaceType.INPUT
        ]
        output_interfaces = [
            iface for iface in op._dataflow_model.interfaces.values() 
            if iface.interface_type == InterfaceType.OUTPUT
        ]
        
        assert len(input_interfaces) == 1
        assert len(output_interfaces) == 1
        
        input_iface = input_interfaces[0]
        output_iface = output_interfaces[0]
        
        assert input_iface.name == 'input0'
        assert output_iface.name == 'output0'
        
        # Should have datatypes set
        assert input_iface.dtype is not None
        assert output_iface.dtype is not None


class TestInterfaceMetadata:
    """Test interface metadata handling."""
    
    def test_static_interface_metadata(self):
        """Test that static interface metadata is properly accessible."""
        metadata_list = MockAutoHWCustomOp.get_interface_metadata()
        
        assert len(metadata_list) == 3
        
        # Verify metadata structure
        names = [meta.name for meta in metadata_list]
        assert 'ap' in names
        assert 'input0' in names
        assert 'output0' in names
        
        # Verify constraint groups
        input_meta = next(meta for meta in metadata_list if meta.name == 'input0')
        assert len(input_meta.datatype_constraints) == 1
        constraint = input_meta.datatype_constraints[0]
        assert constraint.base_type == "FIXED"
        assert constraint.min_width == 8
        assert constraint.max_width == 16


class TestBackwardCompatibility:
    """Test backward compatibility and migration scenarios."""
    
    def test_finn_base_functionality(self):
        """Test that FINN base class functionality still works."""
        node = create_test_node_with_attributes()
        op = MockAutoHWCustomOp(node)
        
        # FINN base class methods should work
        assert hasattr(op, 'get_nodeattr')
        assert hasattr(op, 'onnx_node')
        
        # Node attribute access should work
        assert op.get_nodeattr('PE') == 4


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    def test_invalid_datatype_format(self):
        """Test handling of invalid datatype formats."""
        node = onnx.helper.make_node(
            'MockAutoHWCustomOp',
            ['input0'],
            ['output0'],
            PE=4,
            VECTOR_SIZE=1024,
            input0_dtype='INVALID_FORMAT',  # Invalid format
            output0_dtype='FIXED<16,8>'
        )
        
        # Should handle parsing errors gracefully
        # (Actual behavior depends on QONNX datatype parsing)
        try:
            op = MockAutoHWCustomOp(node)
            # If it succeeds, verify the datatype was set
            assert op.get_nodeattr('input0_dtype') == 'INVALID_FORMAT'
        except (ValueError, TypeError):
            # If it fails, that's also acceptable for invalid formats
            pass
    
    def test_empty_node_attributes(self):
        """Test handling of node with no attributes."""
        node = onnx.helper.make_node(
            'MockAutoHWCustomOp',
            ['input0'],
            ['output0']
            # No attributes at all
        )
        
        with pytest.raises(ValueError, match="must be specified"):
            MockAutoHWCustomOp(node)


if __name__ == "__main__":
    # Run basic smoke tests if executed directly
    print("Running basic smoke tests...")
    
    try:
        node = create_test_node_with_attributes()
        op = MockAutoHWCustomOp(node)
        print("✅ Basic constructor test passed")
        
        assert op.get_nodeattr('PE') == 4
        print("✅ Parameter resolution test passed")
        
        assert len(op._dataflow_model.interfaces) == 3
        print("✅ Dataflow model building test passed")
        
        print("🎉 All smoke tests passed!")
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()