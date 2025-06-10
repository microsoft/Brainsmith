"""
Unit tests for DataflowInterface class.

Tests cover interface creation, validation, property access, constraint checking,
and all core functionality based on current implementation.
"""

import pytest
import numpy as np
from brainsmith.dataflow.core.dataflow_interface import (
    DataflowInterface,
    DataflowInterfaceType,
    DataflowDataType,
    DataTypeConstraint,
    ConstraintType,
    DivisibilityConstraint,
    RangeConstraint
)
from brainsmith.dataflow.core.validation import ValidationSeverity


@pytest.mark.unit
class TestDataflowDataType:
    """Test DataflowDataType class functionality."""
    
    def test_valid_datatype_creation(self):
        """Test creation of valid datatype specifications."""
        # Test signed integer
        dtype = DataflowDataType("INT", 8, True, "")
        assert dtype.base_type == "INT"
        assert dtype.bitwidth == 8
        assert dtype.signed == True
        assert dtype.finn_type == "INT8"
        
        # Test unsigned integer
        dtype = DataflowDataType("UINT", 16, False, "")
        assert dtype.base_type == "UINT"
        assert dtype.bitwidth == 16
        assert dtype.signed == False
        assert dtype.finn_type == "UINT16"
        
        # Test float
        dtype = DataflowDataType("FLOAT", 32, True, "")
        assert dtype.base_type == "FLOAT"
        assert dtype.bitwidth == 32
        assert dtype.signed == True
        assert dtype.finn_type == "FLOAT32"
    
    def test_invalid_datatype_creation(self):
        """Test validation of invalid datatype specifications."""
        # Invalid base type
        with pytest.raises(ValueError, match="Invalid base_type"):
            DataflowDataType("INVALID", 8, True, "")
        
        # Invalid bitwidth
        with pytest.raises(ValueError, match="Invalid bitwidth"):
            DataflowDataType("INT", 0, True, "")
        
        with pytest.raises(ValueError, match="Invalid bitwidth"):
            DataflowDataType("INT", -8, True, "")
        
        # Inconsistent UINT with signed
        with pytest.raises(ValueError, match="UINT cannot be signed"):
            DataflowDataType("UINT", 8, True, "")


@pytest.mark.unit 
class TestDataTypeConstraint:
    """Test DataTypeConstraint class functionality."""
    
    def test_valid_constraint_creation(self, basic_datatype_constraint):
        """Test creation of valid datatype constraints."""
        constraint = basic_datatype_constraint
        
        assert constraint.base_types == ["INT", "UINT"]
        assert constraint.min_bitwidth == 1
        assert constraint.max_bitwidth == 32
        assert constraint.signed_allowed == True
        assert constraint.unsigned_allowed == True
    
    def test_invalid_constraint_creation(self):
        """Test validation of invalid constraint specifications."""
        # Invalid bitwidth range
        with pytest.raises(ValueError):
            DataTypeConstraint(
                base_types=["INT"],
                min_bitwidth=32,
                max_bitwidth=8,  # max < min
                signed_allowed=True,
                unsigned_allowed=False
            )
        
        # Empty base types
        with pytest.raises(ValueError):
            DataTypeConstraint(
                base_types=[],
                min_bitwidth=1,
                max_bitwidth=32,
                signed_allowed=True,
                unsigned_allowed=True
            )
    
    def test_datatype_validation(self, basic_datatype_constraint):
        """Test datatype validation against constraints."""
        constraint = basic_datatype_constraint
        
        # Valid datatypes
        valid_int8 = DataflowDataType("INT", 8, True, "INT8")
        valid_uint16 = DataflowDataType("UINT", 16, False, "UINT16")
        
        assert constraint.validates(valid_int8) == True
        assert constraint.validates(valid_uint16) == True
        
        # Invalid datatypes
        invalid_float = DataflowDataType("FLOAT", 32, True, "FLOAT32")
        invalid_large = DataflowDataType("INT", 64, True, "INT64")
        
        assert constraint.validates(invalid_float) == False
        assert constraint.validates(invalid_large) == False


@pytest.mark.unit
class TestDataflowInterface:
    """Test DataflowInterface class functionality."""
    
    def test_valid_interface_creation(self, basic_datatype, basic_datatype_constraint):
        """Test creation of valid interface."""
        interface = DataflowInterface(
            name="input0",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[64, 56, 56],
            block_dims=[1, 56, 56],
            stream_dims=[1, 1, 8],
            dtype=basic_datatype,
            allowed_datatypes=[basic_datatype_constraint]
        )
        
        assert interface.name == "input0"
        assert interface.interface_type == DataflowInterfaceType.INPUT
        assert interface.tensor_dims == [64, 56, 56]
        assert interface.block_dims == [1, 56, 56]
        assert interface.stream_dims == [1, 1, 8]
        assert interface.dtype == basic_datatype
        assert len(interface.allowed_datatypes) == 1
    
    def test_dimension_relationships_validation(self, basic_datatype):
        """Test validation of dimension relationships."""
        # Valid relationships
        valid_interface = DataflowInterface(
            name="test", 
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[64, 56],
            block_dims=[16, 56], 
            stream_dims=[4, 8],
            dtype=basic_datatype
        )
        
        # tensor_dims divisible by block_dims
        assert 64 % 16 == 0
        assert 56 % 56 == 0
        
        # block_dims divisible by stream_dims
        assert 16 % 4 == 0
        assert 56 % 8 == 0
    
    def test_invalid_dimension_relationships(self, basic_datatype):
        """Test that invalid dimension relationships are caught."""
        # tensor_dims not divisible by block_dims
        with pytest.raises(ValueError, match="must be divisible"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                tensor_dims=[100],
                block_dims=[30],
                stream_dims=[5],
                dtype=basic_datatype
            )
        
        # block_dims not divisible by stream_dims
        with pytest.raises(ValueError, match="must be divisible"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                tensor_dims=[64],
                block_dims=[16],
                stream_dims=[5],  # 16 % 5 != 0
                dtype=basic_datatype
            )
    
    def test_flexible_dimension_lengths(self, basic_datatype):
        """Test interfaces with different dimension lengths."""
        # Different lengths are allowed
        interface = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[64, 56, 56],  # 3D
            block_dims=[16, 56],       # 2D 
            stream_dims=[4],           # 1D
            dtype=basic_datatype
        )
        
        # Should validate successfully
        assert interface.tensor_dims == [64, 56, 56]
        assert interface.block_dims == [16, 56]
        assert interface.stream_dims == [4]
    
    def test_num_blocks_calculation(self, cnn_input_interface):
        """Test calculation of number of blocks."""
        interface = cnn_input_interface
        
        num_blocks = interface.get_num_blocks()
        expected = [64//1, 56//56, 56//56]  # [64, 1, 1]
        assert num_blocks == expected
        
        total_blocks = interface.get_total_blocks()
        assert total_blocks == 64 * 1 * 1  # 64 total blocks
    
    def test_stream_width_calculation(self, transformer_input_interface):
        """Test AXI stream width calculation."""
        interface = transformer_input_interface
        
        # stream_dims = [1, 64], dtype = 8-bit
        # Expected width = 1 * 64 * 8 = 512 bits
        stream_width = interface.calculate_stream_width()
        assert stream_width == 512
        
        # Test with different stream dimensions
        interface.stream_dims = [1, 32]
        stream_width = interface.calculate_stream_width()
        assert stream_width == 256  # 1 * 32 * 8 = 256 bits
    
    def test_transfer_cycles_calculation(self, cnn_weight_interface):
        """Test transfer cycles calculation."""
        interface = cnn_weight_interface
        
        # block_dims = [32, 1, 3, 3] = 288 elements
        # stream_dims = [8, 1, 1, 1] = 8 elements per cycle
        # Expected cycles = 288 / 8 = 36 cycles
        cycles = interface.get_transfer_cycles()
        assert cycles == 36
    
    def test_cII_calculation(self, simple_interfaces):
        """Test calculation initiation interval calculation."""
        interface = simple_interfaces["input"]
        
        # block_dims = [16], stream_dims = [4]
        # cII = 16 / 4 = 4 cycles
        cII = interface.calculate_cII()
        assert cII == 4
    
    def test_parallelism_application(self, cnn_input_interface):
        """Test application of parallelism parameters."""
        interface = cnn_input_interface
        original_stream_dims = interface.stream_dims.copy()
        
        # Apply input parallelism
        interface.apply_parallelism(iPar=16)
        
        # stream_dims should be updated
        assert interface.stream_dims[0] == 16
        # Other dimensions should remain unchanged
        assert interface.stream_dims[1:] == original_stream_dims[1:]
    
    def test_constraint_validation(self, basic_datatype):
        """Test interface constraint validation."""
        constraint = DataTypeConstraint(
            base_types=["INT"],
            min_bitwidth=8,
            max_bitwidth=16,
            signed_allowed=True,
            unsigned_allowed=False
        )
        
        interface = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[64],
            block_dims=[16],
            stream_dims=[4],
            dtype=basic_datatype,
            allowed_datatypes=[constraint]
        )
        
        # Validate should pass
        result = interface.validate()
        assert result.is_valid()
    
    def test_axi_metadata_handling(self, basic_datatype):
        """Test AXI metadata storage and access."""
        axi_metadata = {
            "TDATA_WIDTH": 128,
            "TUSER_WIDTH": 8,
            "TID_WIDTH": 4
        }
        
        interface = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[64],
            block_dims=[16],
            stream_dims=[4],
            dtype=basic_datatype,
            axi_metadata=axi_metadata
        )
        
        assert interface.axi_metadata == axi_metadata
        assert interface.axi_metadata["TDATA_WIDTH"] == 128
    
    def test_memory_footprint_calculation(self, cnn_weight_interface):
        """Test memory footprint calculation."""
        interface = cnn_weight_interface
        
        # block_dims = [32, 1, 3, 3] = 288 elements
        # dtype = 8 bits = 1 byte per element
        # Expected footprint = 288 bytes
        footprint = interface.calculate_memory_footprint()
        assert footprint == 288
    
    def test_string_representation(self, cnn_input_interface):
        """Test string representation of interface."""
        interface = cnn_input_interface
        
        str_repr = str(interface)
        assert "input0" in str_repr
        assert "INPUT" in str_repr
        assert "[64, 56, 56]" in str_repr
        assert "[1, 56, 56]" in str_repr
        assert "INT8" in str_repr
    
    def test_interface_factory_methods(self, basic_datatype):
        """Test factory methods for interface creation."""
        # Test from_tensor_shape factory
        interface = DataflowInterface.from_tensor_shape(
            name="factory_test",
            interface_type=DataflowInterfaceType.INPUT,
            original_shape=[1, 64, 56, 56],  # Including batch
            block_dims=[1, 56, 56],
            dtype=basic_datatype
        )
        
        # Should exclude batch dimension
        assert interface.tensor_dims == [64, 56, 56]
        assert interface.block_dims == [1, 56, 56]
        assert interface.stream_dims == [1, 1, 1]  # Default stream dims
    
    def test_interface_validation_errors(self, basic_datatype):
        """Test comprehensive validation error detection."""
        # Empty dimensions
        with pytest.raises(ValueError, match="cannot be empty"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                tensor_dims=[],
                block_dims=[16],
                stream_dims=[4],
                dtype=basic_datatype
            )
        
        # Zero or negative dimensions
        with pytest.raises(ValueError, match="must be positive"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                tensor_dims=[0, 56],
                block_dims=[16, 56],
                stream_dims=[4, 8],
                dtype=basic_datatype
            )
        
        # Block larger than tensor
        with pytest.raises(ValueError, match="must be divisible"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                tensor_dims=[32],
                block_dims=[64],
                stream_dims=[8],
                dtype=basic_datatype
            )


@pytest.mark.unit
class TestTensorShapeReconstruction:
    """Test tensor shape reconstruction and chunking methods."""
    
    def test_tensor_shape_reconstruction(self, cnn_input_interface):
        """Test reconstruction of original tensor shape."""
        interface = cnn_input_interface
        
        # Should reconstruct to original shape
        reconstructed = interface.reconstruct_tensor_shape()
        assert reconstructed == [64, 56, 56]  # Same as tensor_dims
    
    def test_compute_tensor_dims_from_chunking(self):
        """Test computation of tensor dimensions from original shape and chunking."""
        
        # Test broadcast mode
        original_shape = [30, 50]
        block_dims = [50]
        tensor_dims = DataflowInterface._compute_tensor_dims_from_chunking(
            original_shape, block_dims, "broadcast"
        )
        assert tensor_dims == [30]
        
        # Test divide mode
        original_shape = [20, 40]
        block_dims = [4, 8]
        tensor_dims = DataflowInterface._compute_tensor_dims_from_chunking(
            original_shape, block_dims, "divide"
        )
        assert tensor_dims == [5, 5]  # [20//4, 40//8]
        
        # Test 1D case
        original_shape = [1500]
        block_dims = [50]
        tensor_dims = DataflowInterface._compute_tensor_dims_from_chunking(
            original_shape, block_dims, "broadcast"
        )
        assert tensor_dims == [30]  # 1500 / 50 = 30
    
    def test_real_world_chunking_examples(self, test_utils):
        """Test real-world tensor chunking scenarios."""
        
        # CNN ResNet-style
        resnet_interface = test_utils.create_test_interface(
            tensor_dims=[256, 14, 14],   # 256 channels, 14x14 spatial
            block_dims=[1, 14, 14],      # Process 1 channel at a time
            stream_dims=[1, 1, 8],       # 8 elements per cycle
            interface_type=DataflowInterfaceType.INPUT
        )
        
        num_blocks = resnet_interface.get_num_blocks()
        assert num_blocks == [256, 1, 1]  # 256 blocks along channel dimension
        
        # Transformer BERT-style
        bert_interface = test_utils.create_test_interface(
            tensor_dims=[512, 768],      # 512 tokens, 768 features
            block_dims=[1, 768],         # Process 1 token at a time
            stream_dims=[1, 64],         # 64 features per cycle
            interface_type=DataflowInterfaceType.INPUT
        )
        
        num_blocks = bert_interface.get_num_blocks()
        assert num_blocks == [512, 1]   # 512 blocks along sequence dimension
        
        transfer_cycles = bert_interface.get_transfer_cycles()
        assert transfer_cycles == 768 // 64  # 12 cycles per token
    
    def test_chunking_validation(self, test_utils):
        """Test validation of chunking configurations."""
        
        # Valid chunking
        valid_interface = test_utils.create_test_interface(
            tensor_dims=[128, 128],
            block_dims=[32, 128],
            stream_dims=[8, 16],
            interface_type=DataflowInterfaceType.INPUT
        )
        
        validation_result = valid_interface.validate()
        assert validation_result.is_valid()
        
        # Invalid chunking: misaligned dimensions
        with pytest.raises(ValueError):
            test_utils.create_test_interface(
                tensor_dims=[127, 128],  # Not divisible by block_dims
                block_dims=[32, 128],
                stream_dims=[8, 16],
                interface_type=DataflowInterfaceType.INPUT
            )