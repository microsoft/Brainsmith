"""
Unit tests for DataflowInterface class

Tests cover interface creation, validation, constraint checking,
and datatype constraint functionality.
"""

import pytest
import numpy as np

from brainsmith.dataflow.core.dataflow_interface import (
    DataflowInterface,
    DataflowDataType,
    DataTypeConstraint,
    ConstraintType,
    DivisibilityConstraint,
    RangeConstraint
)
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.validation import ValidationSeverity

class TestDataflowDataType:
    """Test DataflowDataType class"""
    
    def test_valid_datatype_creation(self):
        """Test creation of valid datatype specifications"""
        # Test signed integer
        dtype = DataflowDataType(
            base_type="INT",
            bitwidth=8,
            signed=True,
            finn_type=""
        )
        assert dtype.base_type == "INT"
        assert dtype.bitwidth == 8
        assert dtype.signed == True
        assert dtype.finn_type == "INT8"
        
        # Test unsigned integer
        dtype = DataflowDataType(
            base_type="UINT",
            bitwidth=16,
            signed=False,
            finn_type=""
        )
        assert dtype.base_type == "UINT"
        assert dtype.bitwidth == 16
        assert dtype.signed == False
        assert dtype.finn_type == "UINT16"
        
        # Test floating point
        dtype = DataflowDataType(
            base_type="FLOAT",
            bitwidth=32,
            signed=True,
            finn_type=""
        )
        assert dtype.base_type == "FLOAT"
        assert dtype.bitwidth == 32
        assert dtype.finn_type == "FLOAT32"
    
    def test_invalid_datatype_creation(self):
        """Test creation with invalid parameters raises errors"""
        # Invalid base type
        with pytest.raises(ValueError, match="Invalid base_type"):
            DataflowDataType(
                base_type="INVALID",
                bitwidth=8,
                signed=True,
                finn_type=""
            )
        
        # Invalid bitwidth
        with pytest.raises(ValueError, match="Invalid bitwidth"):
            DataflowDataType(
                base_type="INT",
                bitwidth=0,
                signed=True,
                finn_type=""
            )
        
        # UINT cannot be signed
        with pytest.raises(ValueError, match="UINT base_type cannot be signed"):
            DataflowDataType(
                base_type="UINT",
                bitwidth=8,
                signed=True,
                finn_type=""
            )

class TestDataTypeConstraint:
    """Test DataTypeConstraint class"""
    
    def test_valid_constraint_creation(self):
        """Test creation of valid datatype constraints"""
        constraint = DataTypeConstraint(
            base_types=["INT", "UINT"],
            min_bitwidth=1,
            max_bitwidth=32,
            signed_allowed=True,
            unsigned_allowed=True
        )
        assert constraint.base_types == ["INT", "UINT"]
        assert constraint.min_bitwidth == 1
        assert constraint.max_bitwidth == 32
        assert constraint.signed_allowed == True
        assert constraint.unsigned_allowed == True
    
    def test_invalid_constraint_creation(self):
        """Test creation with invalid parameters raises errors"""
        # Invalid bitwidth
        with pytest.raises(ValueError, match="min_bitwidth must be positive"):
            DataTypeConstraint(
                base_types=["INT"],
                min_bitwidth=0,
                max_bitwidth=32,
                signed_allowed=True,
                unsigned_allowed=True
            )
        
        # max < min bitwidth
        with pytest.raises(ValueError, match="max_bitwidth must be >= min_bitwidth"):
            DataTypeConstraint(
                base_types=["INT"],
                min_bitwidth=32,
                max_bitwidth=16,
                signed_allowed=True,
                unsigned_allowed=True
            )
        
        # Neither signed nor unsigned allowed
        with pytest.raises(ValueError, match="At least one of signed_allowed or unsigned_allowed must be True"):
            DataTypeConstraint(
                base_types=["INT"],
                min_bitwidth=1,
                max_bitwidth=32,
                signed_allowed=False,
                unsigned_allowed=False
            )
    
    def test_datatype_validation(self):
        """Test datatype validation against constraints"""
        constraint = DataTypeConstraint(
            base_types=["INT", "UINT"],
            min_bitwidth=4,
            max_bitwidth=16,
            signed_allowed=True,
            unsigned_allowed=False
        )
        
        # Valid datatype
        valid_dtype = DataflowDataType("INT", 8, True, "")
        assert constraint.is_valid_datatype(valid_dtype) == True
        
        # Invalid base type
        invalid_dtype = DataflowDataType("FLOAT", 8, True, "")
        assert constraint.is_valid_datatype(invalid_dtype) == False
        
        # Invalid bitwidth (too small)
        invalid_dtype = DataflowDataType("INT", 2, True, "")
        assert constraint.is_valid_datatype(invalid_dtype) == False
        
        # Invalid bitwidth (too large)
        invalid_dtype = DataflowDataType("INT", 32, True, "")
        assert constraint.is_valid_datatype(invalid_dtype) == False
        
        # Invalid sign (unsigned not allowed)
        invalid_dtype = DataflowDataType("UINT", 8, False, "")
        assert constraint.is_valid_datatype(invalid_dtype) == False

class TestDataflowInterface:
    """Test DataflowInterface class"""
    
    def test_valid_interface_creation(self):
        """Test creation of valid interface with constraint support"""
        dtype = DataflowDataType("INT", 8, True, "")
        constraint = DataTypeConstraint(
            base_types=["INT", "UINT"],
            min_bitwidth=1,
            max_bitwidth=32,
            signed_allowed=True,
            unsigned_allowed=True
        )
        
        interface = DataflowInterface(
            name="input0",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[16],
            block_dims=[16],
            stream_dims=[4],
            dtype=dtype,
            allowed_datatypes=[constraint]
        )
        
        assert interface.name == "input0"
        assert interface.interface_type == InterfaceType.INPUT
        assert interface.tensor_dims == [16]
        assert interface.block_dims == [16]
        assert interface.stream_dims == [4]
        assert interface.dtype == dtype
        assert len(interface.allowed_datatypes) == 1
        assert interface.allowed_datatypes[0] == constraint
    
    def test_invalid_dimension_relationships(self):
        """Test that invalid dimension relationships raise errors"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # block_dims not divisible by stream_dims (streaming constraint)
        with pytest.raises(ValueError, match="block_dims.*must be divisible by stream_dims"):
            DataflowInterface(
                name="test",
                interface_type=InterfaceType.INPUT,
                tensor_dims=[16],  # Fixed: 16 % 16 == 0 (valid chunking)
                block_dims=[16],
                stream_dims=[5],  # 16 % 5 != 0 (invalid streaming)
                dtype=dtype
            )
        
        # tensor_dims not divisible by block_dims (chunking constraint)
        with pytest.raises(ValueError, match="tensor_dims.*must be divisible by block_dims"):
            DataflowInterface(
                name="test2",
                interface_type=InterfaceType.INPUT,
                tensor_dims=[30],
                block_dims=[50],  # 30 % 50 != 0 
                stream_dims=[1],
                dtype=dtype
            )
        
        # Empty dimensions should fail
        with pytest.raises(ValueError, match="cannot be empty"):
            DataflowInterface(
                name="test3",
                interface_type=InterfaceType.INPUT,
                tensor_dims=[],  # Empty qDim
                block_dims=[16],
                stream_dims=[4],
                dtype=dtype
            )
        
    def test_flexible_dimension_lengths(self):
        """Test that different dimension lengths are now supported"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Multi-dimensional qDim with fewer tDim and stream_dims dimensions (BERT-like example)
        interface = DataflowInterface(
            name="bert_input",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[128, 768],     # BERT: seqlen=128, hidden=768
            block_dims=[128],          # Process 128 sequence elements per chunk  
            stream_dims=[8],            # 8-way parallelism (128 % 8 == 0, valid streaming)
            dtype=dtype
        )
        
        # Should validate successfully
        result = interface.validate_constraints()
        assert len(result.errors) == 0
        
        # Check calculations work with different lengths
        num_tensors = interface.get_num_blocks()
        assert len(num_tensors) == 1  # min(len(qDim), len(tDim)) = min(2, 1) = 1
        assert num_tensors == [1]  # qDim[0:1] ÷ tDim[0:1] = [128] ÷ [128] = [1]
        
        # Verify cII calculation works
        cII = interface.calculate_cII()
        assert cII >= 1  # Should calculate based on overlapping dimensions

        # Zero or negative dimensions
        with pytest.raises(ValueError, match="must be positive"):
            DataflowInterface(
                name="test",
                interface_type=InterfaceType.INPUT,
                tensor_dims=[0],
                block_dims=[16],
                stream_dims=[4],
                dtype=dtype
            )
    
    def test_default_constraints(self):
        """Test that default datatype constraints are set when none provided"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        interface = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[16],
            block_dims=[16],
            stream_dims=[4],
            dtype=dtype
        )
        
        # Should have default constraint
        assert len(interface.allowed_datatypes) == 1
        default_constraint = interface.allowed_datatypes[0]
        assert "INT" in default_constraint.base_types
        assert "UINT" in default_constraint.base_types
        assert default_constraint.min_bitwidth == 1
        assert default_constraint.max_bitwidth == 32
        assert default_constraint.signed_allowed == True
        assert default_constraint.unsigned_allowed == True
    
    def test_stream_width_calculation(self):
        """Test AXI stream width calculation"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        interface = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[16],
            block_dims=[16],
            stream_dims=[4],
            dtype=dtype
        )
        
        # 4 elements per cycle * 8 bits = 32 bits, aligned to 8-bit boundary = 32 bits
        assert interface.calculate_stream_width() == 32
        
        # Test with misaligned width
        dtype_misaligned = DataflowDataType("INT", 9, True, "")
        interface.dtype = dtype_misaligned
        
        # 4 elements * 9 bits = 36 bits, aligned to 8-bit boundary = 40 bits
        assert interface.calculate_stream_width() == 40
    
    def test_constraint_validation(self):
        """Test constraint validation functionality"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Valid interface
        interface = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[16],
            block_dims=[16],
            stream_dims=[4],
            dtype=dtype
        )
        
        result = interface.validate_constraints()
        assert result.success == True
        assert len(result.errors) == 0
        
        # Test validation separately - create valid interface first, then modify for testing
        # Valid interface for constraint testing
        interface_for_testing = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[15],  # qDim can be any value now
            block_dims=[15],  # Make divisible for construction
            stream_dims=[5],   # 15 % 5 = 0, valid for construction
            dtype=dtype
        )
        
        # Now modify to create invalid streaming constraint for testing
        interface_for_testing.block_dims = [16]  # 16 % 5 != 0, invalid streaming
        
        result = interface_for_testing.validate_constraints()
        assert result.success == False
        assert len(result.errors) > 0
        assert any("divisible" in error.message for error in result.errors)
    
    def test_parallelism_application(self):
        """Test parallelism parameter application"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Input interface
        input_interface = DataflowInterface(
            name="input0",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[16],
            block_dims=[16],
            stream_dims=[1],
            dtype=dtype
        )
        
        input_interface.apply_parallelism(iPar=4)
        assert input_interface.stream_dims[0] == 4
        
        # Weight interface
        weight_interface = DataflowInterface(
            name="weights",
            interface_type=InterfaceType.WEIGHT,
            tensor_dims=[32],
            block_dims=[32],
            stream_dims=[1],
            dtype=dtype
        )
        
        weight_interface.apply_parallelism(wPar=8)
        assert weight_interface.stream_dims[0] == 8
    
    def test_axi_signal_generation(self):
        """Test AXI signal generation for different interface types"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Input interface (slave)
        input_interface = DataflowInterface(
            name="input0",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[16],
            block_dims=[16],
            stream_dims=[4],
            dtype=dtype
        )
        
        signals = input_interface.get_axi_signals()
        assert "input0_TDATA" in signals
        assert "input0_TVALID" in signals
        assert "input0_TREADY" in signals
        assert signals["input0_TDATA"]["direction"] == "input"
        assert signals["input0_TVALID"]["direction"] == "input"
        assert signals["input0_TREADY"]["direction"] == "output"
        
        # Output interface (master)
        output_interface = DataflowInterface(
            name="output0",
            interface_type=InterfaceType.OUTPUT,
            tensor_dims=[16],
            block_dims=[16],
            stream_dims=[4],
            dtype=dtype
        )
        
        signals = output_interface.get_axi_signals()
        assert "output0_TDATA" in signals
        assert "output0_TVALID" in signals
        assert "output0_TREADY" in signals
        assert signals["output0_TDATA"]["direction"] == "output"
        assert signals["output0_TVALID"]["direction"] == "output"
        assert signals["output0_TREADY"]["direction"] == "input"
    
    def test_datatype_validation(self):
        """Test datatype validation against allowed constraints"""
        # Create constraint that only allows 8-bit signed integers
        strict_constraint = DataTypeConstraint(
            base_types=["INT"],
            min_bitwidth=8,
            max_bitwidth=8,
            signed_allowed=True,
            unsigned_allowed=False
        )
        
        dtype = DataflowDataType("INT", 8, True, "")
        interface = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[16],
            block_dims=[16],
            stream_dims=[4],
            dtype=dtype,
            allowed_datatypes=[strict_constraint]
        )
        
        # Valid datatype
        valid_dtype = DataflowDataType("INT", 8, True, "")
        assert interface.validate_datatype(valid_dtype) == True
        
        # Invalid datatype (wrong bitwidth)
        invalid_dtype = DataflowDataType("INT", 16, True, "")
        assert interface.validate_datatype(invalid_dtype) == False
        
        # Invalid datatype (unsigned)
        invalid_dtype = DataflowDataType("UINT", 8, False, "")
        assert interface.validate_datatype(invalid_dtype) == False
    
    def test_memory_footprint_calculation(self):
        """Test memory footprint calculation"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        interface = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[16, 16],
            block_dims=[16, 16],
            stream_dims=[4, 4],
            dtype=dtype
        )
        
        # 16 * 16 elements * 8 bits = 2048 bits
        assert interface.get_memory_footprint() == 2048
    
    def test_transfer_cycles_calculation(self):
        """Test transfer cycles calculation"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        interface = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[64],
            block_dims=[64],
            stream_dims=[4],
            dtype=dtype
        )
        
        # 64 elements total, 4 elements per cycle = 16 cycles
        assert interface.get_transfer_cycles() == 16
    
    def test_string_representation(self):
        """Test string representation of interface"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        interface = DataflowInterface(
            name="test_interface",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[16],
            block_dims=[16],
            stream_dims=[4],
            dtype=dtype
        )
        
        str_repr = str(interface)
        assert "test_interface" in str_repr
        assert "input" in str_repr
        assert "INT8" in str_repr
        assert "[16]" in str_repr
        assert "[4]" in str_repr

class TestTensorChunking:
    """Test tensor chunking functionality"""
    
    def test_tensor_shape_reconstruction(self):
        """Test tensor shape reconstruction from qDim and tDim"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Simple case: tensor_dims=[150], block_dims=[30] → valid chunking (150 % 30 == 0)
        interface = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[150],  # Original tensor shape
            block_dims=[30],   # Chunk size (150 % 30 == 0)
            stream_dims=[10],   # Stream parallelism (30 % 10 == 0)
            dtype=dtype
        )
        
        reconstructed = interface.reconstruct_tensor_shape()
        # In new architecture: qDim * num_tensors where num_tensors = qDim/tDim
        num_tensors = interface.get_num_blocks()  # [150/30] = [5]
        expected = [num_tensors[0] * interface.block_dims[0]]  # [5 * 30] = [150]
        assert reconstructed == expected
        
        # Multi-dimensional case: valid chunking with qDim divisible by tDim
        interface_multi = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[15, 20],   # Original shape (15 % 3 == 0, 20 % 5 == 0)
            block_dims=[3, 5],     # Chunk sizes
            stream_dims=[1, 1],     # Stream parallelism (3 % 1 == 0, 5 % 1 == 0)
            dtype=dtype
        )
        
        reconstructed = interface_multi.reconstruct_tensor_shape()
        # num_tensors = [15/3, 20/5] = [5, 4], reconstructed = [5*3, 4*5] = [15, 20]
        assert reconstructed == [15, 20]  # Should reconstruct to original qDim
    
    def test_tensor_chunking_validation(self):
        """Test validation of tensor chunking against original shape"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Valid chunking: tensor_dims=[150], block_dims=[30] (150 % 30 == 0)
        interface = DataflowInterface(
            name="test",
            interface_type=InterfaceType.INPUT,
            tensor_dims=[150],   # Original tensor shape
            block_dims=[30],    # Chunk size (valid: 150 % 30 == 0)  
            stream_dims=[10],    # Stream parallelism (valid: 30 % 10 == 0)
            dtype=dtype
        )
        
        # Test validation against compatible original shape
        original_shape = [150]  # Same as qDim - should validate successfully
        result = interface.validate_tensor_chunking(original_shape)
        assert result.success == True
        
        # Invalid chunking: element count mismatch
        incompatible_shape = [2000]  # Different element count
        result = interface.validate_tensor_chunking(incompatible_shape)
        assert result.success == False
        assert len(result.errors) > 0
        assert any("mismatch" in (error.message if hasattr(error, 'message') else str(error)) for error in result.errors)
    
    def test_from_tensor_chunking_factory_method(self):
        """Test factory method for creating interfaces from tensor chunking"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Test case: original [150, 50] with block_dims=[50] - valid chunking (150 % 50 == 0)
        interface = DataflowInterface.from_tensor_chunking(
            name="input0",
            interface_type=InterfaceType.INPUT,
            original_shape=[150, 50],  # 2D tensor shape
            block_dims=[50],
            dtype=dtype,
            chunking_mode="broadcast"
        )
        
        assert interface.name == "input0"
        assert interface.interface_type == InterfaceType.INPUT
        assert interface.block_dims == [50]
        assert interface.tensor_dims == [150, 50]  # In new architecture: preserves original shape
        assert interface.stream_dims == [1]  # Default stream_dims for tDim dimensions
        assert interface.dtype == dtype
    
    def test_compute_tensor_dims_from_chunking(self):
        """Test qDim computation from original shape and tDim"""
        
        # Test broadcast mode with single tDim
        # Example: original [30, 50] with block_dims=[50] → qDim should be [30]
        original_shape = [30, 50]  # 1500 total elements
        block_dims = [50]
        tensor_dims = DataflowInterface._compute_tensor_dims_from_chunking(original_shape, block_dims, "broadcast")
        
        # For original [30, 50] with block_dims=[50], qDim should be [30]
        assert tensor_dims == [30]
        
        # Test with multiple tDim dimensions
        # Example: original [10, 30, 50] where total=15000, block_dims=[3,5] (15 elements)
        original_shape = [10, 30, 50]  # 15000 total elements
        block_dims = [3, 5]  # 15 elements per tensor
        tensor_dims = DataflowInterface._compute_tensor_dims_from_chunking(original_shape, block_dims, "broadcast")
        
        # Total elements = 15000, tDim elements = 15, so qDim elements = 1000
        assert tensor_dims == [1000]
        
        # Test divide mode
        original_shape = [20, 40]
        block_dims = [4, 8]
        tensor_dims = DataflowInterface._compute_tensor_dims_from_chunking(original_shape, block_dims, "divide")
        
        # Direct division: [20//4, 40//8] = [5, 5]
        assert tensor_dims == [5, 5]
        
        # Test simple 1D case
        original_shape = [1500]
        block_dims = [50]
        tensor_dims = DataflowInterface._compute_tensor_dims_from_chunking(original_shape, block_dims, "broadcast")
        
        # 1500 / 50 = 30
        assert tensor_dims == [30]
    
    def test_real_world_chunking_examples(self):
        """Test with user-provided real-world examples"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Example 1: tensor [150] with block_dims=[50] → tensor_dims=[150] (shape preserved)
        # This tests with valid divisible dimensions
        interface1 = DataflowInterface.from_tensor_chunking(
            name="example1",
            interface_type=InterfaceType.INPUT,
            original_shape=[150],  # Must be divisible by tDim
            block_dims=[50],
            dtype=dtype
        )
        
        assert interface1.tensor_dims == [150]  # original_shape preserved as qDim
        assert interface1.block_dims == [50]
        
        # Validate num_tensors calculation
        num_tensors = interface1.get_num_blocks()
        assert num_tensors == [3]  # 150 ÷ 50 = 3 chunks
        
        # Example 2: Multi-dimensional tensor with valid divisible dimensions
        interface2 = DataflowInterface.from_tensor_chunking(
            name="example2",
            interface_type=InterfaceType.INPUT,
            original_shape=[15, 10],  # Must be divisible by tDim
            block_dims=[3, 5],
            dtype=dtype
        )
        
        assert interface2.tensor_dims == [15, 10]  # original_shape preserved as qDim
        assert interface2.block_dims == [3, 5]
        
        # Validate num_tensors calculation  
        num_tensors2 = interface2.get_num_blocks()
        assert num_tensors2 == [5, 2]  # [15÷3, 10÷5] = [5, 2]
        
        # Validate element count consistency
        original_elements = np.prod([15, 10])  # 150
        chunks_total = np.prod(num_tensors2)   # 5 * 2 = 10 chunks
        elements_per_chunk = np.prod(interface2.block_dims)  # 3 * 5 = 15
        total_elements = chunks_total * elements_per_chunk  # 10 * 15 = 150
        assert original_elements == total_elements
        
        # Example 3: Simple 1D case - tensor [1500] with block_dims=[50] 
        interface3 = DataflowInterface.from_tensor_chunking(
            name="example3",
            interface_type=InterfaceType.INPUT,
            original_shape=[1500],
            block_dims=[50],
            dtype=dtype
        )
        
        assert interface3.tensor_dims == [1500]  # original_shape preserved as qDim
        # num_tensors should be 1500 // 50 = 30
        num_tensors3 = interface3.get_num_blocks()
        assert num_tensors3 == [30]
        assert interface3.block_dims == [50]
        
        # Validate reconstruction
        reconstructed = interface3.reconstruct_tensor_shape()
        assert np.prod(reconstructed) == 1500

if __name__ == "__main__":
    pytest.main([__file__])
