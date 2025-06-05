"""
Unit tests for DataflowInterface class

Tests cover interface creation, validation, constraint checking,
and datatype constraint functionality.
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
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[16],
            tDim=[16],
            sDim=[4],
            dtype=dtype,
            allowed_datatypes=[constraint]
        )
        
        assert interface.name == "input0"
        assert interface.interface_type == DataflowInterfaceType.INPUT
        assert interface.qDim == [16]
        assert interface.tDim == [16]
        assert interface.sDim == [4]
        assert interface.dtype == dtype
        assert len(interface.allowed_datatypes) == 1
        assert interface.allowed_datatypes[0] == constraint
    
    def test_invalid_dimension_relationships(self):
        """Test that invalid dimension relationships raise errors"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # tDim not divisible by sDim (streaming constraint)
        with pytest.raises(ValueError, match="tDim.*must be divisible by sDim"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                qDim=[8],
                tDim=[16],
                sDim=[5],  # 16 % 5 != 0
                dtype=dtype
            )
        
        # Mismatched dimension lengths
        with pytest.raises(ValueError, match="qDim, tDim, and sDim must have same length"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                qDim=[16, 16],
                tDim=[16],
                sDim=[4],
                dtype=dtype
            )
        
        # Zero or negative dimensions
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                qDim=[0],
                tDim=[16],
                sDim=[4],
                dtype=dtype
            )
    
    def test_default_constraints(self):
        """Test that default datatype constraints are set when none provided"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        interface = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[16],
            tDim=[16],
            sDim=[4],
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
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[16],
            tDim=[16],
            sDim=[4],
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
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[16],
            tDim=[16],
            sDim=[4],
            dtype=dtype
        )
        
        result = interface.validate_constraints()
        assert result.success == True
        assert len(result.errors) == 0
        
        # Test validation separately - create valid interface first, then modify for testing
        # Valid interface for constraint testing
        interface_for_testing = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[15],  # qDim can be any value now
            tDim=[15],  # Make divisible for construction
            sDim=[5],   # 15 % 5 = 0, valid for construction
            dtype=dtype
        )
        
        # Now modify to create invalid streaming constraint for testing
        interface_for_testing.tDim = [16]  # 16 % 5 != 0, invalid streaming
        
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
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[16],
            tDim=[16],
            sDim=[1],
            dtype=dtype
        )
        
        input_interface.apply_parallelism(iPar=4)
        assert input_interface.sDim[0] == 4
        
        # Weight interface
        weight_interface = DataflowInterface(
            name="weights",
            interface_type=DataflowInterfaceType.WEIGHT,
            qDim=[32],
            tDim=[32],
            sDim=[1],
            dtype=dtype
        )
        
        weight_interface.apply_parallelism(wPar=8)
        assert weight_interface.sDim[0] == 8
    
    def test_axi_signal_generation(self):
        """Test AXI signal generation for different interface types"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Input interface (slave)
        input_interface = DataflowInterface(
            name="input0",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[16],
            tDim=[16],
            sDim=[4],
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
            interface_type=DataflowInterfaceType.OUTPUT,
            qDim=[16],
            tDim=[16],
            sDim=[4],
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
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[16],
            tDim=[16],
            sDim=[4],
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
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[16, 16],
            tDim=[16, 16],
            sDim=[4, 4],
            dtype=dtype
        )
        
        # 16 * 16 elements * 8 bits = 2048 bits
        assert interface.get_memory_footprint() == 2048
    
    def test_transfer_cycles_calculation(self):
        """Test transfer cycles calculation"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        interface = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[64],
            tDim=[64],
            sDim=[4],
            dtype=dtype
        )
        
        # 64 elements total, 4 elements per cycle = 16 cycles
        assert interface.get_transfer_cycles() == 16
    
    def test_string_representation(self):
        """Test string representation of interface"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        interface = DataflowInterface(
            name="test_interface",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[16],
            tDim=[16],
            sDim=[4],
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
        
        # Simple case: qDim=[30], tDim=[50] → original=[1500] elements
        interface = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[30],
            tDim=[50],
            sDim=[10],
            dtype=dtype
        )
        
        reconstructed = interface.reconstruct_tensor_shape()
        assert reconstructed == [1500]  # 30 * 50
        
        # Multi-dimensional case: qDim=[10, 20], tDim=[3, 5] → concatenated dimensions
        interface_multi = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[10, 20],
            tDim=[3, 5],
            sDim=[1, 1],
            dtype=dtype
        )
        
        reconstructed = interface_multi.reconstruct_tensor_shape()
        assert reconstructed == [30, 100]  # Element-wise multiplication: [10*3, 20*5]
    
    def test_tensor_chunking_validation(self):
        """Test validation of tensor chunking against original shape"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Valid chunking: original [10, 30, 50] with qDim=[30], tDim=[50]
        # Total elements: 10*30*50 = 15000, qDim*tDim = 30*50 = 1500
        # Should be valid if batch handling is considered separately
        interface = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[30],
            tDim=[50],
            sDim=[10],
            dtype=dtype
        )
        
        # Test validation against compatible original shape
        original_shape = [1500]  # 30 * 50 = 1500 elements total
        result = interface.validate_tensor_chunking(original_shape)
        assert result.success == True
        
        # Invalid chunking: element count mismatch
        incompatible_shape = [2000]  # Different element count
        result = interface.validate_tensor_chunking(incompatible_shape)
        assert result.success == False
        assert len(result.errors) > 0
        assert any("mismatch" in error.message for error in result.errors)
    
    def test_from_tensor_chunking_factory_method(self):
        """Test factory method for creating interfaces from tensor chunking"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Test case: original [30, 50] with tDim=[50] → qDim=[30]
        # This represents chunking where we process 50 elements at a time from a 30x50 tensor
        interface = DataflowInterface.from_tensor_chunking(
            name="input0",
            interface_type=DataflowInterfaceType.INPUT,
            original_shape=[30, 50],  # 2D tensor shape
            tDim=[50],
            dtype=dtype,
            chunking_mode="broadcast"
        )
        
        assert interface.name == "input0"
        assert interface.interface_type == DataflowInterfaceType.INPUT
        assert interface.tDim == [50]
        assert interface.qDim == [30]  # Should be 30 from first dimension
        assert interface.sDim == [50]  # Initially matches tDim
        assert interface.dtype == dtype
    
    def test_compute_qDim_from_chunking(self):
        """Test qDim computation from original shape and tDim"""
        
        # Test broadcast mode with single tDim
        # Example: original [30, 50] with tDim=[50] → qDim should be [30]
        original_shape = [30, 50]  # 1500 total elements
        tDim = [50]
        qDim = DataflowInterface._compute_qDim_from_chunking(original_shape, tDim, "broadcast")
        
        # For original [30, 50] with tDim=[50], qDim should be [30]
        assert qDim == [30]
        
        # Test with multiple tDim dimensions
        # Example: original [10, 30, 50] where total=15000, tDim=[3,5] (15 elements)
        original_shape = [10, 30, 50]  # 15000 total elements
        tDim = [3, 5]  # 15 elements per tensor
        qDim = DataflowInterface._compute_qDim_from_chunking(original_shape, tDim, "broadcast")
        
        # Total elements = 15000, tDim elements = 15, so qDim elements = 1000
        assert qDim == [1000]
        
        # Test divide mode
        original_shape = [20, 40]
        tDim = [4, 8]
        qDim = DataflowInterface._compute_qDim_from_chunking(original_shape, tDim, "divide")
        
        # Direct division: [20//4, 40//8] = [5, 5]
        assert qDim == [5, 5]
        
        # Test simple 1D case
        original_shape = [1500]
        tDim = [50]
        qDim = DataflowInterface._compute_qDim_from_chunking(original_shape, tDim, "broadcast")
        
        # 1500 / 50 = 30
        assert qDim == [30]
    
    def test_real_world_chunking_examples(self):
        """Test with user-provided real-world examples"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        # Example 1: tensor [10, 30, 50] with tDim=[50] → qDim=[30]
        # Total elements without batch = 30*50 = 1500
        interface1 = DataflowInterface.from_tensor_chunking(
            name="example1",
            interface_type=DataflowInterfaceType.INPUT,
            original_shape=[30, 50],  # Non-batch dimensions
            tDim=[50],
            dtype=dtype
        )
        
        assert interface1.qDim == [30]
        assert interface1.tDim == [50]
        
        # Validate reconstruction
        reconstructed = interface1.reconstruct_tensor_shape()
        assert np.prod(reconstructed) == np.prod([30, 50])
        
        # Example 2: tensor [10, 30, 50] with tDim=[3,5] → qDim=[1000]
        # Where total elements = 10*30*50 = 15000
        # tDim=[3,5] means 15 elements per calculation
        # So qDim should represent 15000/15 = 1000 calculations
        interface2 = DataflowInterface.from_tensor_chunking(
            name="example2",
            interface_type=DataflowInterfaceType.INPUT,
            original_shape=[10, 30, 50],
            tDim=[3, 5],
            dtype=dtype
        )
        
        assert interface2.qDim == [1000, 1]  # 15000 // 15 = 1000, padded to match tDim length
        assert interface2.tDim == [3, 5]
        
        # Validate element count consistency
        original_elements = np.prod([10, 30, 50])
        chunked_elements = np.prod(interface2.qDim) * np.prod(interface2.tDim)
        assert original_elements == chunked_elements
        
        # Example 3: Simple 1D case - tensor [1500] with tDim=[50] → qDim=[30]
        interface3 = DataflowInterface.from_tensor_chunking(
            name="example3",
            interface_type=DataflowInterfaceType.INPUT,
            original_shape=[1500],
            tDim=[50],
            dtype=dtype
        )
        
        assert interface3.qDim == [30]  # 1500 // 50 = 30
        assert interface3.tDim == [50]
        
        # Validate reconstruction
        reconstructed = interface3.reconstruct_tensor_shape()
        assert np.prod(reconstructed) == 1500

if __name__ == "__main__":
    pytest.main([__file__])
