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
        
        # qDim < tDim
        with pytest.raises(ValueError, match="qDim.*must be >= tDim"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                qDim=[8],
                tDim=[16],
                sDim=[4],
                dtype=dtype
            )
        
        # tDim < sDim
        with pytest.raises(ValueError, match="tDim.*must be >= sDim"):
            DataflowInterface(
                name="test",
                interface_type=DataflowInterfaceType.INPUT,
                qDim=[16],
                tDim=[4],
                sDim=[8],
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
        
        # Invalid divisibility: qDim not divisible by tDim
        interface_invalid = DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[15],
            tDim=[16],
            sDim=[4],
            dtype=dtype
        )
        
        result = interface_invalid.validate_constraints()
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

if __name__ == "__main__":
    pytest.main([__file__])
