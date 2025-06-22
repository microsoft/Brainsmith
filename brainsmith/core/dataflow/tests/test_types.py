############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for basic types module"""

import pytest
from brainsmith.core.dataflow.core.types import (
    Shape, InterfaceDirection, DataType,
    prod, shape_to_string, parse_shape, shapes_compatible,
    broadcast_shapes, flatten_shape, reshape_compatible,
    tile_shape, is_valid_tiling,
    INT8, INT16, INT32, BINARY, BIPOLAR
)


class TestInterfaceDirection:
    """Tests for InterfaceDirection enum"""
    
    def test_enum_values(self):
        """Test enum member values"""
        assert InterfaceDirection.INPUT.value == "input"
        assert InterfaceDirection.OUTPUT.value == "output"
        assert InterfaceDirection.WEIGHT.value == "weight"
        assert InterfaceDirection.CONFIG.value == "config"
    
    def test_from_string(self):
        """Test creation from string"""
        assert InterfaceDirection.from_string("input") == InterfaceDirection.INPUT
        assert InterfaceDirection.from_string("OUTPUT") == InterfaceDirection.OUTPUT
        assert InterfaceDirection.from_string("Weight") == InterfaceDirection.WEIGHT
        
        with pytest.raises(ValueError):
            InterfaceDirection.from_string("invalid")


class TestDataType:
    """Tests for DataType class"""
    
    def test_creation(self):
        """Test basic DataType creation"""
        dt = DataType("INT16", 16, signed=True)
        assert dt.name == "INT16"
        assert dt.bits == 16
        assert dt.signed == True
    
    def test_validation(self):
        """Test DataType validation"""
        with pytest.raises(ValueError):
            DataType("INT8", 0)  # Invalid bit width
        
        with pytest.raises(ValueError):
            DataType("", 8)  # Empty name
    
    def test_from_string(self):
        """Test parsing from string"""
        # Integer types
        dt = DataType.from_string("INT8")
        assert dt.name == "INT8"
        assert dt.bits == 8
        assert dt.signed == True
        
        dt = DataType.from_string("UINT16")
        assert dt.name == "UINT16"
        assert dt.bits == 16
        assert dt.signed == False
        
        # Special types
        dt = DataType.from_string("BIPOLAR")
        assert dt.name == "BIPOLAR"
        assert dt.bits == 1
        assert dt.signed == True
        
        dt = DataType.from_string("BINARY")
        assert dt.name == "BINARY"
        assert dt.bits == 1
        assert dt.signed == False
        
        # Floating point
        dt = DataType.from_string("FP16")
        assert dt.name == "FP16"
        assert dt.bits == 16
        
        # Invalid format
        with pytest.raises(ValueError):
            DataType.from_string("INVALID123")
    
    def test_predefined_types(self):
        """Test predefined common types"""
        assert INT8.bits == 8
        assert INT16.bits == 16
        assert INT32.bits == 32
        assert BINARY.bits == 1
        assert BIPOLAR.bits == 1


class TestShapeUtilities:
    """Tests for shape manipulation utilities"""
    
    def test_prod(self):
        """Test product calculation"""
        assert prod((2, 3, 4)) == 24
        assert prod((5,)) == 5
        assert prod(()) == 1  # Empty shape
        assert prod((1, 1, 1)) == 1
    
    def test_shape_to_string(self):
        """Test shape string conversion"""
        assert shape_to_string((32, 64)) == "(32,64)"
        assert shape_to_string((100,)) == "(100)"
        assert shape_to_string(()) == "()"
    
    def test_parse_shape(self):
        """Test shape parsing"""
        assert parse_shape("(32,64)") == (32, 64)
        assert parse_shape("32,64") == (32, 64)
        assert parse_shape("(100)") == (100,)
        assert parse_shape("100") == (100,)
        assert parse_shape("") == ()
        assert parse_shape("()") == ()
    
    def test_shapes_compatible(self):
        """Test shape compatibility checking"""
        # Compatible shapes
        assert shapes_compatible((32, 64), (32, 64)) == True
        assert shapes_compatible((32, 1), (32, 64)) == True  # Broadcasting
        assert shapes_compatible((1, 64), (32, 64)) == True  # Broadcasting
        
        # Incompatible shapes
        assert shapes_compatible((32, 64), (64, 32)) == False
        assert shapes_compatible((32,), (32, 64)) == False  # Different dims
    
    def test_broadcast_shapes(self):
        """Test shape broadcasting"""
        assert broadcast_shapes((32, 1), (32, 64)) == (32, 64)
        assert broadcast_shapes((1, 64), (32, 64)) == (32, 64)
        assert broadcast_shapes((1, 1), (32, 64)) == (32, 64)
        
        with pytest.raises(ValueError):
            broadcast_shapes((32, 32), (64, 64))
    
    def test_tile_shape(self):
        """Test tiling calculation"""
        assert tile_shape((100, 100), (32, 32)) == (4, 4)  # Even tiling with padding
        assert tile_shape((64, 64), (32, 32)) == (2, 2)    # Even tiling
        assert tile_shape((100,), (32,)) == (4,)           # 1D tiling
        
        with pytest.raises(ValueError):
            tile_shape((100, 100), (32,))  # Mismatched dimensions
    
    def test_is_valid_tiling(self):
        """Test tiling validity"""
        assert is_valid_tiling((64, 64), (32, 32)) == True
        assert is_valid_tiling((100, 100), (32, 32)) == True
        
        assert is_valid_tiling((64, 64), (128, 32)) == False  # Block larger than tensor
        assert is_valid_tiling((64, 64), (0, 32)) == False    # Zero block size
        assert is_valid_tiling((64, 64), (32,)) == False      # Mismatched dims