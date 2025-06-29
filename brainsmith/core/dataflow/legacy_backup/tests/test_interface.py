############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for Interface class"""

import pytest
from brainsmith.core.dataflow.types import (
    InterfaceDirection, DataType, INT16, INT32
)
from brainsmith.core.dataflow.interface import Interface


class TestInterfaceCreation:
    """Tests for Interface creation and validation"""
    
    def test_basic_creation(self):
        """Test basic interface creation"""
        intf = Interface(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(32, 512),
            block_dims=(32, 512),  # Must match tensor dimensions
            stream_dims=(1, 16)     # Must match tensor dimensions
        )
        
        assert intf.name == "input"
        assert intf.direction == InterfaceDirection.INPUT
        assert intf.dtype == INT16
        assert intf.tensor_dims == (32, 512)
        assert intf.block_dims == [(32, 512)]  # Normalized to list
        assert intf.stream_dims == (1, 16)
        assert intf.ipar == 16  # 1 * 16
        assert intf.n_phases == 1
        assert not intf.is_csdf
    
    def test_csdf_creation(self):
        """Test CSDF interface with ragged blocks"""
        intf = Interface(
            name="ragged",
            direction=InterfaceDirection.OUTPUT,
            dtype=INT32,
            tensor_dims=(100,),
            block_dims=[(32,), (32,), (36,)],  # Ragged tiling
            stream_dims=(8,)
        )
        
        assert intf.n_phases == 3
        assert intf.is_csdf
        assert intf.rate_pattern == [4, 4, 4]  # 32/8, 32/8, 36/8 = 4.5 -> 4 (integer division)
        assert intf.ii_pattern == [4, 4, 5]  # ceil(36/8) = 5
    
    def test_default_values(self):
        """Test default parameter values"""
        intf = Interface(
            name="test",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType("INT8", 8),
            tensor_dims=(64, 64),
            block_dims=(64, 64)
        )
        
        assert intf.stream_dims == (1, 1)  # Default based on tensor dims
        assert intf.skip_prob == [0.0]   # Default no sparsity
        assert intf.optional == False    # Default required
    
    def test_validation_errors(self):
        """Test validation of invalid inputs"""
        # Wrong types
        with pytest.raises(TypeError):
            Interface(
                name=123,  # Should be string
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(32,),
                block_dims=(32,)
            )
        
        # Dimension mismatch
        with pytest.raises(ValueError):
            Interface(
                name="test",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(32, 64),  # 2D
                block_dims=(32,)       # 1D - mismatch!
            )
        
        # Stream larger than block
        with pytest.raises(ValueError):
            Interface(
                name="test",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(64,),
                block_dims=(32,),
                stream_dims=(64,)  # Larger than block!
            )
        
        # Invalid tiling
        with pytest.raises(ValueError):
            Interface(
                name="test",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(64,),
                block_dims=(128,)  # Block larger than tensor!
            )


class TestInterfaceProperties:
    """Tests for Interface computed properties"""
    
    def test_ipar_calculation(self):
        """Test interface parallelism calculation"""
        intf = Interface(
            name="test",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(32, 64, 128),
            block_dims=(32, 64, 128),
            stream_dims=(4, 8, 2)
        )
        
        assert intf.ipar == 4 * 8 * 2  # 64
    
    def test_rate_pattern(self):
        """Test CSDF rate pattern calculation"""
        # Uniform blocks
        intf = Interface(
            name="uniform",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(128,),
            block_dims=(32,),
            stream_dims=(8,)
        )
        assert intf.rate_pattern == [4]  # 32/8
        
        # Ragged blocks
        intf = Interface(
            name="ragged",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(100,),
            block_dims=[(40,), (40,), (20,)],
            stream_dims=(10,)
        )
        assert intf.rate_pattern == [4, 4, 2]
    
    def test_bandwidth_calculation(self):
        """Test bandwidth calculation"""
        intf = Interface(
            name="test",
            direction=InterfaceDirection.INPUT,
            dtype=DataType("INT8", 8),
            tensor_dims=(64,),
            block_dims=(64,),
            stream_dims=(16,)
        )
        
        assert intf.bandwidth_bits == 16 * 8  # 128 bits/cycle
    
    def test_tokens_per_inference(self):
        """Test token counting"""
        # Single block covers tensor
        intf = Interface(
            name="test",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(64, 64),
            block_dims=(64, 64),
            stream_dims=(1, 1)
        )
        assert intf.tokens_per_inference == 1
        
        # Multiple blocks needed
        intf = Interface(
            name="test",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(128, 128),
            block_dims=(32, 32),
            stream_dims=(1, 1)
        )
        assert intf.tokens_per_inference == 16  # 4x4 blocks


class TestInterfaceSparsity:
    """Tests for sparsity features"""
    
    def test_skip_prob_validation(self):
        """Test skip probability validation"""
        # Valid skip probs
        intf = Interface(
            name="sparse",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(64,),
            block_dims=[(32,), (32,)],
            stream_dims=(8,),
            skip_prob=[0.5, 0.3]
        )
        assert intf.skip_prob == [0.5, 0.3]
        
        # Wrong length
        with pytest.raises(ValueError):
            Interface(
                name="sparse",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(64,),
                block_dims=[(32,), (32,)],
                stream_dims=(8,),
                skip_prob=[0.5]  # Should be length 2
            )
        
        # Invalid probability
        with pytest.raises(ValueError):
            Interface(
                name="sparse",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(64,),
                block_dims=(32,),
                stream_dims=(8,),
                skip_prob=[1.5]  # > 1.0
            )
    
    def test_effective_rate(self):
        """Test effective rate with sparsity"""
        # No sparsity
        intf = Interface(
            name="dense",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(64,),
            block_dims=(32,),
            stream_dims=(8,)
        )
        assert intf.effective_rate() == 4.0  # 32/8
        
        # With sparsity
        intf = Interface(
            name="sparse",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(64,),
            block_dims=(32,),
            stream_dims=(8,),
            skip_prob=[0.5]
        )
        assert intf.effective_rate() == 2.0  # 4.0 * (1 - 0.5)
        
        # CSDF with varying sparsity
        intf = Interface(
            name="csdf_sparse",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(100,),
            block_dims=[(40,), (40,), (20,)],
            stream_dims=(10,),
            skip_prob=[0.0, 0.5, 0.8]
        )
        # Phase 0: 4 * 1.0 = 4
        # Phase 1: 4 * 0.5 = 2
        # Phase 2: 2 * 0.2 = 0.4
        # Average: (4 + 2 + 0.4) / 3 = 2.133...
        assert abs(intf.effective_rate() - 2.133333) < 0.001


class TestInterfaceConnections:
    """Tests for interface connection validation"""
    
    def test_valid_connections(self):
        """Test valid interface connections"""
        output = Interface(
            name="out",
            direction=InterfaceDirection.OUTPUT,
            dtype=INT16,
            tensor_dims=(32, 64),
            block_dims=(32, 64),
            stream_dims=(1, 8)
        )
        
        input = Interface(
            name="in",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(32, 64),
            block_dims=(32, 64),
            stream_dims=(1, 4)  # Different stream dims OK
        )
        
        # Should not raise
        output.validate_connection(input)
    
    def test_invalid_connections(self):
        """Test invalid interface connections"""
        output = Interface(
            name="out",
            direction=InterfaceDirection.OUTPUT,
            dtype=INT16,
            tensor_dims=(32, 64),
            block_dims=(32, 64)
        )
        
        # Wrong direction
        output2 = Interface(
            name="out2",
            direction=InterfaceDirection.OUTPUT,
            dtype=INT16,
            tensor_dims=(32, 64),
            block_dims=(32, 64)
        )
        
        with pytest.raises(ValueError, match="Cannot connect"):
            output.validate_connection(output2)
        
        # Wrong tensor dims
        input_wrong_dims = Interface(
            name="in",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(64, 32),  # Swapped!
            block_dims=(64, 32)
        )
        
        with pytest.raises(ValueError, match="Tensor dimension mismatch"):
            output.validate_connection(input_wrong_dims)
        
        # Wrong data type
        input_wrong_type = Interface(
            name="in",
            direction=InterfaceDirection.INPUT,
            dtype=INT32,  # Different type!
            tensor_dims=(32, 64),
            block_dims=(32, 64)
        )
        
        with pytest.raises(ValueError, match="Data type mismatch"):
            output.validate_connection(input_wrong_type)


class TestInterfacePhases:
    """Tests for CSDF phase handling"""
    
    def test_get_phase_info(self):
        """Test phase information retrieval"""
        intf = Interface(
            name="csdf",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(100,),
            block_dims=[(40,), (40,), (20,)],
            stream_dims=(10,),
            skip_prob=[0.0, 0.5, 0.8]
        )
        
        # Check each phase
        phase0 = intf.get_phase_info(0)
        assert phase0["block_dims"] == (40,)
        assert phase0["block_size"] == 40
        assert phase0["rate"] == 4
        assert phase0["ii"] == 4
        assert phase0["skip_prob"] == 0.0
        
        phase2 = intf.get_phase_info(2)
        assert phase2["block_dims"] == (20,)
        assert phase2["block_size"] == 20
        assert phase2["rate"] == 2
        assert phase2["ii"] == 2
        assert phase2["skip_prob"] == 0.8
        
        # Invalid phase
        with pytest.raises(ValueError):
            intf.get_phase_info(3)