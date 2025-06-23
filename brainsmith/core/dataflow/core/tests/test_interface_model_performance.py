############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for InterfaceModel performance metrics with dynamic stream_dims"""

import pytest
from typing import Tuple

from brainsmith.core.dataflow.core.interface_model import InterfaceModel
from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection


class TestInterfaceModelPerformance:
    """Test performance metrics with dynamic stream dimensions"""
    
    def test_rate_pattern_with_dynamic_ipar(self):
        """Test rate pattern calculation with dynamic iPar"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # With iPar = 1
        model.ipar = 1
        rate_1 = model.rate_pattern[0]
        
        # With iPar = 8
        model.ipar = 8
        rate_8 = model.rate_pattern[0]
        
        # Higher parallelism should reduce rate
        assert rate_8 < rate_1
        assert rate_1 == 32 * 28 * 28  # Full block
        assert rate_8 == 32 * 28 * 28 // 8  # Divided by parallelism
    
    def test_ii_pattern_with_dynamic_ipar(self):
        """Test initiation interval with dynamic iPar"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # With iPar = 1
        model.ipar = 1
        ii_1 = model.ii_pattern[0]
        
        # With iPar = 8
        model.ipar = 8
        ii_8 = model.ii_pattern[0]
        
        # Higher parallelism should reduce initiation interval
        assert ii_8 < ii_1
        assert ii_1 == 32 * 28 * 28  # Cycles for full block
        assert ii_8 == 32 * 28 * 28 // 8  # Reduced by parallelism
    
    def test_bandwidth_with_dynamic_ipar(self):
        """Test bandwidth calculations with dynamic iPar"""
        # Create with definition for dtype info
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        
        model = idef.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # With iPar = 1
        model.ipar = 1
        bw_bits_1 = model.bandwidth_bits
        bw_bytes_1 = model.bandwidth_bytes
        
        # With iPar = 8
        model.ipar = 8
        bw_bits_8 = model.bandwidth_bits
        bw_bytes_8 = model.bandwidth_bytes
        
        # Bandwidth should scale with parallelism
        assert bw_bits_8 == 8 * bw_bits_1
        assert bw_bytes_8 == 8 * bw_bytes_1
        assert bw_bits_1 == 8  # INT8 = 8 bits
        assert bw_bits_8 == 64  # 8 * 8 bits
    
    def test_effective_bandwidth_with_ipar(self):
        """Test effective bandwidth calculation"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT16")
        )
        
        model = idef.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # Set iPar and utilization
        model.ipar = 4
        model.actual_utilization = 0.8
        
        # Calculate effective bandwidth at 200MHz
        eff_bw = model.effective_bandwidth(clock_freq_mhz=200.0)
        
        # Should be: 4 * 16 bits * 200MHz * 0.8 / 8 bits/byte
        # Result already in MB/s
        expected = 4 * 16 * 200 * 0.8 / 8  # = 1280 MB/s
        assert abs(eff_bw - expected) < 0.01
    
    def test_performance_metrics_comprehensive(self):
        """Test comprehensive performance metrics with dynamic iPar"""
        idef = InterfaceDefinition(
            name="weights",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType.from_string("INT8")
        )
        
        model = idef.create_model(
            tensor_dims=(128, 64, 3, 3),
            block_dims=(16, 16, 3, 3)
        )
        
        # Set iPar
        model.ipar = 16
        
        # Get comprehensive metrics
        metrics = model.calculate_performance_metrics()
        
        # Verify stream dimensions were calculated
        assert metrics["stream_dims"] == (16, 1, 1, 1)
        assert metrics["interface_parallelism"] == 16
        
        # Verify bandwidth calculations
        assert metrics["bandwidth_bits_per_cycle"] == 16 * 8  # iPar * INT8
        assert metrics["bandwidth_bytes_per_cycle"] == 16.0
        
        # Verify rate and II patterns
        assert len(metrics["rate_pattern"]) == 1
        assert len(metrics["ii_pattern"]) == 1
        
        # Rate should be block_size / iPar
        block_size = 16 * 16 * 3 * 3
        assert metrics["rate_pattern"][0] == block_size // 16
    
    def test_csdf_with_dynamic_ipar(self):
        """Test CSDF phases with dynamic iPar"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=[(1, 32, 28, 28), (1, 16, 14, 14)],  # Two phases
            skip_prob=[0.1, 0.2]
        )
        
        # Set iPar
        model.ipar = 8
        
        # Should use first phase for stream dim calculation
        assert model.stream_dims == (1, 8, 1, 1)
        
        # Get phase info
        phase0 = model.get_phase_info(0)
        phase1 = model.get_phase_info(1)
        
        # Both phases should use same stream dims but different rates
        assert phase0["rate"] == (32 * 28 * 28) // 8
        assert phase1["rate"] == (16 * 14 * 14) // 8
        
        # Effective rate should account for sparsity
        eff_rate = model.effective_rate()
        assert eff_rate < (phase0["rate"] + phase1["rate"]) / 2  # Reduced by skip
    
    def test_cache_coherency(self):
        """Test that cache invalidation works correctly"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # Set initial iPar and calculate metrics
        model.ipar = 4
        metrics1 = model.calculate_performance_metrics().copy()
        
        # Change iPar
        model.ipar = 8
        metrics2 = model.calculate_performance_metrics().copy()
        
        # Metrics should be different
        assert metrics1["interface_parallelism"] == 4
        assert metrics2["interface_parallelism"] == 8
        assert metrics1["bandwidth_bits_per_cycle"] != metrics2["bandwidth_bits_per_cycle"]
        assert metrics1["ii_pattern"] != metrics2["ii_pattern"]
    
    def test_memory_requirements_unchanged(self):
        """Test that memory requirements don't depend on iPar"""
        idef = InterfaceDefinition(
            name="buffer",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT32")
        )
        
        model = idef.create_model(
            tensor_dims=(128, 256),
            block_dims=(32, 64)
        )
        
        # Get memory requirements with different iPar values
        model.ipar = 1
        mem1 = model.calculate_memory_requirements()
        
        model.ipar = 16
        mem2 = model.calculate_memory_requirements()
        
        # Memory requirements should be the same (only depend on dimensions)
        assert mem1["tensor_memory_bytes"] == mem2["tensor_memory_bytes"]
        assert mem1["max_block_memory_bytes"] == mem2["max_block_memory_bytes"]
        
        # Stream buffer should change with iPar
        assert mem1["stream_buffer_bytes"] != mem2["stream_buffer_bytes"]
        assert mem2["stream_buffer_bytes"] == 16 * mem1["stream_buffer_bytes"]


if __name__ == "__main__":
    pytest.main([__file__])