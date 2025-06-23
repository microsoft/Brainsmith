############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for InterfaceModel iPar and stream dimension calculation"""

import pytest
from typing import Tuple

from brainsmith.core.dataflow.core.interface_model import InterfaceModel
from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection


class TestInterfaceModelIPar:
    """Test automatic stream dimension calculation from iPar"""
    
    def test_default_ipar(self):
        """Test default iPar value is 1"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        assert model.ipar == 1
        assert model.stream_dims == (1, 1, 1, 1)
    
    def test_set_ipar_simple(self):
        """Test setting iPar with simple divisible case"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # Set iPar = 8, should apply to second dimension (32)
        model.ipar = 8
        assert model.ipar == 8
        assert model.stream_dims == (1, 8, 1, 1)
    
    def test_set_ipar_multiple_dimensions(self):
        """Test iPar distribution across multiple dimensions"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 16, 16, 16)
        )
        
        # Set iPar = 64, should factor as 16 * 4
        model.ipar = 64
        assert model.ipar == 64
        assert model.stream_dims == (1, 16, 4, 1)
    
    def test_set_ipar_prime_number(self):
        """Test iPar with prime number"""
        model = InterfaceModel(
            tensor_dims=(1, 128, 224, 224),
            block_dims=(1, 64, 32, 32)
        )
        
        # Set iPar = 7 (prime), should apply to first viable dimension
        model.ipar = 7
        assert model.ipar == 7
        # 7 doesn't divide 64 evenly, so algorithm will find best fit
        assert model.stream_dims[0] == 1
        assert model.stream_dims[1] * model.stream_dims[2] * model.stream_dims[3] <= 7
    
    def test_set_stream_dims_updates_ipar(self):
        """Test that setting stream_dims updates iPar"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # Set stream dims directly
        model.stream_dims = (1, 4, 2, 1)
        assert model.ipar == 8  # 1 * 4 * 2 * 1
        assert model.stream_dims == (1, 4, 2, 1)
    
    def test_ipar_cache_invalidation(self):
        """Test that changing iPar invalidates performance caches"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # Calculate some metrics to populate cache
        _ = model.rate_pattern
        _ = model.ii_pattern
        assert len(model._cached_metrics) > 0
        
        # Change iPar should clear cache
        model.ipar = 4
        assert len(model._cached_metrics) == 0
    
    def test_ipar_affects_performance_metrics(self):
        """Test that iPar affects performance calculations"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # Get metrics with iPar = 1
        model.ipar = 1
        metrics_1 = model.calculate_performance_metrics()
        ii_1 = model.ii_pattern[0]
        
        # Get metrics with iPar = 8
        model.ipar = 8
        metrics_8 = model.calculate_performance_metrics()
        ii_8 = model.ii_pattern[0]
        
        # Higher parallelism should reduce initiation interval
        assert ii_8 < ii_1
        assert metrics_8["interface_parallelism"] == 8
        assert metrics_1["interface_parallelism"] == 1
    
    def test_ipar_with_csdf(self):
        """Test iPar with CSDF (multiple block dimensions)"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=[(1, 32, 28, 28), (1, 16, 14, 14)]  # Two phases
        )
        
        # Set iPar - should use first phase for calculation
        model.ipar = 4
        assert model.stream_dims == (1, 4, 1, 1)
    
    def test_ipar_validation(self):
        """Test iPar validation"""
        model = InterfaceModel(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # Test invalid values
        with pytest.raises(ValueError):
            model.ipar = 0
        
        with pytest.raises(ValueError):
            model.ipar = -1
    
    def test_create_model_from_definition(self):
        """Test creating model from definition preserves iPar behavior"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=[":", "params['C_TILE']", 28, 28]
        )
        
        # Create model
        model = idef.create_model(
            tensor_dims=(1, 64, 224, 224),
            parameter_binding={"C_TILE": 32}
        )
        
        # Should have derived block dims
        assert model.block_dims == [(1, 32, 28, 28)]
        
        # Should be able to set iPar
        model.ipar = 8
        assert model.stream_dims == (1, 8, 1, 1)
    
    def test_complex_ipar_factorization(self):
        """Test complex iPar factorization scenarios"""
        model = InterfaceModel(
            tensor_dims=(128, 256),
            block_dims=(32, 64)
        )
        
        # Test various iPar values
        test_cases = [
            (2, (2, 1)),      # Simple case
            (4, (4, 1)),      # Fits in first dim
            (8, (8, 1)),      # Still fits in first dim
            (16, (16, 1)),    # Half of first dim
            (32, (32, 1)),    # Full first dim
            (64, (32, 2)),    # Spills to second dim
            (128, (32, 4)),   # More in second dim
        ]
        
        for ipar, expected in test_cases:
            model.ipar = ipar
            assert model.stream_dims == expected, f"iPar={ipar}: expected {expected}, got {model.stream_dims}"


if __name__ == "__main__":
    pytest.main([__file__])