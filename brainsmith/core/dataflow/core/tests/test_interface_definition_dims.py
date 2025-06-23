############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for InterfaceDefinition block dimension derivation"""

import pytest
from typing import Tuple, Dict, Any

from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection


class TestBlockDimensionDerivation:
    """Test block dimension derivation functionality"""
    
    def test_default_chunking_no_layout(self):
        """Test default chunking with no ONNX layout - should return full tensor"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        
        tensor_dims = (1, 3, 224, 224)
        block_dims = idef.derive_block_dims(tensor_dims, {})
        
        assert block_dims == tensor_dims
    
    def test_default_chunking_nchw(self):
        """Test default chunking with NCHW layout"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            onnx_layout="NCHW"
        )
        
        tensor_dims = (1, 3, 224, 224)
        block_dims = idef.derive_block_dims(tensor_dims, {})
        
        # For now, default is full tensor even with layout
        assert block_dims == tensor_dims
    
    def test_expression_list_literal_integers(self):
        """Test block_dims_expr with literal integers"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=[1, 32, 16, 16]
        )
        
        tensor_dims = (1, 64, 224, 224)
        block_dims = idef.derive_block_dims(tensor_dims, {})
        
        assert block_dims == (1, 32, 16, 16)
    
    def test_expression_list_colon_syntax(self):
        """Test block_dims_expr with ':' for full dimension"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=[":", 32, ":", ":"]
        )
        
        tensor_dims = (1, 64, 224, 224)
        block_dims = idef.derive_block_dims(tensor_dims, {})
        
        assert block_dims == (1, 32, 224, 224)
    
    def test_expression_list_tensor_indexing(self):
        """Test block_dims_expr with tensor[i] syntax"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=["tensor[0]", "tensor[1] // 2", 16, 16]
        )
        
        tensor_dims = (2, 64, 224, 224)
        block_dims = idef.derive_block_dims(tensor_dims, {})
        
        assert block_dims == (2, 32, 16, 16)
    
    def test_expression_list_param_substitution(self):
        """Test block_dims_expr with parameter substitution"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=[":", "params['C_TILE']", "params['H_TILE']", "params['W_TILE']"]
        )
        
        tensor_dims = (1, 64, 224, 224)
        params = {"C_TILE": 16, "H_TILE": 14, "W_TILE": 14}
        block_dims = idef.derive_block_dims(tensor_dims, params)
        
        assert block_dims == (1, 16, 14, 14)
    
    def test_expression_list_complex_expressions(self):
        """Test block_dims_expr with complex expressions"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=[
                "tensor[0]",
                "min(tensor[1], params['MAX_C'])",
                "tensor[2] // params['SPATIAL_FACTOR']",
                "tensor[3] // params['SPATIAL_FACTOR']"
            ]
        )
        
        tensor_dims = (1, 128, 224, 224)
        params = {"MAX_C": 64, "SPATIAL_FACTOR": 4}
        block_dims = idef.derive_block_dims(tensor_dims, params)
        
        assert block_dims == (1, 64, 56, 56)
    
    def test_function_based_specification(self):
        """Test block_dims_expr with callable function"""
        def adaptive_blocking(tensor_dims: Tuple[int, ...], 
                            params: Dict[str, int],
                            config: Dict[str, Any]) -> Tuple[int, ...]:
            if config.get("high_performance", False):
                return (tensor_dims[0], 64, 32, 32)
            else:
                return (tensor_dims[0], 16, 8, 8)
        
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=adaptive_blocking
        )
        
        tensor_dims = (1, 128, 224, 224)
        
        # Test with high_performance=True
        block_dims = idef.derive_block_dims(tensor_dims, {}, {"high_performance": True})
        assert block_dims == (1, 64, 32, 32)
        
        # Test with high_performance=False
        block_dims = idef.derive_block_dims(tensor_dims, {}, {"high_performance": False})
        assert block_dims == (1, 16, 8, 8)
    
    def test_create_model_with_derived_blocks(self):
        """Test create_model automatically derives block dimensions"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=[":", "params['C_TILE']", 28, 28]
        )
        
        tensor_dims = (1, 64, 224, 224)
        params = {"C_TILE": 32}
        
        # Create model without specifying block_dims
        model = idef.create_model(
            tensor_dims=tensor_dims,
            parameter_binding=params
        )
        
        assert model.tensor_dims == tensor_dims
        # block_dims is stored as a list for CSDF support
        assert model.block_dims == [(1, 32, 28, 28)]
    
    def test_error_invalid_expression(self):
        """Test error handling for invalid expressions"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=["tensor[0]", "invalid_syntax + +", 16, 16]
        )
        
        tensor_dims = (1, 64, 224, 224)
        
        with pytest.raises(ValueError) as exc_info:
            idef.derive_block_dims(tensor_dims, {})
        
        assert "Error evaluating block dimension expression" in str(exc_info.value)
    
    def test_error_missing_parameter(self):
        """Test error handling for missing parameter"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=[":", "params['MISSING_PARAM']", 16, 16]
        )
        
        tensor_dims = (1, 64, 224, 224)
        
        with pytest.raises(ValueError) as exc_info:
            idef.derive_block_dims(tensor_dims, {})
        
        assert "MISSING_PARAM" in str(exc_info.value)
    
    def test_error_index_out_of_range(self):
        """Test error handling for tensor index out of range"""
        idef = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=["tensor[10]", 32, 16, 16]  # Index 10 out of range
        )
        
        tensor_dims = (1, 64, 224, 224)
        
        with pytest.raises(ValueError) as exc_info:
            idef.derive_block_dims(tensor_dims, {})
        
        assert "Error evaluating block dimension expression" in str(exc_info.value)
    
    def test_mixed_specification_types(self):
        """Test mixing different specification types"""
        idef = InterfaceDefinition(
            name="weights",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=[
                "params['OC_TILE']",  # Parameter
                "params['IC_TILE']",  # Parameter
                ":",                  # Full dimension
                3                     # Literal
            ]
        )
        
        tensor_dims = (128, 64, 3, 3)
        params = {"OC_TILE": 16, "IC_TILE": 8}
        block_dims = idef.derive_block_dims(tensor_dims, params)
        
        assert block_dims == (16, 8, 3, 3)


if __name__ == "__main__":
    pytest.main([__file__])