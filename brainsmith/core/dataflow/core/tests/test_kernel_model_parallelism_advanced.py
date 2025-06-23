############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for advanced KernelModel parallelism propagation"""

import pytest
from typing import List

from brainsmith.core.dataflow.core.kernel_model import KernelModel
from brainsmith.core.dataflow.core.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.interface_model import InterfaceModel
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection
from brainsmith.core.dataflow.core.relationships import DimensionRelationship, RelationType
from brainsmith.core.dataflow.core.base import ParameterBinding


class TestAdvancedParallelismPropagation:
    """Test sophisticated parallelism propagation scenarios"""
    
    def test_multiple_relationship_propagation(self):
        """Test propagation through MULTIPLE relationships"""
        # Create interface definitions
        input_def = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        expanded_def = InterfaceDefinition(
            name="expanded",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT8")
        )
        
        # Create kernel definition
        kernel_def = KernelDefinition(
            name="expander",
            interface_definitions=[input_def, expanded_def]
        )
        
        # Add MULTIPLE relationship: expanded = 4 * input
        # Note: In relationships, source -> target, so input -> expanded
        kernel_def.add_relationship(
            "input", "expanded", RelationType.MULTIPLE,
            factor=4.0,
            description="Output is 4x larger"
        )
        
        # Create models
        input_model = input_def.create_model(
            tensor_dims=(128, 256),
            block_dims=(32, 64)
        )
        expanded_model = expanded_def.create_model(
            tensor_dims=(128, 1024),  # 4x wider
            block_dims=(32, 256)
        )
        
        kernel_model = KernelModel(
            interface_models=[input_model, expanded_model],
            definition=kernel_def
        )
        
        # Apply parallelism to input
        kernel_model.apply_parallelism({"input": 8})
        
        # Expanded should have 4x parallelism
        assert input_model.ipar == 8
        assert expanded_model.ipar == 32  # 8 * 4
    
    def test_divisible_relationship_propagation(self):
        """Test propagation through DIVISIBLE relationships"""
        # Create interface definitions
        data_def = InterfaceDefinition(
            name="data",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT32")
        )
        burst_def = InterfaceDefinition(
            name="burst_size",
            direction=InterfaceDirection.CONFIG,
            dtype=DataType.from_string("INT32")
        )
        
        kernel_def = KernelDefinition(
            name="burst_aligned",
            interface_definitions=[data_def, burst_def]
        )
        
        # Add DIVISIBLE relationship
        kernel_def.add_relationship(
            "data", "burst_size", RelationType.DIVISIBLE,
            description="Data must be divisible by burst size"
        )
        
        # Create models
        data_model = data_def.create_model(
            tensor_dims=(1024,),
            block_dims=(256,)
        )
        burst_model = burst_def.create_model(
            tensor_dims=(1,),
            block_dims=(1,)
        )
        
        kernel_model = KernelModel(
            interface_models=[data_model, burst_model],
            definition=kernel_def
        )
        
        # Apply parallelism to data
        kernel_model.apply_parallelism({"data": 16})
        
        # Burst size should get a divisor of 16
        assert data_model.ipar == 16
        assert burst_model.ipar in [1, 2, 4, 8, 16]  # Valid divisors
    
    def test_conflict_resolution(self):
        """Test conflict resolution when multiple sources affect one target"""
        # Create a diamond pattern: A -> B, A -> C, B -> D, C -> D
        a_def = InterfaceDefinition("A", InterfaceDirection.INPUT, DataType.from_string("INT8"))
        b_def = InterfaceDefinition("B", InterfaceDirection.OUTPUT, DataType.from_string("INT8"))
        c_def = InterfaceDefinition("C", InterfaceDirection.OUTPUT, DataType.from_string("INT8"))
        d_def = InterfaceDefinition("D", InterfaceDirection.OUTPUT, DataType.from_string("INT8"))
        
        kernel_def = KernelDefinition(
            name="diamond",
            interface_definitions=[a_def, b_def, c_def, d_def]
        )
        
        # Add relationships
        kernel_def.add_relationship("A", "B", RelationType.EQUAL)
        kernel_def.add_relationship("A", "C", RelationType.MULTIPLE, factor=2.0)
        kernel_def.add_relationship("B", "D", RelationType.EQUAL)
        kernel_def.add_relationship("C", "D", RelationType.EQUAL)
        
        # Create models
        a_model = a_def.create_model((128, 256), (32, 64))
        b_model = b_def.create_model((128, 256), (32, 64))
        c_model = c_def.create_model((256, 256), (64, 64))
        d_model = d_def.create_model((256, 256), (64, 64))
        
        kernel_model = KernelModel(
            interface_models=[a_model, b_model, c_model, d_model],
            definition=kernel_def
        )
        
        # Apply parallelism only to A
        kernel_model.apply_parallelism({"A": 8})
        
        # Check propagation
        assert a_model.ipar == 8
        assert b_model.ipar == 8  # Equal to A
        assert c_model.ipar == 16  # 2x A
        # D should use minimum of B and C proposals to satisfy both
        assert d_model.ipar == 8  # min(8 from B, 16 from C)
    
    def test_transitive_propagation(self):
        """Test transitive propagation through relationship chains"""
        # Create chain: A -> B -> C -> D
        defs = []
        for name in ["A", "B", "C", "D"]:
            direction = InterfaceDirection.INPUT if name == "A" else InterfaceDirection.OUTPUT
            defs.append(InterfaceDefinition(name, direction, DataType.from_string("INT8")))
        
        kernel_def = KernelDefinition(
            name="chain",
            interface_definitions=defs
        )
        
        # Add chain of relationships
        kernel_def.add_relationship("A", "B", RelationType.EQUAL)
        kernel_def.add_relationship("B", "C", RelationType.EQUAL)
        kernel_def.add_relationship("C", "D", RelationType.EQUAL)
        
        # Create models
        models = []
        for def_ in defs:
            models.append(def_.create_model((128, 256), (32, 64)))
        
        kernel_model = KernelModel(
            interface_models=models,
            definition=kernel_def
        )
        
        # Apply parallelism only to A
        kernel_model.apply_parallelism({"A": 16})
        
        # Check transitive propagation
        for model in models:
            assert model.ipar == 16  # All should have same parallelism
    
    def test_partial_propagation_with_dimension_mismatch(self):
        """Test propagation when dimensions don't fully match"""
        # Create interfaces with different dimensions
        matrix_def = InterfaceDefinition(
            name="matrix",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        vector_def = InterfaceDefinition(
            name="vector",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        
        kernel_def = KernelDefinition(
            name="matmul_vector",
            interface_definitions=[matrix_def, vector_def]
        )
        
        # Add relationship: matrix.cols == vector.size
        kernel_def.add_relationship(
            "matrix", "vector", RelationType.EQUAL,
            source_dim=1, target_dim=0
        )
        
        # Create models
        matrix_model = matrix_def.create_model(
            tensor_dims=(512, 256),
            block_dims=(128, 64)  # Can parallelize rows
        )
        vector_model = vector_def.create_model(
            tensor_dims=(256,),
            block_dims=(64,)
        )
        
        kernel_model = KernelModel(
            interface_models=[matrix_model, vector_model],
            definition=kernel_def
        )
        
        # Apply parallelism to matrix
        kernel_model.apply_parallelism({"matrix": 8})
        
        # Vector should get propagated parallelism based on dimension match
        assert matrix_model.ipar == 8
        assert vector_model.ipar == 8  # Can match the column parallelism
    
    def test_no_propagation_when_target_has_parallelism(self):
        """Test that existing parallelism is not overwritten"""
        # Create simple two-interface kernel
        a_def = InterfaceDefinition("A", InterfaceDirection.INPUT, DataType.from_string("INT8"))
        b_def = InterfaceDefinition("B", InterfaceDirection.OUTPUT, DataType.from_string("INT8"))
        
        kernel_def = KernelDefinition(
            name="test",
            interface_definitions=[a_def, b_def]
        )
        
        kernel_def.add_relationship("A", "B", RelationType.EQUAL)
        
        # Create models
        a_model = a_def.create_model((128, 256), (32, 64))
        b_model = b_def.create_model((128, 256), (32, 64))
        
        kernel_model = KernelModel(
            interface_models=[a_model, b_model],
            definition=kernel_def
        )
        
        # Apply parallelism to both (B first)
        kernel_model.apply_parallelism({"B": 4, "A": 16})
        
        # B should keep its explicitly set value
        assert a_model.ipar == 16
        assert b_model.ipar == 4  # Not overwritten by propagation
    
    def test_complex_conv2d_propagation(self):
        """Test realistic conv2d parallelism propagation"""
        # Create conv2d interfaces
        ifmap_def = InterfaceDefinition("ifmap", InterfaceDirection.INPUT, 
                                       DataType.from_string("UINT8"))
        weights_def = InterfaceDefinition("weights", InterfaceDirection.WEIGHT,
                                         DataType.from_string("INT8"))
        bias_def = InterfaceDefinition("bias", InterfaceDirection.WEIGHT,
                                      DataType.from_string("INT32"))
        ofmap_def = InterfaceDefinition("ofmap", InterfaceDirection.OUTPUT,
                                       DataType.from_string("INT16"))
        
        kernel_def = KernelDefinition(
            name="conv2d",
            interface_definitions=[ifmap_def, weights_def, bias_def, ofmap_def]
        )
        
        # Add conv2d relationships
        kernel_def.add_relationship(
            "weights", "ifmap", RelationType.EQUAL,
            source_dim=1, target_dim=1,  # Input channels
            description="Input channels match"
        )
        kernel_def.add_relationship(
            "ofmap", "weights", RelationType.EQUAL,
            source_dim=1, target_dim=0,  # Output channels
            description="Output channels match"
        )
        kernel_def.add_relationship(
            "ofmap", "bias", RelationType.EQUAL,
            source_dim=1, target_dim=0,  # Output channels to bias size
            description="Output channels match bias size"
        )
        
        # Create realistic conv2d models
        ifmap_model = ifmap_def.create_model(
            tensor_dims=(1, 64, 224, 224),  # NCHW
            block_dims=(1, 64, 14, 14)
        )
        weights_model = weights_def.create_model(
            tensor_dims=(128, 64, 3, 3),  # OIHW
            block_dims=(16, 64, 3, 3)
        )
        bias_model = bias_def.create_model(
            tensor_dims=(128,),
            block_dims=(16,)
        )
        ofmap_model = ofmap_def.create_model(
            tensor_dims=(1, 128, 222, 222),
            block_dims=(1, 16, 14, 14)
        )
        
        kernel_model = KernelModel(
            interface_models=[ifmap_model, weights_model, bias_model, ofmap_model],
            definition=kernel_def
        )
        
        # Apply parallelism to output channels
        kernel_model.apply_parallelism({"ofmap": 16})
        
        # Check propagation follows conv2d constraints
        assert ofmap_model.ipar == 16       # Explicitly set
        assert weights_model.ipar == 16      # Propagated from ofmap (output channels)
        assert bias_model.ipar == 16         # Propagated from ofmap (bias size)
        
        # Note: ifmap could get parallelism propagated through the weights->ifmap
        # channel relationship, which is actually correct for conv2d when we're
        # parallelizing output channels and the implementation can handle it
        print(f"ifmap.ipar = {ifmap_model.ipar}")  # For debugging
        
        # For this test, let's just verify the core relationships were propagated
        assert weights_model.ipar == ofmap_model.ipar  # Output channel parallelism
        assert bias_model.ipar == ofmap_model.ipar     # Bias matches output channels


if __name__ == "__main__":
    pytest.main([__file__])