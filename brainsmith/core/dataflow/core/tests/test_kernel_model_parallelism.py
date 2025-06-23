############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for KernelModel parallelism application and propagation"""

import pytest
from typing import List

from brainsmith.core.dataflow.core.kernel_model import KernelModel
from brainsmith.core.dataflow.core.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.interface_model import InterfaceModel
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection
from brainsmith.core.dataflow.core.relationships import DimensionRelationship, RelationType
from brainsmith.core.dataflow.core.base import ParameterBinding


class TestKernelModelParallelism:
    """Test KernelModel parallelism application and propagation"""
    
    def test_apply_parallelism_simple(self):
        """Test applying parallelism to specific interfaces"""
        # Create interface definitions
        input_def = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        output_def = InterfaceDefinition(
            name="output",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT32")
        )
        
        # Create kernel definition
        kernel_def = KernelDefinition(
            name="simple_kernel",
            interface_definitions=[input_def, output_def]
        )
        
        # Create interface models
        input_model = input_def.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        output_model = output_def.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        # Create kernel model
        kernel_model = KernelModel(
            interface_models=[input_model, output_model],
            definition=kernel_def
        )
        
        # Apply parallelism
        kernel_model.apply_parallelism({
            "input": 8,
            "output": 4
        })
        
        # Verify parallelism was applied
        assert input_model.ipar == 8
        assert output_model.ipar == 4
        assert input_model.stream_dims == (1, 8, 1, 1)
        assert output_model.stream_dims == (1, 4, 1, 1)
    
    def test_parallelism_propagation(self):
        """Test parallelism propagation through relationships"""
        # Create interface definitions
        input_def = InterfaceDefinition(
            name="A",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        weights_def = InterfaceDefinition(
            name="B",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType.from_string("INT8")
        )
        output_def = InterfaceDefinition(
            name="C",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT32")
        )
        
        # Create kernel definition with relationships
        kernel_def = KernelDefinition(
            name="matmul",
            interface_definitions=[input_def, weights_def, output_def]
        )
        
        # Add relationship: A.cols == B.rows
        kernel_def.add_relationship(
            "A", "B", RelationType.EQUAL,
            source_dim=1, target_dim=0,
            description="Matrix multiplication constraint"
        )
        
        # Create interface models
        input_model = input_def.create_model(
            tensor_dims=(128, 256),
            block_dims=(32, 64)
        )
        weights_model = weights_def.create_model(
            tensor_dims=(256, 512),
            block_dims=(64, 128)
        )
        output_model = output_def.create_model(
            tensor_dims=(128, 512),
            block_dims=(32, 128)
        )
        
        # Create kernel model
        kernel_model = KernelModel(
            interface_models=[input_model, weights_model, output_model],
            definition=kernel_def
        )
        
        # Apply parallelism only to input - should propagate to weights
        kernel_model.apply_parallelism({"A": 8})
        
        # Check propagation occurred
        assert input_model.ipar == 8
        assert weights_model.ipar == 8  # Propagated due to EQUAL relationship
        assert output_model.ipar == 1   # Not affected
    
    def test_get_parallelism_state(self):
        """Test getting current parallelism state"""
        # Create simple kernel
        input_def = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        output_def = InterfaceDefinition(
            name="output",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT8")
        )
        
        kernel_def = KernelDefinition(
            name="test",
            interface_definitions=[input_def, output_def]
        )
        
        input_model = input_def.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        output_model = output_def.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        kernel_model = KernelModel(
            interface_models=[input_model, output_model],
            definition=kernel_def
        )
        
        # Apply parallelism
        kernel_model.apply_parallelism({
            "input": 4,
            "output": 8
        })
        
        # Get state
        state = kernel_model.get_parallelism_state()
        assert state == {"input": 4, "output": 8}
    
    def test_performance_update_with_parallelism(self):
        """Test that performance metrics update with parallelism changes"""
        # Create kernel with definition for dtype
        input_def = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT16")
        )
        
        kernel_def = KernelDefinition(
            name="test",
            interface_definitions=[input_def]
        )
        
        input_model = input_def.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        kernel_model = KernelModel(
            interface_models=[input_model],
            definition=kernel_def,
            clock_freq_mhz=200.0
        )
        
        # Get initial bandwidth
        bw1 = kernel_model.bandwidth_requirements()
        
        # Apply parallelism
        kernel_model.apply_parallelism({"input": 8})
        
        # Get updated bandwidth
        bw2 = kernel_model.bandwidth_requirements()
        
        # Bandwidth should increase with parallelism
        assert bw2["input"] > bw1["input"]
        assert bw2["input"] == 8 * bw1["input"]
    
    def test_clock_frequency_update(self):
        """Test updating clock frequency"""
        input_def = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        
        kernel_def = KernelDefinition(
            name="test",
            interface_definitions=[input_def]
        )
        
        input_model = input_def.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        kernel_model = KernelModel(
            interface_models=[input_model],
            definition=kernel_def,
            clock_freq_mhz=100.0,
            parameter_binding=ParameterBinding({"total_operations": 1000000})
        )
        
        # Get initial metrics
        kernel_model.apply_parallelism({"input": 4})
        throughput1 = kernel_model.throughput_fps()
        
        # Update clock frequency
        kernel_model.update_clock_frequency(200.0)
        throughput2 = kernel_model.throughput_fps()
        
        # Throughput should double with doubled frequency
        assert abs(throughput2 - 2 * throughput1) < 0.01
    
    def test_resource_estimation_with_parallelism(self):
        """Test resource estimation scales with parallelism"""
        input_def = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        weights_def = InterfaceDefinition(
            name="weights",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType.from_string("INT8")
        )
        
        kernel_def = KernelDefinition(
            name="test",
            interface_definitions=[input_def, weights_def]
        )
        
        input_model = input_def.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        weights_model = weights_def.create_model(
            tensor_dims=(64, 64, 3, 3),
            block_dims=(16, 16, 3, 3)
        )
        
        kernel_model = KernelModel(
            interface_models=[input_model, weights_model],
            definition=kernel_def
        )
        
        # Get resources with iPar=1
        resources1 = kernel_model.estimate_resources()
        
        # Apply parallelism
        kernel_model.apply_parallelism({
            "input": 8,
            "weights": 8
        })
        
        # Get resources with iPar=8
        resources2 = kernel_model.estimate_resources()
        
        # Resources should scale with parallelism
        assert resources2["LUT"] > resources1["LUT"]
        assert resources2["FF"] > resources1["FF"]
        assert resources2["DSP"] > resources1["DSP"]  # Weight interface uses DSPs
    
    def test_parallelism_with_complex_relationships(self):
        """Test parallelism with multiple related interfaces"""
        # Create a more complex kernel (e.g., convolution)
        ifmap_def = InterfaceDefinition(
            name="ifmap",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("UINT8")
        )
        weights_def = InterfaceDefinition(
            name="weights",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType.from_string("INT8")
        )
        bias_def = InterfaceDefinition(
            name="bias",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType.from_string("INT32")
        )
        ofmap_def = InterfaceDefinition(
            name="ofmap",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT16")
        )
        
        kernel_def = KernelDefinition(
            name="conv2d",
            interface_definitions=[ifmap_def, weights_def, bias_def, ofmap_def]
        )
        
        # Add relationships
        kernel_def.add_relationship(
            "weights", "ifmap", RelationType.EQUAL,
            source_dim=1, target_dim=1,
            description="Input channels match"
        )
        kernel_def.add_relationship(
            "ofmap", "weights", RelationType.EQUAL,
            source_dim=1, target_dim=0,
            description="Output channels match"
        )
        kernel_def.add_relationship(
            "bias", "ofmap", RelationType.EQUAL,
            source_dim=0, target_dim=1,
            description="Bias size matches output channels"
        )
        
        # Create models
        ifmap_model = ifmap_def.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 64, 28, 28)
        )
        weights_model = weights_def.create_model(
            tensor_dims=(128, 64, 3, 3),
            block_dims=(16, 64, 3, 3)
        )
        bias_model = bias_def.create_model(
            tensor_dims=(128,),
            block_dims=(16,)
        )
        ofmap_model = ofmap_def.create_model(
            tensor_dims=(1, 128, 222, 222),
            block_dims=(1, 16, 28, 28)
        )
        
        kernel_model = KernelModel(
            interface_models=[ifmap_model, weights_model, bias_model, ofmap_model],
            definition=kernel_def
        )
        
        # Apply parallelism to weights - should propagate
        kernel_model.apply_parallelism({"weights": 16})
        
        # Check propagation through relationships
        assert weights_model.ipar == 16
        # Note: Current implementation only propagates to interfaces with ipar=1
        # More sophisticated propagation could be added
    
    def test_cache_invalidation_on_parallelism_change(self):
        """Test that caches are properly invalidated when parallelism changes"""
        input_def = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        
        kernel_def = KernelDefinition(
            name="test",
            interface_definitions=[input_def]
        )
        
        input_model = input_def.create_model(
            tensor_dims=(1, 64, 224, 224),
            block_dims=(1, 32, 28, 28)
        )
        
        kernel_model = KernelModel(
            interface_models=[input_model],
            definition=kernel_def
        )
        
        # Calculate some metrics to populate caches
        _ = kernel_model.bandwidth_requirements()
        _ = kernel_model.estimate_resources()
        assert len(kernel_model._cached_metrics) > 0
        assert len(input_model._cached_metrics) > 0
        
        # Apply parallelism should clear caches
        kernel_model.apply_parallelism({"input": 4})
        
        assert len(kernel_model._cached_metrics) == 0
        assert len(input_model._cached_metrics) == 0


if __name__ == "__main__":
    pytest.main([__file__])