############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Comprehensive tests for the new SDIM architecture with separate
InputInterface and OutputInterface classes.
"""

import pytest
from typing import List, Tuple

from brainsmith.core.dataflow.core import (
    InputDefinition, OutputDefinition, KernelDefinitionV2, KernelModelV2,
    InputInterface, OutputInterface,
    DataType, RelationType,
    fixed_tiles, parameterized_tiles
)


class TestInputInterface:
    """Test InputInterface SDIM functionality"""
    
    def test_sdim_initialization(self):
        """Test SDIM initialization patterns"""
        # Create input definition
        input_def = InputDefinition(
            name="data",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        )
        
        # Create model
        model = input_def.create_model((1024, 1024))
        
        # Default SDIM should be (1, 1)
        assert model.sdim == (1, 1)
        assert model.streaming_bandwidth == 1
        
    def test_sdim_uniform_configuration(self):
        """Test uniform SDIM configuration"""
        input_def = InputDefinition(
            name="data",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        )
        
        model = input_def.create_model((1024, 1024))
        
        # Set uniform SDIM
        model.sdim = 16
        assert model.sdim == (16, 16)
        assert model.streaming_bandwidth == 256  # 16 * 16
        
    def test_sdim_per_dimension_configuration(self):
        """Test per-dimension SDIM configuration"""
        input_def = InputDefinition(
            name="matrix",
            dtype=DataType.from_string("FP32"),
            block_dims_expr=fixed_tiles(32, 64, 128)
        )
        
        model = input_def.create_model((256, 512, 1024))
        
        # Set different SDIM per dimension
        model.sdim = (4, 8, 16)
        assert model.sdim == (4, 8, 16)
        assert model.streaming_bandwidth == 512  # 4 * 8 * 16
        
    def test_sdim_sparse_configuration(self):
        """Test sparse SDIM configuration"""
        input_def = InputDefinition(
            name="tensor",
            dtype=DataType.from_string("INT8"),
            block_dims_expr=fixed_tiles(16, 32, 64, 128)
        )
        
        model = input_def.create_model((128, 256, 512, 1024))
        
        # Set sparse SDIM (only some dimensions)
        model.sdim = {1: 4, 3: 8}
        assert model.sdim == (1, 4, 1, 8)
        assert model.streaming_bandwidth == 32  # 1 * 4 * 1 * 8
        
    def test_sdim_validation(self):
        """Test SDIM validation"""
        input_def = InputDefinition(
            name="data",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        )
        
        model = input_def.create_model((1024, 1024))
        
        # Invalid: exceeds block dimensions
        with pytest.raises(ValueError, match="exceeds block dimension"):
            model.sdim = (128, 64)  # 128 > 64
            
        # Invalid: wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 2 dimensions"):
            model.sdim = (16, 16, 16)
            
        # Invalid: zero or negative values
        with pytest.raises(ValueError, match="must be positive"):
            model.sdim = (16, 0)
            
    def test_initiation_interval_calculation(self):
        """Test initiation interval calculation with SDIM"""
        input_def = InputDefinition(
            name="data",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        )
        
        model = input_def.create_model((256, 256))
        
        # Total blocks: (256/64) * (256/64) = 16
        # Default SDIM (1, 1): cycles_per_block = 64*64/1 = 4096
        assert model.initiation_interval == 16 * 4096
        
        # With SDIM (8, 8): cycles_per_block = 64*64/64 = 64
        model.sdim = (8, 8)
        assert model.initiation_interval == 16 * 64


class TestOutputInterface:
    """Test OutputInterface functionality"""
    
    def test_output_no_sdim_configuration(self):
        """Test that outputs cannot have SDIM configured"""
        output_def = OutputDefinition(
            name="result",
            dtype=DataType.from_string("FP32"),
            block_dims_expr=fixed_tiles(64, 64)
        )
        
        model = output_def.create_model((1024, 1024))
        
        # Should not have sdim property
        assert not hasattr(model, 'sdim')
        
        # Should have streaming_rate property
        assert hasattr(model, 'streaming_rate')
        # Default rate (no computation info)
        assert model.streaming_rate == 1
        
    def test_output_streaming_rate(self):
        """Test output streaming rate property"""
        output_def = OutputDefinition(
            name="result",
            dtype=DataType.from_string("FP32"),
            block_dims_expr=fixed_tiles(32, 32)
        )
        
        model = output_def.create_model((512, 512))
        
        # Simulate setting computed rate
        model._streaming_rate = 64
        assert model.streaming_rate == 64


class TestKernelModelV2:
    """Test KernelModelV2 with separate input/output handling"""
    
    def test_kernel_creation(self):
        """Test creating kernel with inputs and outputs"""
        kernel_def = KernelDefinitionV2(name="test_kernel")
        
        # Add inputs
        kernel_def.add_input(InputDefinition(
            name="a",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_input(InputDefinition(
            name="b",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        
        # Add output
        kernel_def.add_output(OutputDefinition(
            name="c",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        
        # Create models
        a_model = kernel_def.get_input("a").create_model((1024, 1024))
        b_model = kernel_def.get_input("b").create_model((1024, 1024))
        c_model = kernel_def.get_output("c").create_model((1024, 1024))
        
        kernel = KernelModelV2(
            input_models=[a_model, b_model],
            output_models=[c_model],
            definition=kernel_def
        )
        
        assert len(kernel.input_models) == 2
        assert len(kernel.output_models) == 1
        assert kernel.get_input_model("a") == a_model
        assert kernel.get_output_model("c") == c_model
        
    def test_configure_sdim_inputs_only(self):
        """Test that configure_sdim only accepts input names"""
        kernel_def = KernelDefinitionV2(name="test")
        
        kernel_def.add_input(InputDefinition(
            name="input",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_output(OutputDefinition(
            name="output",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        
        input_model = kernel_def.get_input("input").create_model((1024, 1024))
        output_model = kernel_def.get_output("output").create_model((1024, 1024))
        
        kernel = KernelModelV2(
            input_models=[input_model],
            output_models=[output_model],
            definition=kernel_def
        )
        
        # Configuring input should work
        kernel.configure_sdim({"input": 16})
        assert kernel.get_input_model("input").sdim == (16, 16)
        
        # Configuring output should fail
        with pytest.raises(ValueError, match="Cannot configure SDIM for output"):
            kernel.configure_sdim({"output": 16})
            
    def test_sdim_relationship_propagation(self):
        """Test SDIM propagation through relationships"""
        kernel_def = KernelDefinitionV2(name="elementwise")
        
        # Two inputs that must stream together
        kernel_def.add_input(InputDefinition(
            name="a",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_input(InputDefinition(
            name="b",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        
        # EQUAL relationship
        kernel_def.add_relationship("a", "b", RelationType.EQUAL)
        
        # Create kernel
        a_model = kernel_def.get_input("a").create_model((1024, 1024))
        b_model = kernel_def.get_input("b").create_model((1024, 1024))
        
        kernel = KernelModelV2(
            input_models=[a_model, b_model],
            output_models=[],
            definition=kernel_def
        )
        
        # Configure only 'a'
        kernel.configure_sdim({"a": 16})
        
        # 'b' should get same SDIM through relationship
        assert kernel.get_input_model("a").sdim == (16, 16)
        assert kernel.get_input_model("b").sdim == (16, 16)
        
    def test_dependent_relationship_propagation(self):
        """Test DEPENDENT relationship propagation"""
        kernel_def = KernelDefinitionV2(name="matmul")
        
        # Matrix multiply inputs
        kernel_def.add_input(InputDefinition(
            name="A",  # M x K
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 32)
        ))
        kernel_def.add_input(InputDefinition(
            name="B",  # K x N
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 128)
        ))
        
        # K dimensions must match
        kernel_def.add_relationship(
            "A", "B", RelationType.DEPENDENT,
            source_dim=1, target_dim=0,
            dependency_type="copy"
        )
        
        # Create kernel
        a_model = kernel_def.get_input("A").create_model((512, 256))
        b_model = kernel_def.get_input("B").create_model((256, 1024))
        
        kernel = KernelModelV2(
            input_models=[a_model, b_model],
            output_models=[],
            definition=kernel_def
        )
        
        # Configure A with different SDIM per dimension
        kernel.configure_sdim({"A": (8, 16)})
        
        # B should get constrained SDIM
        assert kernel.get_input_model("A").sdim == (8, 16)
        assert kernel.get_input_model("B").sdim == (16, 1)  # dim 0 copied from A dim 1
        
    def test_get_sdim_parameters(self):
        """Test exposing configurable SDIM parameters"""
        kernel_def = KernelDefinitionV2(name="complex")
        
        # Three inputs with relationships
        kernel_def.add_input(InputDefinition(
            name="x",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_input(InputDefinition(
            name="y",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_input(InputDefinition(
            name="z",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 32, 32)
        ))
        
        # x and y must stream together
        kernel_def.add_relationship("x", "y", RelationType.EQUAL)
        
        # Create kernel
        x_model = kernel_def.get_input("x").create_model((1024, 1024))
        y_model = kernel_def.get_input("y").create_model((1024, 1024))
        z_model = kernel_def.get_input("z").create_model((512, 512, 512))
        
        kernel = KernelModelV2(
            input_models=[x_model, y_model, z_model],
            output_models=[],
            definition=kernel_def
        )
        
        # Get exposed parameters
        params = kernel.get_sdim_parameters()
        
        # Should have x and z (y follows x)
        assert "x" in params
        assert "z" in params
        assert "y" not in params  # Constrained by relationship
        
        # Check parameter info
        assert params["x"].total_dimensions == 2
        assert params["x"].free_dimensions == [0, 1]
        assert params["z"].total_dimensions == 3
        
    def test_compute_output_rates(self):
        """Test computing output streaming rates"""
        kernel_def = KernelDefinitionV2(name="transform")
        
        kernel_def.add_input(InputDefinition(
            name="input",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_output(OutputDefinition(
            name="output",
            dtype=DataType.from_string("FP32"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        
        input_model = kernel_def.get_input("input").create_model((1024, 1024))
        output_model = kernel_def.get_output("output").create_model((1024, 1024))
        
        kernel = KernelModelV2(
            input_models=[input_model],
            output_models=[output_model],
            definition=kernel_def
        )
        
        # Configure input SDIM
        kernel.configure_sdim({"input": 16})
        
        # Compute output rates (mock implementation)
        kernel.compute_output_rates()
        
        # In a real implementation, this would be based on kernel pattern
        # For now, just verify the method exists and runs
        assert hasattr(kernel, 'compute_output_rates')


class TestKernelDefinitionV2:
    """Test KernelDefinitionV2 functionality"""
    
    def test_add_input_output(self):
        """Test adding inputs and outputs"""
        kernel_def = KernelDefinitionV2(name="test")
        
        # Add inputs
        input1 = InputDefinition(
            name="in1",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64)
        )
        input2 = InputDefinition(
            name="in2",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64)
        )
        
        kernel_def.add_input(input1)
        kernel_def.add_input(input2)
        
        # Add output
        output = OutputDefinition(
            name="out",
            dtype=DataType.from_string("FP32"),
            block_dims_expr=fixed_tiles(64)
        )
        kernel_def.add_output(output)
        
        assert len(kernel_def.input_definitions) == 2
        assert len(kernel_def.output_definitions) == 1
        assert kernel_def.get_input("in1") == input1
        assert kernel_def.get_output("out") == output
        
    def test_duplicate_names(self):
        """Test that duplicate names are rejected"""
        kernel_def = KernelDefinitionV2(name="test")
        
        # Add first input
        kernel_def.add_input(InputDefinition(
            name="data",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64)
        ))
        
        # Duplicate input name should fail
        with pytest.raises(ValueError, match="already exists"):
            kernel_def.add_input(InputDefinition(
                name="data",
                dtype=DataType.from_string("FP32"),
                block_dims_expr=fixed_tiles(32)
            ))
            
        # Output with same name as input should fail
        with pytest.raises(ValueError, match="already used by an input"):
            kernel_def.add_output(OutputDefinition(
                name="data",
                dtype=DataType.from_string("FP32"),
                block_dims_expr=fixed_tiles(64)
            ))
            
    def test_relationship_validation(self):
        """Test relationship validation"""
        kernel_def = KernelDefinitionV2(name="test")
        
        kernel_def.add_input(InputDefinition(
            name="a",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64)
        ))
        kernel_def.add_output(OutputDefinition(
            name="b",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64)
        ))
        
        # Valid relationship
        kernel_def.add_relationship("a", "b", RelationType.EQUAL)
        
        # Invalid - nonexistent interfaces
        with pytest.raises(ValueError, match="not found"):
            kernel_def.add_relationship("a", "c", RelationType.EQUAL)
            
        with pytest.raises(ValueError, match="not found"):
            kernel_def.add_relationship("c", "b", RelationType.EQUAL)


class TestEndToEndScenarios:
    """Test complete scenarios with the new architecture"""
    
    def test_convolution_scenario(self):
        """Test convolution with weights as inputs"""
        kernel_def = KernelDefinitionV2(name="conv2d")
        
        # Feature maps
        kernel_def.add_input(InputDefinition(
            name="feature_maps",  # [C_in, H, W]
            dtype=DataType.from_string("UINT8"),
            block_dims_expr=fixed_tiles(32, 14, 14)
        ))
        
        # Convolution weights (not WEIGHT type)
        kernel_def.add_input(InputDefinition(
            name="conv_weights",  # [C_out, C_in, K, K]
            dtype=DataType.from_string("INT8"),
            block_dims_expr=fixed_tiles(64, 32, 3, 3)
        ))
        
        # Bias
        kernel_def.add_input(InputDefinition(
            name="bias",  # [C_out]
            dtype=DataType.from_string("INT32"),
            block_dims_expr=fixed_tiles(64)
        ))
        
        # Output
        kernel_def.add_output(OutputDefinition(
            name="output",  # [C_out, H, W]
            dtype=DataType.from_string("INT32"),
            block_dims_expr=fixed_tiles(64, 14, 14)
        ))
        
        # Relationships
        kernel_def.add_relationship(
            "feature_maps", "conv_weights", RelationType.EQUAL,
            source_dim=0, target_dim=1,
            description="Input channels must match"
        )
        kernel_def.add_relationship(
            "conv_weights", "bias", RelationType.EQUAL,
            source_dim=0, target_dim=0,
            description="Output channels must match"
        )
        
        # Create models
        fm_model = kernel_def.get_input("feature_maps").create_model((256, 224, 224))
        cw_model = kernel_def.get_input("conv_weights").create_model((512, 256, 3, 3))
        bias_model = kernel_def.get_input("bias").create_model((512,))
        out_model = kernel_def.get_output("output").create_model((512, 224, 224))
        
        kernel = KernelModelV2(
            input_models=[fm_model, cw_model, bias_model],
            output_models=[out_model],
            definition=kernel_def
        )
        
        # Configure streaming
        kernel.configure_sdim({
            "feature_maps": [8, 1, 1],     # 8 channels at a time
            "conv_weights": [16, 8, 3, 3]  # Constrained by relationship
        })
        
        # Verify configuration
        assert kernel.get_input_model("feature_maps").sdim == (8, 1, 1)
        assert kernel.get_input_model("conv_weights").sdim == (16, 8, 3, 3)
        assert kernel.get_input_model("bias").sdim == (16,)  # From relationship
        
        # Output has no configurable SDIM
        assert not hasattr(kernel.get_output_model("output"), "sdim")