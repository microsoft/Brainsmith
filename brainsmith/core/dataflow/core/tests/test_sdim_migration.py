############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Migration tests showing the transition from old iPar/WEIGHT model
to the new SDIM architecture
"""

import pytest
from typing import Dict, Any

from brainsmith.core.dataflow.core import (
    # New imports
    InputDefinition, OutputDefinition, KernelDefinitionV2, KernelModelV2,
    InputInterface, OutputInterface,
    # Common imports
    DataType, RelationType,
    fixed_tiles, parameterized_tiles
)


class TestMigrationScenarios:
    """Test migrating common patterns to new architecture"""
    
    def test_migrate_simple_elementwise(self):
        """Migrate simple element-wise operation"""
        
        # OLD WAY (conceptual - showing what would change)
        # kernel_def = KernelDefinition(name="add")
        # kernel_def.add_interface(InterfaceDefinition(
        #     name="a",
        #     direction=InterfaceDirection.INPUT,
        #     dtype=DataType.from_string("FP16"),
        #     block_dims_expr=fixed_tiles(64, 64)
        # ))
        # kernel_def.add_interface(InterfaceDefinition(
        #     name="b",
        #     direction=InterfaceDirection.INPUT,
        #     dtype=DataType.from_string("FP16"),
        #     block_dims_expr=fixed_tiles(64, 64)
        # ))
        # kernel_def.add_interface(InterfaceDefinition(
        #     name="c",
        #     direction=InterfaceDirection.OUTPUT,
        #     dtype=DataType.from_string("FP16"),
        #     block_dims_expr=fixed_tiles(64, 64)
        # ))
        # kernel.apply_parallelism({"a": 16, "b": 16, "c": 16})  # WRONG!
        
        # NEW WAY
        kernel_def = KernelDefinitionV2(name="add")
        
        # Separate input definitions
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
        
        # Separate output definition
        kernel_def.add_output(OutputDefinition(
            name="c",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        
        # Inputs must stream together
        kernel_def.add_relationship("a", "b", RelationType.EQUAL)
        
        # Create models
        a_model = kernel_def.get_input("a").create_model((1024, 1024))
        b_model = kernel_def.get_input("b").create_model((1024, 1024))
        c_model = kernel_def.get_output("c").create_model((1024, 1024))
        
        kernel = KernelModelV2(
            input_models=[a_model, b_model],
            output_models=[c_model],
            definition=kernel_def
        )
        
        # Configure only inputs (no output configuration)
        kernel.configure_sdim({"a": 16})  # b gets 16 from relationship
        
        # Verify
        assert kernel.get_input_model("a").sdim == (16, 16)
        assert kernel.get_input_model("b").sdim == (16, 16)
        # Output has no sdim property
        assert not hasattr(kernel.get_output_model("c"), "sdim")
        
    def test_migrate_weight_interfaces(self):
        """Migrate WEIGHT interfaces to regular inputs"""
        
        # OLD WAY (conceptual)
        # kernel_def.add_interface(InterfaceDefinition(
        #     name="input",
        #     direction=InterfaceDirection.INPUT,
        #     ...
        # ))
        # kernel_def.add_interface(InterfaceDefinition(
        #     name="weights",
        #     direction=InterfaceDirection.WEIGHT,  # Special type
        #     ...
        # ))
        # kernel_def.add_interface(InterfaceDefinition(
        #     name="bias",
        #     direction=InterfaceDirection.WEIGHT,  # Special type
        #     ...
        # ))
        
        # NEW WAY - all are inputs
        kernel_def = KernelDefinitionV2(name="linear")
        
        kernel_def.add_input(InputDefinition(
            name="input",  # [batch, in_features]
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 256)
        ))
        
        kernel_def.add_input(InputDefinition(
            name="weights",  # [out_features, in_features] - just an input!
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(512, 256)
        ))
        
        kernel_def.add_input(InputDefinition(
            name="bias",  # [out_features] - just an input!
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(512)
        ))
        
        kernel_def.add_output(OutputDefinition(
            name="output",  # [batch, out_features]
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 512)
        ))
        
        # Relationships between inputs
        kernel_def.add_relationship(
            "input", "weights", RelationType.EQUAL,
            source_dim=1, target_dim=1,
            description="in_features must match"
        )
        
        kernel_def.add_relationship(
            "weights", "bias", RelationType.EQUAL,
            source_dim=0, target_dim=0,
            description="out_features must match"
        )
        
        # All inputs can have SDIM configured
        input_model = kernel_def.get_input("input").create_model((128, 1024))
        weight_model = kernel_def.get_input("weights").create_model((2048, 1024))
        bias_model = kernel_def.get_input("bias").create_model((2048,))
        output_model = kernel_def.get_output("output").create_model((128, 2048))
        
        kernel = KernelModelV2(
            input_models=[input_model, weight_model, bias_model],
            output_models=[output_model],
            definition=kernel_def
        )
        
        # Configure streaming for all inputs
        kernel.configure_sdim({
            "input": [8, 16],
            "weights": [32, 16],  # in_features constrained
            # bias gets [32] from relationship
        })
        
        assert kernel.get_input_model("weights").sdim == (32, 16)
        assert kernel.get_input_model("bias").sdim == (32,)
        
    def test_migrate_scalar_ipar_to_multidim_sdim(self):
        """Migrate from scalar iPar to multi-dimensional SDIM"""
        
        # OLD WAY (conceptual)
        # interface.ipar = 16  # Single scalar value
        # # This was interpreted as streaming 16 elements per cycle total
        
        # NEW WAY - explicit per dimension
        kernel_def = KernelDefinitionV2(name="conv")
        
        kernel_def.add_input(InputDefinition(
            name="feature_map",  # [C, H, W]
            dtype=DataType.from_string("UINT8"),
            block_dims_expr=fixed_tiles(32, 14, 14)
        ))
        
        kernel_def.add_output(OutputDefinition(
            name="output",
            dtype=DataType.from_string("UINT8"),
            block_dims_expr=fixed_tiles(32, 14, 14)
        ))
        
        fm_model = kernel_def.get_input("feature_map").create_model((256, 224, 224))
        out_model = kernel_def.get_output("output").create_model((256, 224, 224))
        
        kernel = KernelModelV2(
            input_models=[fm_model],
            output_models=[out_model],
            definition=kernel_def
        )
        
        # OLD: ipar=16 might mean different things
        # NEW: Be explicit about streaming pattern
        
        # Option 1: Stream 16 channels
        kernel.configure_sdim({"feature_map": [16, 1, 1]})
        assert fm_model.streaming_bandwidth == 16
        
        # Option 2: Stream 4x4 spatial patch
        kernel.configure_sdim({"feature_map": [1, 4, 4]})
        assert fm_model.streaming_bandwidth == 16
        
        # Option 3: Mixed streaming
        kernel.configure_sdim({"feature_map": [4, 2, 2]})
        assert fm_model.streaming_bandwidth == 16
        
        # Much clearer what's happening!
        
    def test_migrate_dependent_dimensions(self):
        """Migrate to DEPENDENT relationships for matrix multiply"""
        
        # OLD WAY would use EQUAL on entire interfaces
        # NEW WAY uses DEPENDENT for specific dimensions
        
        kernel_def = KernelDefinitionV2(name="matmul")
        
        kernel_def.add_input(InputDefinition(
            name="A",  # [M, K]
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 32)
        ))
        
        kernel_def.add_input(InputDefinition(
            name="B",  # [K, N]
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 128)
        ))
        
        kernel_def.add_output(OutputDefinition(
            name="C",  # [M, N]
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 128)
        ))
        
        # NEW: Dimension-specific dependency
        kernel_def.add_relationship(
            "A", "B", RelationType.DEPENDENT,
            source_dim=1, target_dim=0,
            dependency_type="copy",
            description="K dimensions must match"
        )
        
        # Create and configure
        a_model = kernel_def.get_input("A").create_model((512, 256))
        b_model = kernel_def.get_input("B").create_model((256, 1024))
        c_model = kernel_def.get_output("C").create_model((512, 1024))
        
        kernel = KernelModelV2(
            input_models=[a_model, b_model],
            output_models=[c_model],
            definition=kernel_def
        )
        
        # Configure with awareness of dependency
        kernel.configure_sdim({"A": [8, 16]})
        
        # B automatically gets constrained
        assert kernel.get_input_model("A").sdim == (8, 16)
        assert kernel.get_input_model("B").sdim == (16, 1)  # K dim copied
        

class TestAPICleanup:
    """Test that old API patterns are properly rejected"""
    
    def test_no_weight_interface_type(self):
        """Verify WEIGHT is not a valid interface type"""
        # In new architecture, there's no WEIGHT type
        # Everything is either INPUT or OUTPUT
        
        kernel_def = KernelDefinitionV2(name="test")
        
        # All interfaces must be explicitly input or output
        kernel_def.add_input(InputDefinition(
            name="weights",  # Even weights are inputs
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        
        assert len(kernel_def.input_definitions) == 1
        assert kernel_def.input_definitions[0].name == "weights"
        
    def test_cannot_configure_output_sdim(self):
        """Verify outputs cannot have SDIM configured"""
        kernel_def = KernelDefinitionV2(name="test")
        
        kernel_def.add_input(InputDefinition(
            name="in",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64)
        ))
        kernel_def.add_output(OutputDefinition(
            name="out",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64)
        ))
        
        in_model = kernel_def.get_input("in").create_model((1024,))
        out_model = kernel_def.get_output("out").create_model((1024,))
        
        kernel = KernelModelV2(
            input_models=[in_model],
            output_models=[out_model],
            definition=kernel_def
        )
        
        # This works
        kernel.configure_sdim({"in": 16})
        
        # This fails
        with pytest.raises(ValueError, match="Cannot configure SDIM for output"):
            kernel.configure_sdim({"out": 16})
            
    def test_clear_separation_of_concerns(self):
        """Test that input/output interfaces have different APIs"""
        input_def = InputDefinition(
            name="data",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64)
        )
        output_def = OutputDefinition(
            name="result",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64)
        )
        
        input_model = input_def.create_model((1024,))
        output_model = output_def.create_model((1024,))
        
        # Input has sdim property
        assert hasattr(input_model, 'sdim')
        assert hasattr(input_model, 'streaming_bandwidth')
        
        # Output has streaming_rate property
        assert hasattr(output_model, 'streaming_rate')
        assert not hasattr(output_model, 'sdim')
        assert not hasattr(output_model, 'streaming_bandwidth')
        

class TestMigrationGuidance:
    """Tests that demonstrate migration patterns with helpful examples"""
    
    def test_migration_checklist(self):
        """Demonstrate key migration steps"""
        
        # Step 1: Change InterfaceDirection.WEIGHT to INPUT
        kernel_def = KernelDefinitionV2(name="example")
        
        # OLD: InterfaceDirection.WEIGHT
        # NEW: Just use add_input()
        kernel_def.add_input(InputDefinition(
            name="kernel_weights",
            dtype=DataType.from_string("INT8"),
            block_dims_expr=fixed_tiles(3, 3, 32, 64)
        ))
        
        # Step 2: Replace InterfaceModel with Input/OutputInterface
        # OLD: model = InterfaceModel(...)
        # NEW: 
        input_model = kernel_def.get_input("kernel_weights").create_model(
            (3, 3, 256, 512)
        )
        assert isinstance(input_model, InputInterface)
        
        # Step 3: Replace apply_parallelism with configure_sdim
        # OLD: kernel.apply_parallelism({"in": 16, "out": 16})
        # NEW: kernel.configure_sdim({"in": 16})  # Only inputs!
        
        # Step 4: Use DEPENDENT for dimension-specific constraints
        kernel_def.add_input(InputDefinition(
            name="other",
            dtype=DataType.from_string("INT8"),
            block_dims_expr=fixed_tiles(64, 128)
        ))
        
        # OLD: RelationType.EQUAL on whole interface
        # NEW: RelationType.DEPENDENT on specific dimensions
        kernel_def.add_relationship(
            "kernel_weights", "other", RelationType.DEPENDENT,
            source_dim=3, target_dim=0,
            dependency_type="copy"
        )
        
        # Step 5: Remove output SDIM configuration
        kernel_def.add_output(OutputDefinition(
            name="output",
            dtype=DataType.from_string("INT32"),
            block_dims_expr=fixed_tiles(64, 128)
        ))
        
        # OLD: Would configure output parallelism
        # NEW: Output streaming is computed, not configured
        
        # Create kernel to demonstrate
        other_model = kernel_def.get_input("other").create_model((512, 1024))
        output_model = kernel_def.get_output("output").create_model((512, 1024))
        
        kernel = KernelModelV2(
            input_models=[input_model, other_model],
            output_models=[output_model],
            definition=kernel_def
        )
        
        # Only configure inputs
        kernel.configure_sdim({
            "kernel_weights": [3, 3, 8, 16]  # 8 input channels, 16 output channels
            # "other" gets [16, 1] from DEPENDENT relationship
        })
        
        # Verify configuration
        assert input_model.sdim == (3, 3, 8, 16)
        assert other_model.sdim == (16, 1)
        # Output has no sdim
        assert not hasattr(output_model, "sdim")