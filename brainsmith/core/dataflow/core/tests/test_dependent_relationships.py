############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Tests for DEPENDENT relationship type and dimension-specific constraints
"""

import pytest
from typing import List, Tuple

from brainsmith.core.dataflow.core import (
    InputDefinition, OutputDefinition, KernelDefinitionV2, KernelModelV2,
    DataType, RelationType, DimensionRelationship,
    fixed_tiles
)


class TestDependentRelationship:
    """Test DEPENDENT relationship functionality"""
    
    def test_dependent_copy_single_dimension(self):
        """Test DEPENDENT with copy dependency on single dimension"""
        kernel_def = KernelDefinitionV2(name="test")
        
        # Two inputs with different shapes
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
        
        # K dimension must match (copy)
        kernel_def.add_relationship(
            "A", "B", RelationType.DEPENDENT,
            source_dim=1, target_dim=0,
            dependency_type="copy"
        )
        
        # Create models
        a_model = kernel_def.get_input("A").create_model((512, 256))
        b_model = kernel_def.get_input("B").create_model((256, 1024))
        
        kernel = KernelModelV2(
            input_models=[a_model, b_model],
            output_models=[],
            definition=kernel_def
        )
        
        # Configure A
        kernel.configure_sdim({"A": [8, 16]})
        
        # B's first dimension should copy A's second dimension
        assert kernel.get_input_model("A").sdim == (8, 16)
        assert kernel.get_input_model("B").sdim == (16, 1)  # First dim = 16 (copied)
        
    def test_dependent_scaled_relationship(self):
        """Test DEPENDENT with scaled dependency"""
        kernel_def = KernelDefinitionV2(name="test")
        
        kernel_def.add_input(InputDefinition(
            name="input",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 128)
        ))
        kernel_def.add_input(InputDefinition(
            name="weights",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 128)
        ))
        
        # Weights dim 0 = input dim 0 / 2
        kernel_def.add_relationship(
            "input", "weights", RelationType.DEPENDENT,
            source_dim=0, target_dim=0,
            dependency_type="scaled",
            scale_factor=0.5
        )
        
        # Create models
        input_model = kernel_def.get_input("input").create_model((1024, 2048))
        weights_model = kernel_def.get_input("weights").create_model((512, 2048))
        
        kernel = KernelModelV2(
            input_models=[input_model, weights_model],
            output_models=[],
            definition=kernel_def
        )
        
        # Configure input
        kernel.configure_sdim({"input": [16, 8]})
        
        # Weights dim 0 should be scaled
        assert kernel.get_input_model("input").sdim == (16, 8)
        assert kernel.get_input_model("weights").sdim == (8, 1)  # 16 * 0.5 = 8
        
    def test_dependent_min_relationship(self):
        """Test DEPENDENT with min dependency"""
        kernel_def = KernelDefinitionV2(name="test")
        
        kernel_def.add_input(InputDefinition(
            name="A",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_input(InputDefinition(
            name="B",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 32)
        ))
        kernel_def.add_input(InputDefinition(
            name="C",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(48, 48)
        ))
        
        # C takes minimum of A and B dimensions
        kernel_def.add_relationship(
            "A", "C", RelationType.DEPENDENT,
            source_dim=0, target_dim=0,
            dependency_type="min",
            other_source="B",
            other_source_dim=0
        )
        
        # Create models
        a_model = kernel_def.get_input("A").create_model((512, 512))
        b_model = kernel_def.get_input("B").create_model((256, 256))
        c_model = kernel_def.get_input("C").create_model((384, 384))
        
        kernel = KernelModelV2(
            input_models=[a_model, b_model, c_model],
            output_models=[],
            definition=kernel_def
        )
        
        # Configure A and B
        kernel.configure_sdim({
            "A": [16, 16],
            "B": [8, 8]
        })
        
        # C should take minimum
        assert kernel.get_input_model("C").sdim == (8, 1)  # min(16, 8) = 8
        
    def test_multiple_dependent_relationships(self):
        """Test multiple DEPENDENT relationships on same interface"""
        kernel_def = KernelDefinitionV2(name="complex")
        
        # Three inputs forming a chain
        kernel_def.add_input(InputDefinition(
            name="X",  # [A, B, C]
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 64, 128)
        ))
        kernel_def.add_input(InputDefinition(
            name="Y",  # [B, C, D]
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 128, 256)
        ))
        kernel_def.add_input(InputDefinition(
            name="Z",  # [C, D, E]
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(128, 256, 512)
        ))
        
        # Y[0] = X[1] (B dimension)
        kernel_def.add_relationship(
            "X", "Y", RelationType.DEPENDENT,
            source_dim=1, target_dim=0,
            dependency_type="copy"
        )
        
        # Y[1] = X[2] (C dimension)
        kernel_def.add_relationship(
            "X", "Y", RelationType.DEPENDENT,
            source_dim=2, target_dim=1,
            dependency_type="copy"
        )
        
        # Z[0] = Y[1] (C dimension chain)
        kernel_def.add_relationship(
            "Y", "Z", RelationType.DEPENDENT,
            source_dim=1, target_dim=0,
            dependency_type="copy"
        )
        
        # Create models
        x_model = kernel_def.get_input("X").create_model((256, 512, 1024))
        y_model = kernel_def.get_input("Y").create_model((512, 1024, 2048))
        z_model = kernel_def.get_input("Z").create_model((1024, 2048, 4096))
        
        kernel = KernelModelV2(
            input_models=[x_model, y_model, z_model],
            output_models=[],
            definition=kernel_def
        )
        
        # Configure only X
        kernel.configure_sdim({"X": [4, 8, 16]})
        
        # Check propagation
        assert kernel.get_input_model("X").sdim == (4, 8, 16)
        assert kernel.get_input_model("Y").sdim == (8, 16, 1)  # Dims 0,1 from X
        assert kernel.get_input_model("Z").sdim == (16, 1, 1)  # Dim 0 from Y[1]
        

class TestRelationshipDescription:
    """Test relationship description and representation"""
    
    def test_dependent_relationship_description(self):
        """Test DEPENDENT relationship describe() method"""
        # Copy relationship
        rel = DimensionRelationship(
            source_interface="A",
            target_interface="B",
            relation=RelationType.DEPENDENT,
            source_dim=1,
            target_dim=0,
            dependency_type="copy"
        )
        
        desc = rel.describe()
        assert "B[0]" in desc
        assert "A[1]" in desc
        assert "copy" in desc.lower()
        
        # Scaled relationship
        rel_scaled = DimensionRelationship(
            source_interface="input",
            target_interface="output",
            relation=RelationType.DEPENDENT,
            source_dim=2,
            target_dim=1,
            dependency_type="scaled",
            scale_factor=2.0
        )
        
        desc_scaled = rel_scaled.describe()
        assert "output[1]" in desc_scaled
        assert "input[2]" in desc_scaled
        assert "2.0" in desc_scaled
        
        # Min relationship
        rel_min = DimensionRelationship(
            source_interface="A",
            target_interface="C",
            relation=RelationType.DEPENDENT,
            source_dim=0,
            target_dim=0,
            dependency_type="min",
            other_source="B",
            other_source_dim=1
        )
        
        desc_min = rel_min.describe()
        assert "C[0]" in desc_min
        assert "min" in desc_min.lower()
        assert "A[0]" in desc_min
        assert "B[1]" in desc_min
        

class TestRelationshipEvaluation:
    """Test relationship constraint evaluation"""
    
    def test_dependent_relationship_evaluate(self):
        """Test DEPENDENT relationship evaluate() method"""
        # Create test models
        kernel_def = KernelDefinitionV2(name="test")
        
        kernel_def.add_input(InputDefinition(
            name="A",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 32)
        ))
        kernel_def.add_input(InputDefinition(
            name="B",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 128)
        ))
        
        a_model = kernel_def.get_input("A").create_model((512, 256))
        b_model = kernel_def.get_input("B").create_model((256, 1024))
        
        # Set SDIM values
        a_model.sdim = (8, 16)
        b_model.sdim = (16, 4)
        
        # Create models dict
        models = {"A": a_model, "B": b_model}
        
        # Test copy relationship
        rel = DimensionRelationship(
            source_interface="A",
            target_interface="B",
            relation=RelationType.DEPENDENT,
            source_dim=1,
            target_dim=0,
            dependency_type="copy"
        )
        
        # Should be satisfied (B[0]=16 equals A[1]=16)
        assert rel.evaluate(models) == True
        
        # Change B's SDIM
        b_model.sdim = (8, 4)
        
        # Should not be satisfied
        assert rel.evaluate(models) == False
        
    def test_scaled_relationship_evaluate(self):
        """Test scaled DEPENDENT relationship evaluation"""
        kernel_def = KernelDefinitionV2(name="test")
        
        kernel_def.add_input(InputDefinition(
            name="input",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_input(InputDefinition(
            name="output",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(32, 32)
        ))
        
        input_model = kernel_def.get_input("input").create_model((512, 512))
        output_model = kernel_def.get_input("output").create_model((256, 256))
        
        input_model.sdim = (16, 8)
        output_model.sdim = (8, 4)  # Half of input
        
        models = {"input": input_model, "output": output_model}
        
        # Test scale factor 0.5
        rel = DimensionRelationship(
            source_interface="input",
            target_interface="output",
            relation=RelationType.DEPENDENT,
            source_dim=0,
            target_dim=0,
            dependency_type="scaled",
            scale_factor=0.5
        )
        
        # Should be satisfied (output[0]=8 equals input[0]*0.5=8)
        assert rel.evaluate(models) == True
        
        # Change output SDIM
        output_model.sdim = (4, 4)
        
        # Should not be satisfied
        assert rel.evaluate(models) == False
        

class TestDependentPropagation:
    """Test SDIM propagation through DEPENDENT relationships"""
    
    def test_transitive_dependent_propagation(self):
        """Test transitive propagation through multiple DEPENDENT relationships"""
        kernel_def = KernelDefinitionV2(name="chain")
        
        # Create chain: A -> B -> C -> D
        for name in ["A", "B", "C", "D"]:
            kernel_def.add_input(InputDefinition(
                name=name,
                dtype=DataType.from_string("FP16"),
                block_dims_expr=fixed_tiles(64, 64)
            ))
            
        # Chain relationships
        kernel_def.add_relationship(
            "A", "B", RelationType.DEPENDENT,
            source_dim=1, target_dim=0,
            dependency_type="copy"
        )
        kernel_def.add_relationship(
            "B", "C", RelationType.DEPENDENT,
            source_dim=1, target_dim=0,
            dependency_type="copy"
        )
        kernel_def.add_relationship(
            "C", "D", RelationType.DEPENDENT,
            source_dim=1, target_dim=0,
            dependency_type="copy"
        )
        
        # Create models
        models = []
        for name in ["A", "B", "C", "D"]:
            model = kernel_def.get_input(name).create_model((512, 512))
            models.append(model)
            
        kernel = KernelModelV2(
            input_models=models,
            output_models=[],
            definition=kernel_def
        )
        
        # Configure only A
        kernel.configure_sdim({"A": [8, 16]})
        
        # Check propagation through chain
        assert kernel.get_input_model("A").sdim == (8, 16)
        assert kernel.get_input_model("B").sdim == (16, 1)  # B[0] = A[1]
        assert kernel.get_input_model("C").sdim == (1, 1)   # C[0] = B[1]
        assert kernel.get_input_model("D").sdim == (1, 1)   # D[0] = C[1]
        
    def test_conflict_detection(self):
        """Test detection of conflicting DEPENDENT relationships"""
        kernel_def = KernelDefinitionV2(name="conflict")
        
        kernel_def.add_input(InputDefinition(
            name="A",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_input(InputDefinition(
            name="B",
            dtype=DataType.from_string("FP16"), 
            block_dims_expr=fixed_tiles(64, 64)
        ))
        kernel_def.add_input(InputDefinition(
            name="C",
            dtype=DataType.from_string("FP16"),
            block_dims_expr=fixed_tiles(64, 64)
        ))
        
        # C[0] depends on both A[0] and B[0] with different requirements
        kernel_def.add_relationship(
            "A", "C", RelationType.DEPENDENT,
            source_dim=0, target_dim=0,
            dependency_type="copy"
        )
        kernel_def.add_relationship(
            "B", "C", RelationType.DEPENDENT,
            source_dim=0, target_dim=0,
            dependency_type="scaled",
            scale_factor=2.0
        )
        
        # Create models
        a_model = kernel_def.get_input("A").create_model((512, 512))
        b_model = kernel_def.get_input("B").create_model((512, 512))
        c_model = kernel_def.get_input("C").create_model((512, 512))
        
        kernel = KernelModelV2(
            input_models=[a_model, b_model, c_model],
            output_models=[],
            definition=kernel_def
        )
        
        # Configure A and B with conflicting requirements for C
        with pytest.raises(ValueError, match="Conflicting SDIM constraints"):
            kernel.configure_sdim({
                "A": [16, 16],  # Would require C[0] = 16
                "B": [4, 4]     # Would require C[0] = 8 (4*2)
            })