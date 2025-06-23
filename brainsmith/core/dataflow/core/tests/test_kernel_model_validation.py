############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for KernelModel validation and robustness features"""

import pytest
from typing import List

from brainsmith.core.dataflow.core.kernel_model import KernelModel
from brainsmith.core.dataflow.core.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.interface_model import InterfaceModel
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection
from brainsmith.core.dataflow.core.relationships import RelationType
from brainsmith.core.dataflow.core.kernel_model_validation import (
    KernelModelValidator, ValidationIssue, validate_kernel_model
)


class TestKernelModelValidation:
    """Test kernel model validation functionality"""
    
    def test_validate_missing_interfaces(self):
        """Test detection of missing required interfaces"""
        # Create definition with two interfaces
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
        
        # Create model with only one interface
        input_model = input_def.create_model(
            tensor_dims=(128, 256),
            block_dims=(32, 64)
        )
        
        kernel_model = KernelModel(
            interface_models=[input_model],
            definition=kernel_def
        )
        
        # Validate
        validator = KernelModelValidator(kernel_model)
        issues = validator.validate_configuration()
        
        # Should have error about missing output
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 1
        assert "Missing required interfaces: output" in errors[0].message
    
    def test_validate_dimension_compatibility(self):
        """Test dimension compatibility validation"""
        idef = InterfaceDefinition(
            name="data",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT16")
        )
        
        # Create model with incompatible dimensions
        model = InterfaceModel(
            tensor_dims=(100, 200),  # Not divisible by block
            block_dims=(32, 64),
            definition=idef
        )
        
        kernel_model = KernelModel(
            interface_models=[model]
        )
        
        validator = KernelModelValidator(kernel_model)
        issues = validator.validate_configuration()
        
        # Should have errors about divisibility
        errors = [i for i in issues if "not divisible" in i.message]
        assert len(errors) == 2  # Both dimensions
        assert "Tensor dim 100 not divisible by block dim 32" in errors[0].message
        assert "Tensor dim 200 not divisible by block dim 64" in errors[1].message
    
    def test_validate_parallelism_configuration(self):
        """Test parallelism configuration validation"""
        idef = InterfaceDefinition(
            name="small",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        
        model = idef.create_model(
            tensor_dims=(8, 8),
            block_dims=(4, 4)
        )
        
        # Set iPar too high
        model.ipar = 32  # Block size is only 16
        
        kernel_model = KernelModel(
            interface_models=[model]
        )
        
        validator = KernelModelValidator(kernel_model)
        issues = validator.validate_configuration()
        
        # Should have error about iPar exceeding block size
        errors = [i for i in issues if "exceeds block size" in i.message]
        assert len(errors) == 1
        assert "iPar (32) exceeds block size (16)" in errors[0].message
    
    def test_validate_equal_relationships(self):
        """Test EQUAL relationship validation"""
        # Create mismatched interfaces
        a_def = InterfaceDefinition("A", InterfaceDirection.INPUT, DataType.from_string("INT8"))
        b_def = InterfaceDefinition("B", InterfaceDirection.OUTPUT, DataType.from_string("INT8"))
        
        kernel_def = KernelDefinition(
            name="test",
            interface_definitions=[a_def, b_def]
        )
        
        # Add EQUAL relationship
        kernel_def.add_relationship("A", "B", RelationType.EQUAL, 
                                   source_dim=1, target_dim=0)
        
        # Create models with mismatched dimensions
        a_model = a_def.create_model((128, 256), (32, 64))
        b_model = b_def.create_model((512, 128), (128, 32))  # Mismatch: 256 != 512
        
        kernel_model = KernelModel(
            interface_models=[a_model, b_model],
            definition=kernel_def
        )
        
        validator = KernelModelValidator(kernel_model)
        issues = validator.validate_configuration()
        
        # Should have error about relationship violation
        errors = [i for i in issues if "EQUAL relationship violated" in i.message]
        assert len(errors) == 1
    
    def test_validate_multiple_relationships(self):
        """Test MULTIPLE relationship validation"""
        input_def = InterfaceDefinition("input", InterfaceDirection.INPUT, 
                                       DataType.from_string("INT8"))
        output_def = InterfaceDefinition("output", InterfaceDirection.OUTPUT,
                                        DataType.from_string("INT8"))
        
        kernel_def = KernelDefinition(
            name="expander",
            interface_definitions=[input_def, output_def]
        )
        
        # Add MULTIPLE relationship
        kernel_def.add_relationship("input", "output", RelationType.MULTIPLE,
                                   factor=4.0)
        
        # Create models where output is NOT 4x input
        input_model = input_def.create_model((128,), (32,))
        output_model = output_def.create_model((256,), (64,))  # Only 2x, not 4x
        
        kernel_model = KernelModel(
            interface_models=[input_model, output_model],
            definition=kernel_def
        )
        
        validator = KernelModelValidator(kernel_model)
        issues = validator.validate_configuration()
        
        # Should have error about MULTIPLE violation
        errors = [i for i in issues if "MULTIPLE relationship violated" in i.message]
        assert len(errors) == 1
    
    def test_validate_performance_warnings(self):
        """Test performance feasibility warnings"""
        idef = InterfaceDefinition(
            name="huge",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT32")
        )
        
        model = idef.create_model(
            tensor_dims=(1024, 1024, 1024),
            block_dims=(32, 32, 32)
        )
        
        # Set very high parallelism
        model.ipar = 2048  # Above the 1024 threshold
        
        kernel_model = KernelModel(
            interface_models=[model],
            clock_freq_mhz=500.0
        )
        
        validator = KernelModelValidator(kernel_model)
        issues = validator.validate_configuration()
        
        # Should have warnings about high parallelism and bandwidth
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) >= 2
        assert any("Very high parallelism" in w.message for w in warnings)
        assert any("bandwidth requirement" in w.message for w in warnings)
    
    def test_detect_circular_dependencies(self):
        """Test detection of circular relationship dependencies"""
        # Create interfaces
        defs = []
        for name in ["A", "B", "C"]:
            defs.append(InterfaceDefinition(name, InterfaceDirection.INPUT,
                                          DataType.from_string("INT8")))
        
        kernel_def = KernelDefinition(
            name="circular",
            interface_definitions=defs
        )
        
        # Create circular relationships: A -> B -> C -> A
        kernel_def.add_relationship("A", "B", RelationType.EQUAL)
        kernel_def.add_relationship("B", "C", RelationType.EQUAL)
        kernel_def.add_relationship("C", "A", RelationType.EQUAL)
        
        # Create models
        models = [d.create_model((128,), (32,)) for d in defs]
        
        kernel_model = KernelModel(
            interface_models=models,
            definition=kernel_def
        )
        
        validator = KernelModelValidator(kernel_model)
        conflicts = validator.detect_conflicts()
        
        # Should detect circular dependency
        circular = [c for c in conflicts if "Circular dependency" in c.message]
        assert len(circular) == 1
        assert all(name in circular[0].message for name in ["A", "B", "C"])
    
    def test_suggest_fixes(self):
        """Test fix suggestions"""
        idef = InterfaceDefinition(
            name="data",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        
        # Create model with multiple issues
        model = InterfaceModel(
            tensor_dims=(100, 200),  # Not divisible
            block_dims=(32, 64),
            definition=idef
        )
        model.ipar = 2048  # Very high
        
        kernel_model = KernelModel(
            interface_models=[model]
        )
        
        validator = KernelModelValidator(kernel_model)
        validator.validate_configuration()
        suggestions = validator.suggest_fixes()
        
        # Should have suggestions
        assert len(suggestions) > 0
        assert any("Fix" in s and "errors" in s for s in suggestions)
        assert any("block dimensions" in s for s in suggestions)
    
    def test_validate_kernel_model_function(self):
        """Test the convenience validation function"""
        # Create valid kernel
        idef = InterfaceDefinition(
            name="valid",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        )
        
        model = idef.create_model(
            tensor_dims=(128, 256),
            block_dims=(32, 64)
        )
        
        kernel_model = KernelModel(
            interface_models=[model]
        )
        
        # Should pass validation
        assert validate_kernel_model(kernel_model, verbose=False) == True
        
        # Create invalid kernel
        model.ipar = 10000  # Too high for block size
        assert validate_kernel_model(kernel_model, verbose=False) == False
    
    def test_parallelism_mismatch_warning(self):
        """Test warning for parallelism mismatches in related interfaces"""
        # Create related interfaces
        a_def = InterfaceDefinition("A", InterfaceDirection.INPUT, DataType.from_string("INT8"))
        b_def = InterfaceDefinition("B", InterfaceDirection.OUTPUT, DataType.from_string("INT8"))
        
        kernel_def = KernelDefinition(
            name="test",
            interface_definitions=[a_def, b_def]
        )
        
        # Add relationship with matching dimensions
        kernel_def.add_relationship("A", "B", RelationType.EQUAL,
                                   source_dim=0, target_dim=0)
        
        # Create models with same dimensions but different parallelism
        a_model = a_def.create_model((128, 256), (32, 64))
        b_model = b_def.create_model((128, 256), (32, 64))
        
        a_model.ipar = 8
        b_model.ipar = 4  # Different parallelism
        
        kernel_model = KernelModel(
            interface_models=[a_model, b_model],
            definition=kernel_def
        )
        
        validator = KernelModelValidator(kernel_model)
        issues = validator.validate_configuration()
        
        # Should have warning about parallelism mismatch
        warnings = [i for i in issues if i.severity == "warning" and "Parallelism mismatch" in i.message]
        assert len(warnings) >= 1


if __name__ == "__main__":
    pytest.main([__file__])