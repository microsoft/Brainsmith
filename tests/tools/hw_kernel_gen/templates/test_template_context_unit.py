"""
Unit tests for Phase 2 TemplateContext functionality.

Direct tests of TemplateContext without circular import issues.
"""

import pytest
from pathlib import Path
from typing import List

from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy
from brainsmith.tools.hw_kernel_gen.templates.template_context import TemplateContext, ParameterDefinition
from brainsmith.tools.hw_kernel_gen.parameter_config.parameter_defaults import (
    is_parameter_whitelisted,
    get_default_value,
    WHITELISTED_DEFAULTS
)


class TestTemplateContextUnit:
    """Unit tests for TemplateContext functionality."""
    
    def test_parameter_definition_creation(self):
        """Test ParameterDefinition creation and attributes."""
        param = ParameterDefinition(
            name="PE",
            param_type="int",
            default_value=8,
            description="Processing elements",
            line_number=10,
            template_param_name="$PE$",
            is_whitelisted=True,
            is_required=False
        )
        
        assert param.name == "PE"
        assert param.default_value == 8
        assert param.is_whitelisted is True
        assert param.is_required is False
    
    def test_template_context_basic(self):
        """Test basic TemplateContext creation."""
        # Create parameter definitions
        params = [
            ParameterDefinition(
                name="PE",
                default_value=8,
                is_whitelisted=True,
                is_required=False
            ),
            ParameterDefinition(
                name="CHANNELS",
                default_value=None,
                is_whitelisted=False,
                is_required=True
            )
        ]
        
        # Create interfaces
        interfaces = [
            InterfaceMetadata(
                name="in0_V",
                interface_type=InterfaceType.INPUT,
                allowed_datatypes=[
                    DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False)
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=["PE"],
                    rindex=0
                )
            )
        ]
        
        # Create context
        ctx = TemplateContext(
            module_name="test_module",
            class_name="TestModule",
            source_file=Path("test.sv"),
            interface_metadata=interfaces,
            parameter_definitions=params,
            whitelisted_defaults={"PE": 8},
            required_attributes=["CHANNELS"],
            input_interfaces=interfaces,
            has_inputs=True
        )
        
        assert ctx.module_name == "test_module"
        assert ctx.class_name == "TestModule"
        assert len(ctx.parameter_definitions) == 2
        assert len(ctx.interface_metadata) == 1
    
    def test_node_attribute_definitions(self):
        """Test ONNX node attribute generation."""
        params = [
            ParameterDefinition(
                name="PE",
                default_value=8,
                is_whitelisted=True,
                is_required=False
            ),
            ParameterDefinition(
                name="SIMD",
                default_value=4,
                is_whitelisted=True,
                is_required=False
            ),
            ParameterDefinition(
                name="CHANNELS",
                default_value=None,
                is_whitelisted=False,
                is_required=True
            )
        ]
        
        ctx = TemplateContext(
            module_name="test",
            class_name="Test",
            source_file=Path("test.sv"),
            interface_metadata=[],
            parameter_definitions=params,
            whitelisted_defaults={"PE": 8, "SIMD": 4},
            required_attributes=["CHANNELS"],
            has_inputs=True,
            has_outputs=True,
            has_weights=True
        )
        
        attrs = ctx.get_node_attribute_definitions()
        
        # Check parameter attributes
        assert attrs["PE"] == ("i", False, 8)  # Optional with default
        assert attrs["SIMD"] == ("i", False, 4)  # Optional with default
        assert attrs["CHANNELS"] == ("i", True, None)  # Required
        
        # Check datatype attributes
        assert attrs["inputDataType"] == ("s", True, "")
        assert attrs["outputDataType"] == ("s", True, "")
        assert attrs["weightDataType"] == ("s", True, "")
        
        # Check standard attributes
        assert attrs["runtime_writeable_weights"] == ("i", False, 0)
        assert attrs["numInputVectors"] == ("ints", False, [1])
    
    def test_runtime_parameter_extraction(self):
        """Test runtime parameter extraction code generation."""
        params = [
            ParameterDefinition(name="PE"),
            ParameterDefinition(name="SIMD"),
            ParameterDefinition(name="CHANNELS")
        ]
        
        ctx = TemplateContext(
            module_name="test",
            class_name="Test", 
            source_file=Path("test.sv"),
            interface_metadata=[],
            parameter_definitions=params
        )
        
        lines = ctx.get_runtime_parameter_extraction()
        
        assert lines[0] == "runtime_parameters = {}"
        assert 'runtime_parameters["PE"] = self.get_nodeattr("PE")' in lines
        assert 'runtime_parameters["SIMD"] = self.get_nodeattr("SIMD")' in lines
        assert 'runtime_parameters["CHANNELS"] = self.get_nodeattr("CHANNELS")' in lines
    
    def test_interface_metadata_code_generation(self):
        """Test interface metadata code generation."""
        interfaces = [
            InterfaceMetadata(
                name="in0_V",
                interface_type=InterfaceType.INPUT,
                allowed_datatypes=[
                    DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False)
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=["PE"],
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="out0_V",
                interface_type=InterfaceType.OUTPUT,
                allowed_datatypes=[
                    DataTypeConstraint(finn_type="UINT16", bit_width=16, signed=False)
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=["PE", ":"],
                    rindex=1
                )
            )
        ]
        
        ctx = TemplateContext(
            module_name="test",
            class_name="Test",
            source_file=Path("test.sv"),
            interface_metadata=interfaces,
            parameter_definitions=[]
        )
        
        lines = ctx.get_interface_metadata_code()
        code = "\n".join(lines)
        
        # Check structure
        assert lines[0] == "return ["
        assert lines[-1] == "]"
        
        # Check content
        assert 'name="in0_V"' in code
        assert 'interface_type=InterfaceType.INPUT' in code
        assert "block_shape=['PE']" in code
        assert 'rindex=0' in code
        
        assert 'name="out0_V"' in code
        assert 'interface_type=InterfaceType.OUTPUT' in code
        assert "block_shape=['PE', ':']" in code
        assert 'rindex=1' in code
    
    def test_validation(self):
        """Test TemplateContext validation."""
        # Valid context
        ctx = TemplateContext(
            module_name="test",
            class_name="Test",
            source_file=Path("test.sv"),
            interface_metadata=[
                InterfaceMetadata(
                    name="test",
                    interface_type=InterfaceType.INPUT,
                    allowed_datatypes=[],
                    chunking_strategy=None
                )
            ],
            parameter_definitions=[]
        )
        
        errors = ctx.validate()
        assert len(errors) == 0
        
        # Invalid - missing module name
        ctx_invalid = TemplateContext(
            module_name="",
            class_name="Test",
            source_file=Path("test.sv"),
            interface_metadata=[],
            parameter_definitions=[]
        )
        
        errors = ctx_invalid.validate()
        assert "Module name is required" in errors
        assert "At least one interface is required" in errors
    
    def test_parameter_whitelist_functions(self):
        """Test parameter whitelist utility functions."""
        # Test whitelisted parameters
        assert is_parameter_whitelisted("PE") is True
        assert is_parameter_whitelisted("SIMD") is True
        assert is_parameter_whitelisted("DEPTH") is True
        
        # Test non-whitelisted
        assert is_parameter_whitelisted("CUSTOM_PARAM") is False
        assert is_parameter_whitelisted("UNKNOWN") is False
        
        # Test default values
        assert get_default_value("PE") == 1
        assert get_default_value("SIMD") == 1
        assert get_default_value("DEPTH") == 512
        assert get_default_value("UNKNOWN") == 1  # Default fallback