"""
Test suite for Phase 2 template context generation with parameter handling.

Tests the enhanced TemplateContext generation with:
- Parameter whitelist validation
- Default value handling
- Runtime parameter extraction setup
- Node attribute definitions
"""

import pytest
from pathlib import Path
from typing import List
from dataclasses import dataclass

# Import only what we need to avoid circular imports
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Parameter
from brainsmith.tools.hw_kernel_gen.templates.template_context import TemplateContext, ParameterDefinition
from brainsmith.tools.hw_kernel_gen.parameter_config.parameter_defaults import (
    is_parameter_whitelisted,
    get_default_value,
    WHITELISTED_DEFAULTS
)


# Mock KernelMetadata to avoid circular import
@dataclass
class MockKernelMetadata:
    """Mock KernelMetadata for testing."""
    name: str
    source_file: Path
    parameters: List[Parameter]
    interfaces: List[InterfaceMetadata]


class TestTemplateContextPhase2:
    """Test Phase 2 template context generation."""
    
    def create_test_kernel_metadata(self, 
                                  module_name: str = "test_accelerator",
                                  parameters: List[Parameter] = None,
                                  interfaces: List[InterfaceMetadata] = None) -> MockKernelMetadata:
        """Create test KernelMetadata with parameters and interfaces."""
        if parameters is None:
            parameters = [
                Parameter(
                    name="PE",
                    param_type="int",
                    default_value="8",
                    description="Processing elements",
                    line_number=10,
                    template_param_name="$PE$"
                ),
                Parameter(
                    name="SIMD",
                    param_type="int",
                    default_value="4",
                    description="SIMD width",
                    line_number=11,
                    template_param_name="$SIMD$"
                ),
                Parameter(
                    name="CHANNELS",
                    param_type="int",
                    default_value=None,  # No default - must be provided by FINN
                    description="Number of channels",
                    line_number=12,
                    template_param_name="$CHANNELS$"
                ),
                Parameter(
                    name="CUSTOM_PARAM",
                    param_type="int",
                    default_value="16",  # Has default but not whitelisted
                    description="Custom parameter",
                    line_number=13,
                    template_param_name="$CUSTOM_PARAM$"
                )
            ]
        
        if interfaces is None:
            interfaces = [
                InterfaceMetadata(
                    name="in0_V",
                    interface_type=InterfaceType.INPUT,
                    allowed_datatypes=[
                        DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False)
                    ],
                    chunking_strategy=BlockChunkingStrategy(
                        block_shape=["PE"],  # Symbolic BDIM
                        rindex=0
                    )
                ),
                InterfaceMetadata(
                    name="weights_V",
                    interface_type=InterfaceType.WEIGHT,
                    allowed_datatypes=[
                        DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True)
                    ],
                    chunking_strategy=BlockChunkingStrategy(
                        block_shape=["SIMD", "CHANNELS"],  # Symbolic BDIM
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
                        block_shape=["PE", ":"],  # Symbolic with wildcard
                        rindex=0
                    )
                )
            ]
        
        return MockKernelMetadata(
            name=module_name,
            source_file=Path(f"{module_name}.sv"),
            parameters=parameters,
            interfaces=interfaces
        )
    
    def test_template_context_generation_basic(self):
        """Test basic TemplateContext generation."""
        kernel_metadata = self.create_test_kernel_metadata()
        
        # Generate TemplateContext
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        # Basic checks
        assert isinstance(template_ctx, TemplateContext)
        assert template_ctx.module_name == "test_accelerator"
        assert template_ctx.class_name == "TestAccelerator"
        assert len(template_ctx.parameter_definitions) == 4
        assert len(template_ctx.interface_metadata) == 3
    
    def test_parameter_whitelist_handling(self):
        """Test parameter whitelist and default value handling."""
        kernel_metadata = self.create_test_kernel_metadata()
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        # Check parameter definitions
        param_dict = {p.name: p for p in template_ctx.parameter_definitions}
        
        # PE - whitelisted with RTL default
        pe_param = param_dict["PE"]
        assert pe_param.is_whitelisted is True
        assert pe_param.default_value == 8
        assert pe_param.is_required is False
        assert "PE" in template_ctx.whitelisted_defaults
        assert template_ctx.whitelisted_defaults["PE"] == 8
        
        # SIMD - whitelisted with RTL default
        simd_param = param_dict["SIMD"]
        assert simd_param.is_whitelisted is True
        assert simd_param.default_value == 4
        assert simd_param.is_required is False
        assert "SIMD" in template_ctx.whitelisted_defaults
        assert template_ctx.whitelisted_defaults["SIMD"] == 4
        
        # CHANNELS - no default, becomes required
        channels_param = param_dict["CHANNELS"]
        assert channels_param.is_whitelisted is False  # Not in whitelist
        assert channels_param.default_value is None
        assert channels_param.is_required is True
        assert "CHANNELS" in template_ctx.required_attributes
        assert "CHANNELS" not in template_ctx.whitelisted_defaults
        
        # CUSTOM_PARAM - has default but not whitelisted, becomes required
        custom_param = param_dict["CUSTOM_PARAM"]
        assert custom_param.is_whitelisted is False
        assert custom_param.default_value is None  # Default ignored
        assert custom_param.is_required is True
        assert "CUSTOM_PARAM" in template_ctx.required_attributes
        assert "CUSTOM_PARAM" not in template_ctx.whitelisted_defaults
    
    def test_whitelisted_param_without_rtl_default(self):
        """Test whitelisted parameter without RTL default gets system default."""
        # Create metadata with whitelisted param without default
        parameters = [
            Parameter(
                name="DEPTH",  # Whitelisted parameter
                param_type="int",
                default_value=None,  # No RTL default
                description="Buffer depth",
                line_number=10
            )
        ]
        
        kernel_metadata = self.create_test_kernel_metadata(parameters=parameters)
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        # Check it gets system default
        depth_param = template_ctx.parameter_definitions[0]
        assert depth_param.name == "DEPTH"
        assert depth_param.is_whitelisted is True
        assert depth_param.default_value == 512  # System default from PARAMETER_DEFAULTS
        assert depth_param.is_required is False
        assert template_ctx.whitelisted_defaults["DEPTH"] == 512
    
    def test_node_attribute_definitions(self):
        """Test node attribute generation for ONNX."""
        kernel_metadata = self.create_test_kernel_metadata()
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        # Get node attributes
        node_attrs = template_ctx.get_node_attribute_definitions()
        
        # Check PE - optional with default
        assert "PE" in node_attrs
        assert node_attrs["PE"] == ("i", False, 8)  # (type, required, default)
        
        # Check SIMD - optional with default
        assert "SIMD" in node_attrs
        assert node_attrs["SIMD"] == ("i", False, 4)
        
        # Check CHANNELS - required
        assert "CHANNELS" in node_attrs
        assert node_attrs["CHANNELS"] == ("i", True, None)
        
        # Check CUSTOM_PARAM - required
        assert "CUSTOM_PARAM" in node_attrs
        assert node_attrs["CUSTOM_PARAM"] == ("i", True, None)
        
        # Check datatype attributes
        assert "inputDataType" in node_attrs
        assert node_attrs["inputDataType"] == ("s", True, "")
        assert "outputDataType" in node_attrs
        assert node_attrs["outputDataType"] == ("s", True, "")
        assert "weightDataType" in node_attrs
        assert node_attrs["weightDataType"] == ("s", True, "")
    
    def test_runtime_parameter_extraction_code(self):
        """Test generation of runtime parameter extraction code."""
        kernel_metadata = self.create_test_kernel_metadata()
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        # Get extraction code
        extraction_lines = template_ctx.get_runtime_parameter_extraction()
        
        assert extraction_lines[0] == "runtime_parameters = {}"
        assert 'runtime_parameters["PE"] = self.get_nodeattr("PE")' in extraction_lines
        assert 'runtime_parameters["SIMD"] = self.get_nodeattr("SIMD")' in extraction_lines
        assert 'runtime_parameters["CHANNELS"] = self.get_nodeattr("CHANNELS")' in extraction_lines
        assert 'runtime_parameters["CUSTOM_PARAM"] = self.get_nodeattr("CUSTOM_PARAM")' in extraction_lines
    
    def test_interface_metadata_code_generation(self):
        """Test generation of interface metadata code."""
        kernel_metadata = self.create_test_kernel_metadata()
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        # Get interface metadata code
        metadata_lines = template_ctx.get_interface_metadata_code()
        
        # Check structure
        assert metadata_lines[0] == "return ["
        assert metadata_lines[-1] == "]"
        
        # Check interfaces are included
        code_str = "\n".join(metadata_lines)
        assert 'name="in0_V"' in code_str
        assert 'name="weights_V"' in code_str
        assert 'name="out0_V"' in code_str
        
        # Check symbolic BDIM shapes
        assert "block_shape=['PE']" in code_str
        assert "block_shape=['SIMD', 'CHANNELS']" in code_str
        assert "block_shape=['PE', ':']" in code_str
    
    def test_template_context_validation(self):
        """Test TemplateContext validation."""
        # Valid context
        kernel_metadata = self.create_test_kernel_metadata()
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        errors = template_ctx.validate()
        assert len(errors) == 0
        
        # Invalid context - missing module name
        invalid_ctx = TemplateContext(
            module_name="",
            class_name="Test",
            source_file=Path("test.sv"),
            interface_metadata=[],
            parameter_definitions=[]
        )
        errors = invalid_ctx.validate()
        assert "Module name is required" in errors
        assert "At least one interface is required" in errors
    
    def test_dict_conversion_backward_compatibility(self):
        """Test conversion to dictionary format for backward compatibility."""
        kernel_metadata = self.create_test_kernel_metadata()
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        # Convert to dict
        context_dict = TemplateContextGenerator._template_context_to_dict(template_ctx)
        
        # Check basic fields
        assert context_dict["kernel_name"] == "test_accelerator"
        assert context_dict["class_name"] == "TestAccelerator"
        
        # Check enhanced parameter info is included
        assert "parameter_definitions" in context_dict
        assert "whitelisted_defaults" in context_dict
        assert "required_attributes" in context_dict
        
        # Check RTL parameters have enhanced info
        rtl_params = context_dict["rtl_parameters"]
        pe_param = next(p for p in rtl_params if p["name"] == "PE")
        assert pe_param["is_whitelisted"] is True
        assert pe_param["is_required"] is False
        assert pe_param["default_value"] == 8
    
    def test_interface_categorization(self):
        """Test interfaces are properly categorized by type."""
        kernel_metadata = self.create_test_kernel_metadata()
        template_ctx = TemplateContextGenerator.generate_template_context(kernel_metadata)
        
        # Check categorization
        assert len(template_ctx.input_interfaces) == 1
        assert template_ctx.input_interfaces[0].name == "in0_V"
        
        assert len(template_ctx.output_interfaces) == 1
        assert template_ctx.output_interfaces[0].name == "out0_V"
        
        assert len(template_ctx.weight_interfaces) == 1
        assert template_ctx.weight_interfaces[0].name == "weights_V"
        
        # Check flags
        assert template_ctx.has_inputs is True
        assert template_ctx.has_outputs is True
        assert template_ctx.has_weights is True
        assert template_ctx.has_config is False