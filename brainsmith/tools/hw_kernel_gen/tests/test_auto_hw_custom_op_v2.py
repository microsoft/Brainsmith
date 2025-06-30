############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Unit tests for AutoHWCustomOp Kernel Modeling integration.

This module provides comprehensive test coverage for the AutoHWCustomOp base class,
ensuring correct integration with the Kernel Modeling system and FINN compatibility.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Tuple, List

# QONNX imports
from qonnx.core.datatype import DataType

# Kernel Modeling imports
from brainsmith.core.dataflow import (
    KernelDefinition,
    InputDefinition,
    OutputDefinition,
    RelationType,
    parameterized_tiles,
    fixed_tiles
)
from brainsmith.core.dataflow.base import ParameterBinding
from brainsmith.tools.hw_kernel_gen.data import DatatypeConstraintGroup

# AutoHWCustomOp import
from brainsmith.tools.hw_kernel_gen.auto_hw_custom_op_v2 import AutoHWCustomOp


# Fixtures

@pytest.fixture
def sample_kernel_def():
    """Create a sample KernelDefinition for testing."""
    kernel_def = KernelDefinition(name="test_kernel")
    
    # Add input interface
    kernel_def.add_input(InputDefinition(
        name="input",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 8, 32),
            DatatypeConstraintGroup("UINT", 8, 32)
        ],
        block_dims_expr=parameterized_tiles("Batch", "NumChannels"),
        optional=False
    ))
    
    # Add weight interface (optional)
    kernel_def.add_input(InputDefinition(
        name="weights",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 8, 16)
        ],
        block_dims_expr=parameterized_tiles("NumChannels", "NumFilters"),
        optional=True,
        is_weight=True  # Mark as weight input
    ))
    
    # Add output interface
    kernel_def.add_output(OutputDefinition(
        name="output",
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 8, 32),
            DatatypeConstraintGroup("UINT", 8, 32)
        ],
        block_dims_expr=parameterized_tiles("Batch", "NumFilters")
    ))
    
    # Add relationships
    kernel_def.add_relationship(
        source_name="input",
        target_name="output",
        relationship_type=RelationType.DEPENDENT,
        description="Input channels transformed to output filters"
    )
    
    return kernel_def


@pytest.fixture
def mock_onnx_node():
    """Create a mock ONNX node with attributes."""
    node = MagicMock()
    node.name = "TestOp_0"
    node.op_type = "TestOp"
    node.input = ["input_tensor", "weight_tensor"]
    node.output = ["output_tensor"]
    
    # Mock attributes using FINN-standard names
    attributes = {
        "backend": "fpgadataflow",
        "Batch": 1,
        "NumChannels": 64,
        "NumFilters": 32,
        "inputDataType": "UINT8",    # Regular input datatype
        "weightDataType": "INT8",     # Weight input datatype
        "outputDataType": "UINT16",   # Output datatype
        "input_sdim": [1, 8],  # Modern interface-specific SDIM
        "weights_sdim": [8, 4],  # Per-dimension SDIM for weights
        "SIMD": 8,  # Legacy attribute (will be ignored)
        "PE": 4,    # Legacy attribute
    }
    
    node.attribute = []
    for name, value in attributes.items():
        attr = MagicMock()
        attr.name = name
        if isinstance(value, int):
            attr.i = value
            attr.s = ""
            attr.ints = []
        elif isinstance(value, list):
            attr.i = 0
            attr.s = ""
            attr.ints = value  # Store lists as repeated ints
        else:
            attr.s = value
            attr.i = 0
            attr.ints = []
        node.attribute.append(attr)
    
    # Mock get_attribute method
    def get_attribute(name):
        for attr in node.attribute:
            if attr.name == name:
                if attr.ints:
                    return attr.ints
                elif attr.i != 0:
                    return attr.i
                else:
                    return attr.s
        return None
    
    node.get_attribute = get_attribute
    
    return node


@pytest.fixture
def auto_op(sample_kernel_def, mock_onnx_node):
    """Create AutoHWCustomOp instance for testing."""
    # Create a test class that provides proper shape extraction
    class TestAutoHWCustomOp(AutoHWCustomOp):
        def _extract_input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
            # Extract specs for all inputs
            specs = {}
            
            # Input interface
            specs["input"] = (
                (1, self.get_nodeattr("NumChannels")),
                self._get_interface_datatype("input", is_input=True)
            )
            
            # Weights interface (if present in kernel def)
            if len(self._kernel_def.input_definitions) > 1:
                specs["weights"] = (
                    (self.get_nodeattr("NumChannels"), self.get_nodeattr("NumFilters")),
                    self._get_interface_datatype("weights", is_input=True)
                )
            
            return specs
        
        def _extract_output_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
            # Output interface
            return {
                "output": (
                    (1, self.get_nodeattr("NumFilters")),
                    self._get_interface_datatype("output", is_input=False)
                )
            }
        
        def _extract_parameter_binding(self) -> ParameterBinding:
            # Extract parameters needed by kernel definition
            params = {}
            for param_name in ["Batch", "NumChannels", "NumFilters"]:
                value = self.get_nodeattr(param_name)
                if value is not None:
                    params[param_name] = value
            return ParameterBinding(params) if params else None
        
        def get_nodeattr(self, name: str):
            """Override to use mock node attributes."""
            for attr in self.onnx_node.attribute:
                if attr.name == name:
                    if hasattr(attr, 'ints') and attr.ints:
                        return attr.ints
                    elif attr.i != 0:
                        return attr.i
                    else:
                        return attr.s
            return None
    
    return TestAutoHWCustomOp(mock_onnx_node, sample_kernel_def)


# Test Model Creation

class TestModelCreation:
    """Test lazy model creation and configuration."""
    
    def test_lazy_model_creation(self, auto_op):
        """Test that model is created lazily on first access."""
        # Model should not exist initially
        assert auto_op._kernel_model is None
        
        # Accessing a FINN method should create the model
        dtype = auto_op.get_input_datatype(0)
        assert auto_op._kernel_model is not None
        assert dtype == DataType["UINT8"]
    
    def test_input_spec_extraction(self, auto_op):
        """Test extraction of input specifications."""
        # Force model creation
        auto_op._ensure_kernel_model()
        
        # Check input models
        input_models = list(auto_op._kernel_model.input_models)
        assert len(input_models) >= 1
        
        # Check first input
        inp0 = input_models[0]
        assert inp0.definition.name == "input"
        assert inp0.datatype == DataType["UINT8"]
        assert inp0.tensor_dims == (1, 64)  # From NumChannels
    
    def test_output_spec_extraction(self, auto_op):
        """Test extraction of output specifications."""
        # Force model creation
        auto_op._ensure_kernel_model()
        
        # Check output models
        output_models = list(auto_op._kernel_model.output_models)
        assert len(output_models) == 1
        
        # Check output
        out0 = output_models[0]
        assert out0.definition.name == "output"
        assert out0.datatype == DataType["UINT16"]
        assert out0.tensor_dims == (1, 32)  # From NumFilters
    
    def test_parameter_binding(self, auto_op):
        """Test parameter extraction from node attributes."""
        # Force model creation
        auto_op._ensure_kernel_model()
        
        # Check that parameters were bound
        assert auto_op._kernel_model.parameter_binding is not None
        params = auto_op._kernel_model.parameter_binding.parameters
        assert "NumChannels" in params
        assert params["NumChannels"] == 64
        assert "NumFilters" in params
        assert params["NumFilters"] == 32
    
    def test_sdim_configuration_legacy(self, auto_op):
        """Test SDIM configuration from legacy SIMD attribute."""
        # Force model creation
        auto_op._ensure_kernel_model()
        
        # Check SDIM was applied from SIMD attribute
        input_models = list(auto_op._kernel_model.input_models)
        if input_models:
            # SIMD=8 should be applied to inputs, but only to second dimension
            # First dimension (batch) remains 1
            assert input_models[0].sdim == (1, 8)
    
    def test_sdim_configuration_modern(self, sample_kernel_def):
        """Test SDIM configuration with interface-specific attributes."""
        # Create node with modern attributes
        node = MagicMock()
        node.name = "TestOp_1"
        node.attribute = []
        
        # Add modern SDIM attributes with FINN-standard names
        attrs = {
            "backend": "fpgadataflow",
            "Batch": 1,
            "NumChannels": 64,
            "NumFilters": 32,
            "inputDataType": "UINT8",    # Regular input
            "weightDataType": "INT8",    # Weight input
            "outputDataType": "UINT16",  # Output
            "input_sdim": 16,  # Interface-specific SDIM
        }
        
        for name, value in attrs.items():
            attr = MagicMock()
            attr.name = name
            attr.i = value if isinstance(value, int) else 0
            attr.s = value if isinstance(value, str) else ""
            node.attribute.append(attr)
        
        # Create op with proper attribute access
        class TestOp(AutoHWCustomOp):
            def get_nodeattr(self, name: str):
                for attr in self.onnx_node.attribute:
                    if attr.name == name:
                        return attr.i if attr.i != 0 else attr.s
                return None
            
            def _extract_input_specs(self):
                specs = {}
                specs["input"] = (
                    (self.get_nodeattr("Batch"), self.get_nodeattr("NumChannels")),
                    self._get_interface_datatype("input", is_input=True)
                )
                if len(self._kernel_def.input_definitions) > 1:
                    specs["weights"] = (
                        (self.get_nodeattr("NumChannels"), self.get_nodeattr("NumFilters")),
                        self._get_interface_datatype("weights", is_input=True)
                    )
                return specs
            
            def _extract_output_specs(self):
                return {
                    "output": (
                        (self.get_nodeattr("Batch"), self.get_nodeattr("NumFilters")),
                        self._get_interface_datatype("output", is_input=False)
                    )
                }
            
            def _extract_parameter_binding(self):
                params = {}
                for param_name in ["Batch", "NumChannels", "NumFilters"]:
                    value = self.get_nodeattr(param_name)
                    if value is not None:
                        params[param_name] = value
                return ParameterBinding(params) if params else None
        
        op = TestOp(node, sample_kernel_def)
        op._ensure_kernel_model()
        
        # Check SDIM configuration
        assert op._sdim_config.get("input") == 16


# Test FINN Methods

class TestFINNMethods:
    """Test FINN abstract method implementations."""
    
    def test_get_input_datatype(self, auto_op):
        """Test input datatype retrieval."""
        # First input
        dtype0 = auto_op.get_input_datatype(0)
        assert dtype0 == DataType["UINT8"]
        
        # Second input (weights)
        dtype1 = auto_op.get_input_datatype(1)
        assert dtype1 == DataType["INT8"]
        
        # Out of bounds
        with pytest.raises(IndexError):
            auto_op.get_input_datatype(2)
    
    def test_get_output_datatype(self, auto_op):
        """Test output datatype retrieval."""
        # First output
        dtype0 = auto_op.get_output_datatype(0)
        assert dtype0 == DataType["UINT16"]
        
        # Out of bounds
        with pytest.raises(IndexError):
            auto_op.get_output_datatype(1)
    
    def test_get_normal_input_shape(self, auto_op):
        """Test normal input shape retrieval."""
        # First input
        shape0 = auto_op.get_normal_input_shape(0)
        assert shape0 == [1, 64]  # Batch=1, NumChannels=64
        
        # Second input (weights)
        shape1 = auto_op.get_normal_input_shape(1)
        assert shape1 == [64, 32]  # NumChannels x NumFilters
    
    def test_get_normal_output_shape(self, auto_op):
        """Test normal output shape retrieval."""
        shape = auto_op.get_normal_output_shape(0)
        assert shape == [1, 32]  # Batch=1, NumFilters=32
    
    def test_get_folded_input_shape(self, auto_op):
        """Test folded input shape calculation."""
        # Force model creation with SDIM
        auto_op._ensure_kernel_model()
        
        # Get folded shape
        folded = auto_op.get_folded_input_shape(0)
        
        # Expected: [num_blocks..., folded_block_dims...]
        # Input has 2D: (batch=1, channels=64) with SDIM=(1, 8)
        # block_dims = (1, 64), num_blocks = (1, 1)
        # folded_block = block_dims / sdim = (1/1, 64/8) = (1, 8)
        assert len(folded) == 4  # 2 num_blocks + 2 folded_block
        assert folded == [1, 1, 1, 8]  # [num_blocks_batch, num_blocks_ch, folded_batch, folded_ch]
    
    def test_get_folded_output_shape(self, auto_op):
        """Test folded output shape calculation."""
        folded = auto_op.get_folded_output_shape(0)
        
        # Output should have minimal folding without explicit streaming_rate
        assert len(folded) >= 2
    
    def test_get_instream_width(self, auto_op):
        """Test input stream width calculation."""
        # Force model creation
        auto_op._ensure_kernel_model()
        
        # First input: UINT8 with SIMD=8
        width = auto_op.get_instream_width(0)
        expected = 8 * 8  # 8 bits * SIMD=8
        assert width == expected
    
    def test_get_outstream_width(self, auto_op):
        """Test output stream width calculation."""
        # UINT16 output
        width = auto_op.get_outstream_width(0)
        assert width >= 16  # At least datatype width
    
    def test_get_number_output_values(self, auto_op):
        """Test total output value count."""
        count = auto_op.get_number_output_values()
        # 1 batch * 32 filters = 32
        assert count == 32


# Test Node Attributes

class TestNodeAttributes:
    """Test node attribute management."""
    
    def test_get_nodeattr_types(self, auto_op):
        """Test attribute type definitions."""
        attrs = auto_op.get_nodeattr_types()
        
        # Check FINN-standard datatype attributes
        assert "inputDataType" in attrs     # Regular input
        assert "weightDataType" in attrs    # Weight input
        assert "outputDataType" in attrs    # Output
        
        # Check standard attributes
        assert "SIMD" in attrs
        assert "PE" in attrs
        
        # Check attribute types
        assert attrs["inputDataType"][0] == "s"    # String
        assert attrs["inputDataType"][1] == True   # Required
        assert attrs["weightDataType"][0] == "s"   # String
        assert attrs["weightDataType"][1] == True  # Required
        assert attrs["outputDataType"][0] == "s"   # String
        assert attrs["outputDataType"][1] == True  # Required
        assert attrs["SIMD"][0] == "i"             # Integer
        assert attrs["SIMD"][1] == False           # Optional
    
    def test_has_weight_inputs(self, auto_op):
        """Test weight input detection."""
        assert auto_op._has_weight_inputs() == True
        
        # Test without weights
        kernel_def = KernelDefinition(name="no_weights")
        kernel_def.add_input(InputDefinition(
            name="input",
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            block_dims_expr=fixed_tiles(256)
        ))
        kernel_def.add_output(OutputDefinition(
            name="output",
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            block_dims_expr=fixed_tiles(256)
        ))
        
        # Create minimal concrete subclass
        class NoWeightsOp(AutoHWCustomOp):
            def _extract_input_specs(self):
                return {
                    "input": (
                        (1, 256),
                        self._get_interface_datatype("input", is_input=True)
                    )
                }
            
            def _extract_output_specs(self):
                return {
                    "output": (
                        (1, 256),
                        self._get_interface_datatype("output", is_input=False)
                    )
                }
            
            def _extract_parameter_binding(self):
                return None
            
            def get_nodeattr(self, name: str):
                # Use parent's mock node
                return auto_op.get_nodeattr(name)
        
        op_no_weights = NoWeightsOp(auto_op.onnx_node, kernel_def)
        assert op_no_weights._has_weight_inputs() == False


# Test Legacy Compatibility

class TestLegacyCompatibility:
    """Test legacy SIMD/PE compatibility."""
    
    def test_get_legacy_simd(self, auto_op):
        """Test SIMD extraction from input SDIM."""
        auto_op._ensure_kernel_model()
        simd = auto_op._get_legacy_simd()
        assert simd == 8  # From SIMD attribute
    
    def test_get_legacy_pe(self, auto_op):
        """Test PE extraction from weight/output SDIM."""
        auto_op._ensure_kernel_model()
        pe = auto_op._get_legacy_pe()
        # Should get from weights since they exist
        assert pe >= 1


# Test Resource Estimation

class TestResourceEstimation:
    """Test resource estimation methods."""
    
    def test_bram_estimation(self, auto_op):
        """Test BRAM estimation."""
        bram = auto_op.bram_estimation()
        assert isinstance(bram, int)
        assert bram >= 0
    
    def test_lut_estimation(self, auto_op):
        """Test LUT estimation."""
        luts = auto_op.lut_estimation()
        assert isinstance(luts, int)
        assert luts >= 100  # Minimum threshold
    
    def test_dsp_estimation(self, auto_op):
        """Test DSP estimation."""
        dsps = auto_op.dsp_estimation()
        assert isinstance(dsps, int)
        assert dsps >= 0
    
    def test_uram_estimation(self, auto_op):
        """Test URAM estimation."""
        urams = auto_op.uram_estimation()
        assert isinstance(urams, int)
        assert urams >= 0
    
    def test_uram_estimation_with_ultra(self, auto_op):
        """Test URAM estimation with ultra RAM style."""
        # Save original get_nodeattr
        original_get_nodeattr = auto_op.get_nodeattr
        
        # Override get_nodeattr to return "ultra" for ram_style
        def mock_get_nodeattr(name):
            if name == "ram_style":
                return "ultra"
            return original_get_nodeattr(name)
        
        auto_op.get_nodeattr = mock_get_nodeattr
        
        # Mock BRAM estimation to return non-zero
        with patch.object(auto_op, 'bram_estimation', return_value=10):
            urams = auto_op.uram_estimation()
            assert urams > 0  # Should convert BRAMs to URAMs
    
    def test_get_exp_cycles(self, auto_op):
        """Test expected cycle calculation."""
        cycles = auto_op.get_exp_cycles()
        assert isinstance(cycles, int)
        assert cycles > 0


# Test Optional Methods

class TestOptionalMethods:
    """Test optional method implementations."""
    
    def test_verify_node(self, auto_op):
        """Test node verification."""
        messages = auto_op.verify_node()
        assert isinstance(messages, list)
        
        # Check for expected messages
        backend_msg = any("Backend" in msg for msg in messages)
        assert backend_msg
        
        # Check for datatype messages
        dtype_msgs = [msg for msg in messages if "datatype" in msg.lower()]
        assert len(dtype_msgs) >= 2  # At least input and output
    
    def test_execute_node(self, auto_op):
        """Test node execution (pass-through)."""
        # Create context with input tensor
        context = {
            "input_tensor": np.random.rand(1, 64).astype(np.float32)
        }
        
        # Execute
        auto_op.execute_node(context, None)
        
        # Check output was created
        assert "output_tensor" in context
        assert np.array_equal(context["output_tensor"], context["input_tensor"])


# Test Error Handling

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_datatype_attribute(self, sample_kernel_def):
        """Test error when datatype attribute is missing."""
        # Create node without datatype attributes
        node = MagicMock()
        node.attribute = []
        node.name = "BadOp"
        
        # Only add non-datatype attributes
        attr = MagicMock()
        attr.name = "NumChannels"
        attr.i = 64
        attr.s = ""
        node.attribute.append(attr)
        
        class TestOp(AutoHWCustomOp):
            def get_nodeattr(self, name: str):
                for attr in self.onnx_node.attribute:
                    if attr.name == name:
                        return attr.i if attr.i != 0 else attr.s
                return None
            
            def _extract_input_specs(self):
                # This should fail because no datatype attributes are set
                specs = {}
                specs["input"] = (
                    (1, 64),
                    self._get_interface_datatype("input", is_input=True)
                )
                if len(self._kernel_def.input_definitions) > 1:
                    specs["weights"] = (
                        (64, 32),
                        self._get_interface_datatype("weights", is_input=True)
                    )
                return specs
            
            def _extract_output_specs(self):
                return {
                    "output": (
                        (1, 32),
                        self._get_interface_datatype("output", is_input=False)
                    )
                }
            
            def _extract_parameter_binding(self):
                return ParameterBinding({"NumChannels": 64})
        
        op = TestOp(node, sample_kernel_def)
        
        # Should raise ValueError when trying to create model
        with pytest.raises(ValueError, match="No datatype specified"):
            op._ensure_kernel_model()
    
    def test_invalid_folding(self, auto_op):
        """Test error when block dims not divisible by SDIM."""
        # Force model creation
        auto_op._ensure_kernel_model()
        
        # Manually set invalid SDIM
        input_models = list(auto_op._kernel_model.input_models)
        if input_models:
            # Set SDIM that doesn't divide block_dims evenly
            # Need to match dimensionality - input has 2 dims (batch, channels)
            input_models[0].sdim = (1, 7)  # 64 not divisible by 7
            
            # Should raise ValueError
            with pytest.raises(ValueError, match="not divisible by SDIM"):
                auto_op.get_folded_input_shape(0)
    
    def test_empty_kernel_definition(self):
        """Test with empty kernel definition."""
        # Create empty kernel
        empty_kernel = KernelDefinition(name="empty")
        
        # Create node
        node = MagicMock()
        node.attribute = []
        node.input = []
        node.output = []
        
        # Create minimal concrete subclass for empty kernel
        class EmptyOp(AutoHWCustomOp):
            def _extract_input_specs(self):
                # Empty kernel has no inputs
                return {}
            
            def _extract_output_specs(self):
                # Empty kernel has no outputs
                return {}
            
            def _extract_parameter_binding(self):
                return None  # No parameters
        
        # Create op
        op = EmptyOp(node, empty_kernel)
        
        # Should handle gracefully
        assert op.get_number_output_values() == 0
        assert op.bram_estimation() == 0


# Integration Tests

class TestIntegration:
    """Integration tests with complete scenarios."""
    
    def test_complete_workflow(self, sample_kernel_def):
        """Test complete workflow from definition to resource estimation."""
        # Create node with all attributes
        node = MagicMock()
        node.name = "CompleteOp"
        node.input = ["data_in", "weights_in"]
        node.output = ["data_out"]
        
        attrs = {
            "backend": "fpgadataflow",
            "Batch": 1,
            "NumChannels": 128,
            "NumFilters": 64,
            "inputDataType": "INT16",   # Regular input
            "weightDataType": "INT8",   # Weight input
            "outputDataType": "INT32",  # Output
            "input_sdim": 16,
            "weights_sdim": 8,
        }
        
        node.attribute = []
        for name, value in attrs.items():
            attr = MagicMock()
            attr.name = name
            if isinstance(value, int):
                attr.i = value
                attr.s = ""
            else:
                attr.s = value
                attr.i = 0
            node.attribute.append(attr)
        
        # Create custom op class
        class CompleteOp(AutoHWCustomOp):
            def get_nodeattr(self, name: str):
                for attr in self.onnx_node.attribute:
                    if attr.name == name:
                        return attr.i if attr.i != 0 else attr.s
                return None
            
            def _extract_input_specs(self):
                specs = {}
                specs["input"] = (
                    (1, self.get_nodeattr("NumChannels")),
                    self._get_interface_datatype("input", is_input=True)
                )
                if len(self._kernel_def.input_definitions) > 1:
                    specs["weights"] = (
                        (self.get_nodeattr("NumChannels"), self.get_nodeattr("NumFilters")),
                        self._get_interface_datatype("weights", is_input=True)
                    )
                return specs
            
            def _extract_output_specs(self):
                return {
                    "output": (
                        (1, self.get_nodeattr("NumFilters")),
                        self._get_interface_datatype("output", is_input=False)
                    )
                }
            
            def _extract_parameter_binding(self):
                params = {}
                for param_name in ["Batch", "NumChannels", "NumFilters"]:
                    value = self.get_nodeattr(param_name)
                    if value is not None:
                        params[param_name] = value
                return ParameterBinding(params) if params else None
        
        # Create operation
        op = CompleteOp(node, sample_kernel_def)
        
        # Test all FINN methods
        assert op.get_input_datatype(0) == DataType["INT16"]
        assert op.get_input_datatype(1) == DataType["INT8"]
        assert op.get_output_datatype(0) == DataType["INT32"]
        
        assert op.get_normal_input_shape(0) == [1, 128]
        assert op.get_normal_output_shape(0) == [1, 64]
        
        # Stream widths
        in_width = op.get_instream_width(0)
        assert in_width == 16 * 16  # INT16 * SDIM=16
        
        # Resource estimation
        assert op.bram_estimation() >= 0
        assert op.lut_estimation() > 0
        assert op.get_exp_cycles() > 0
        
        # Verification
        messages = op.verify_node()
        assert len(messages) > 0
        assert any("✓" in msg for msg in messages)  # Some success messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])