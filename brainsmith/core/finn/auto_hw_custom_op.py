############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Base class for automatic hardware custom operators.

Uses the direct factory flow for clean model creation:

    KernelSchema + TensorContext + NodeAttrs → DirectFactory → KernelModel

This architecture eliminates intermediate objects and provides a direct
path from inputs to model with all validation happening inside the factory.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence, Set

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.core.dataflow import (
    KernelSchema,
    KernelValidator,
    TensorContext
)
from brainsmith.core.dataflow.direct_factory import DirectKernelFactory
from brainsmith.core.dataflow.models import (
    KernelModel,
    InputModel,
    OutputModel,
    update_kernel_stream_config
)
from brainsmith.core.dataflow.shape_utils import (
    create_folded_shape,
    calculate_stream_width
)
from brainsmith.core.dataflow.types import prod
from brainsmith.core.dataflow.schema_compiler import SchemaCompiler, CompiledSchema


class AutoHWCustomOp(HWCustomOp, ABC):
    """Base class for automatic hardware custom operators.
    
    Key features:
    - Clean contextualized model creation flow
    - Automatic cache invalidation on nodeattr changes
    - Comprehensive validation at each step
    
    Subclasses must:
    - Define kernel_schema class attribute with KernelSchema instance
    - Implement abstract methods from HWCustomOp base class
    """
    
    # =============================================================================
    # Class Attributes
    # =============================================================================
    
    # Subclasses must define this class attribute
    kernel_schema: KernelSchema = None
    
    # =============================================================================
    # Initialization
    # =============================================================================
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize with contextualized architecture."""
        super().__init__(onnx_node, **kwargs)
        
        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )
        
        # Cache for model
        self._kernel_model: Optional[KernelModel] = None
        
        # Use compiled schema for optimized runtime operations
        self._compiled_schema: CompiledSchema = SchemaCompiler.compile(self.kernel_schema)
        
        # Validator for multi-phase validation
        self._validator = KernelValidator()
    
    # =============================================================================
    # Public API - Model Access
    # =============================================================================
    
    def get_kernel_model(self) -> KernelModel:
        """Get the cached kernel model.
        
        Returns cached model if available, otherwise raises error.
        Call refresh_kernel_model() first to populate cache.
        """
        if self._kernel_model is None:
            raise RuntimeError(
                "Kernel model not initialized. Call refresh_kernel_model() first."
            )
        return self._kernel_model
    
    
    # =============================================================================
    # Public API - Model Refresh
    # =============================================================================
    
    def refresh_kernel_model(self, model: ModelWrapper) -> None:
        """Refresh kernel model using direct factory.
        
        Flow: Schema + TensorContext + NodeAttrs → DirectFactory → Model
        
        Args:
            model: The global ModelWrapper instance
        """
        # Create tensor context
        tensor_context = TensorContext.from_model_wrapper(self.onnx_node, model)
        
        # Get all nodeattrs
        nodeattrs = self._get_all_nodeattrs()
        
        # Direct creation - no intermediate objects
        self._kernel_model = DirectKernelFactory.create_model(
            self.kernel_schema,
            tensor_context,
            nodeattrs
        )
    
    
    # =============================================================================
    # Override - Nodeattr Management with Change Detection
    # =============================================================================
    
    def set_nodeattr(self, name: str, value: Any) -> None:
        """Override to invalidate cache on relevant changes."""
        # Get old value if it exists
        try:
            old_value = self.get_nodeattr(name)
        except (AttributeError, Exception):
            old_value = None
        
        super().set_nodeattr(name, value)
        
        # Check if this affects model
        if self._compiled_schema.is_model_affecting(name):
            if old_value != value:
                # Clear cache
                self._kernel_model = None
    
    # =============================================================================
    # Override - HWCustomOp Interface Methods (Delegating to Cached Model)
    # =============================================================================
    
    def get_folded_output_shape(self, ind=0):
        """Get folded output shape using cached model."""
        model = self.get_kernel_model()
        output = model.outputs[ind]
        return create_folded_shape(output.tensor_dims, output.block_dims)
    
    def get_number_output_values(self):
        """Get total output values using cached model."""
        model = self.get_kernel_model()
        total = 0
        for output in model.outputs:
            total += prod(output.tensor_dims)
        return total
    
    def get_exp_cycles(self):
        """Get expected cycles using cached model."""
        model = self.get_kernel_model()
        return model.initiation_interval
    
    def get_input_datatype(self, ind=0) -> DataType:
        """Get input datatype from cached model."""
        return self.get_kernel_model().inputs[ind].datatype
    
    def get_output_datatype(self, ind=0) -> DataType:
        """Get output datatype from cached model."""
        return self.get_kernel_model().outputs[ind].datatype
    
    def get_normal_input_shape(self, ind=0) -> List[int]:
        """Get normal input shape from cached model."""
        return list(self.get_kernel_model().inputs[ind].tensor_dims)
    
    def get_normal_output_shape(self, ind=0) -> List[int]:
        """Get normal output shape from cached model."""
        return list(self.get_kernel_model().outputs[ind].tensor_dims)
    
    def get_folded_input_shape(self, ind=0) -> List[int]:
        """Get folded input shape using cached model."""
        model = self.get_kernel_model()
        inp = model.inputs[ind]
        return create_folded_shape(inp.tensor_dims, inp.block_dims)
    
    def get_instream_width(self, ind=0) -> int:
        """Get input stream width in bits."""
        inp = self.get_kernel_model().inputs[ind]
        return calculate_stream_width(inp.streaming_bandwidth, inp.datatype.bitwidth())
    
    def get_outstream_width(self, ind=0) -> int:
        """Get output stream width in bits."""
        out = self.get_kernel_model().outputs[ind]
        return out.streaming_rate * out.datatype.bitwidth()
    
    # =============================================================================
    # Override - Template and Performance Methods
    # =============================================================================
    
    def get_template_values(self):
        """Get template values from cached model."""
        model = self.get_kernel_model()
        
        templates = {}
        
        # Input parameters
        for inp in model.inputs:
            templates[f"{inp.name}_block_dims"] = inp.block_dims
            templates[f"{inp.name}_stream_dims"] = inp.stream_dims
        
        # Output parameters  
        for out in model.outputs:
            templates[f"{out.name}_block_dims"] = out.block_dims
        
        # Kernel parameters
        templates.update(model.parameters)
        
        return templates
    
    def get_sdim_parameters(self):
        """Get SDIM configuration options."""
        params = {}
        
        for i, inp_schema in enumerate(self.kernel_schema.inputs):
            if hasattr(inp_schema, 'stream_tiling') and inp_schema.stream_tiling:
                params[inp_schema.name] = {
                    "dimensions": len(inp_schema.stream_tiling),
                    "template": inp_schema.stream_tiling
                }
        
        return params
    
    def calculate_performance_for_sdim(
        self,
        sdim_config: Dict[str, Union[int, List[int]]]
    ) -> Dict[str, Any]:
        """Calculate performance metrics for specific SDIM config."""
        # Get base model
        base_model = self.get_kernel_model()
        
        # Apply SDIM configuration
        configured_model = update_kernel_stream_config(base_model, sdim_config)
        
        # Return metrics
        return configured_model.calculate_metrics()
    
    def get_input_bandwidth_mbps(self, sdim_config=None):
        """Get total input bandwidth."""
        if sdim_config:
            model = update_kernel_stream_config(
                self.get_kernel_model(),
                sdim_config
            )
        else:
            model = self.get_kernel_model()
        
        total_bits = sum(inp.bandwidth_bits for inp in model.inputs)
        return total_bits * model.clock_freq_mhz / 8.0
    
    def get_output_bandwidth_mbps(self):
        """Get total output bandwidth."""
        model = self.get_kernel_model()
        total_bits = sum(out.bandwidth_bits for out in model.outputs)
        return total_bits * model.clock_freq_mhz / 8.0
    
    # =============================================================================
    # Private - Helper Methods
    # =============================================================================
    
    def _get_all_nodeattrs(self) -> Dict[str, Any]:
        """Get all nodeattrs as a dictionary."""
        nodeattrs = {}
        
        for name, (dtype, required, default, *_) in self.get_nodeattr_types().items():
            try:
                value = self.get_nodeattr(name)
                nodeattrs[name] = value
            except (AttributeError, Exception):
                if default is not None:
                    nodeattrs[name] = default
        
        return nodeattrs
    
