############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Base class for automatic hardware custom operators.

Uses cached kernel model state for performance while providing
refresh mechanism via transforms.

Two-Phase Model Creation Flow:

    KernelSchema (static)
           │
           ├─[Phase 1: nodeattr only]─→ ResolvedKernelConfig
           │                                      │
           │                                      ├─[Phase 2: + ModelWrapper]
           │                                      ↓
           │                            TensorContext + DataType resolver
           │                                      │
           │                                      ↓
           └─────────────────────────────→ KernelModel (complete)

Cache invalidation:
  set_nodeattr() → refresh_nodeattr_config() → clears downstream caches
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence, Set

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.core.dataflow import (
    KernelSchema,
    ResolvedInterfaceConfig,
    ResolvedKernelConfig,
    TensorContext,
    KernelModelFactory
)
from brainsmith.core.dataflow.models import (
    KernelModel,
    InputModel,
    OutputModel,
    create_kernel_model,
    update_kernel_stream_config
)
from brainsmith.core.dataflow.template_utils import resolve_template_params
from brainsmith.core.dataflow.shape_utils import (
    create_folded_shape,
    calculate_stream_width
)
from brainsmith.core.dataflow.types import prod
from brainsmith.core.dataflow.schema_compiler import SchemaCompiler, CompiledSchema


class AutoHWCustomOp(HWCustomOp, ABC):
    """Base class for automatic hardware custom operators.
    
    Key features:
    - Two-phase model creation separating nodeattr and ModelWrapper dependencies
    - Automatic cache invalidation on nodeattr changes
    - Efficient partial updates when only nodeattrs change
    
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
        """Initialize with three-level cache system."""
        super().__init__(onnx_node, **kwargs)
        
        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )
        
        # Three levels of cache for two-phase model creation
        self._resolved_config: Optional[ResolvedKernelConfig] = None
        self._tensor_context: Optional[TensorContext] = None  
        self._kernel_model: Optional[KernelModel] = None
        
        # Use compiled schema for optimized runtime operations
        self._compiled_schema: CompiledSchema = SchemaCompiler.compile(self.kernel_schema)
    
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
    
    def refresh_nodeattr_config(self) -> None:
        """Update resolved config from current nodeattrs (Phase 1)."""
        self._resolved_config = self._resolve_from_nodeattrs()
        # Invalidate downstream caches
        self._tensor_context = None
        self._kernel_model = None
    
    def refresh_kernel_model(self, model: ModelWrapper) -> None:
        """Refresh complete kernel model with ModelWrapper info (Phase 2).
        
        This should be called by transforms when shapes or types change.
        
        Args:
            model: The global ModelWrapper instance (not cached)
        """
        # Ensure we have resolved config
        if self._resolved_config is None:
            self.refresh_nodeattr_config()
        
        # Extract tensor context
        self._tensor_context = TensorContext.from_model_wrapper(
            self.onnx_node, model
        )
        
        # Build datatype resolver
        datatype_resolver = self._build_datatype_resolver()
        
        # Create complete model using factory
        self._kernel_model = KernelModelFactory.create_model(
            self._resolved_config,
            self._tensor_context,
            datatype_resolver
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
        
        # Check if this affects model - now O(1) lookup using compiled schema
        if self._compiled_schema.is_model_affecting(name):
            if old_value != value:
                # Use targeted invalidation for better performance
                affected_caches = self._compiled_schema.get_affected_caches(name)
                self._invalidate_caches(affected_caches)
    
    def _invalidate_caches(self, cache_names: Set[str]) -> None:
        """Invalidate specific caches based on dependency analysis."""
        if "resolved_config" in cache_names:
            self._resolved_config = None
        if "tensor_context" in cache_names:
            self._tensor_context = None
        if "kernel_model" in cache_names:
            self._kernel_model = None
        
        # If resolved_config is invalidated, downstream caches must also be cleared
        if "resolved_config" in cache_names:
            self._tensor_context = None
            self._kernel_model = None
        # If tensor_context is invalidated, kernel_model must be cleared
        elif "tensor_context" in cache_names:
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
        # Build from scratch each time
        params = {}
        
        for i, inp_schema in enumerate(self.kernel_schema.inputs):
            if hasattr(inp_schema, 'stream_tiling') and inp_schema.stream_tiling:
                # This input supports streaming configuration
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
    # Private - Phase 1: Configuration Resolution (nodeattr-only)
    # =============================================================================
    
    def _resolve_from_nodeattrs(self) -> ResolvedKernelConfig:
        """Create resolved config from schema + nodeattrs."""
        # Resolve each interface
        inputs = []
        for i, schema in enumerate(self.kernel_schema.inputs):
            inputs.append(self._resolve_interface_config(schema, i, is_input=True))
        
        outputs = []
        for i, schema in enumerate(self.kernel_schema.outputs):
            outputs.append(self._resolve_interface_config(schema, i, is_input=False))
        
        # Extract parameters
        parameters = self._extract_parameters()
        
        # Get clock frequency with default
        try:
            clock_freq_mhz = self.get_nodeattr("clock_freq_mhz")
        except (AttributeError, Exception):
            clock_freq_mhz = 100.0
        
        return ResolvedKernelConfig(
            kernel_name=self.kernel_schema.name,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            clock_freq_mhz=clock_freq_mhz
        )
    
    def _resolve_interface_config(
        self,
        schema,
        position: int,
        is_input: bool
    ) -> ResolvedInterfaceConfig:
        """Resolve interface configuration from schema and nodeattrs."""
        # For inputs, handle stream_tiling
        stream_params = None
        if is_input and hasattr(schema, 'stream_tiling') and schema.stream_tiling:
            stream_params = self._resolve_template_params(schema.stream_tiling)
        
        return ResolvedInterfaceConfig(
            name=schema.name,
            position=position,
            block_params=self._resolve_template_params(schema.block_tiling or [":"]),
            stream_params=stream_params,
            datatype_attr=schema.get_datatype_attr(position),
            is_weight=getattr(schema, 'is_weight', False),
            optional=schema.optional
        )
    
    def _resolve_template_params(
        self,
        template: List[Union[int, str]]
    ) -> List[Union[int, str]]:
        """Resolve nodeattr references in template to their values."""
        return resolve_template_params(
            template, 
            self.get_nodeattr,
            self.get_nodeattr_types()
        )
    
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameter values from nodeattrs using compiled schema."""
        params = {}
        
        # Use compiled schema to get only parameters that actually exist
        for param in self._compiled_schema.all_parameters:
            # Skip datatype and performance parameters
            if param.endswith("Datatype") or param == "clock_freq_mhz":
                continue
                
            try:
                value = self.get_nodeattr(param)
                # Unwrap single-element lists
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                params[param] = value
            except (AttributeError, Exception):
                # Attribute not defined or not set
                pass
        
        return params
    
    # =============================================================================
    # Private - Phase 2: DataType Resolution
    # =============================================================================
    
    def _build_datatype_resolver(self) -> Dict[str, DataType]:
        """Build mapping from datatype attr names to DataType values."""
        resolver = {}
        
        # Resolve datatypes for all interfaces
        for config in self._resolved_config.inputs + self._resolved_config.outputs:
            if config.datatype_attr:
                try:
                    dtype_str = self.get_nodeattr(config.datatype_attr)
                    if dtype_str:
                        resolver[config.datatype_attr] = DataType[dtype_str]
                except (AttributeError, Exception):
                    # Datatype not set, will use default from tensor context
                    pass
        
        return resolver
    
    # =============================================================================
    # Private - Change Detection Support
    # =============================================================================
    
    
