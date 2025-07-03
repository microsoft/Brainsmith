############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Abstract base class for AutoHWCustomOp inference transforms.

Provides standardized patterns for converting ONNX nodes to AutoHWCustomOp
implementations. Pure conversion focus - no parameter optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import warnings

from onnx import helper, NodeProto
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


class InferAutoHWCustomOp(Transformation, ABC):
    """
    Abstract base class for converting ONNX nodes to AutoHWCustomOp implementations.
    
    Pure conversion focus - identifies matching ONNX nodes and replaces them with
    AutoHWCustomOp equivalents. No parameter optimization.
    """
    
    def __init__(self, target_domain: str = "hls"):
        """
        Args:
            target_domain: "hls" or "rtl" for implementation selection
        """
        super().__init__()
        self.target_domain = target_domain
        self._conversion_stats = {"converted": 0, "skipped": 0, "failed": 0}

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        """Main transformation entry point."""
        graph = model.graph
        graph_modified = False
        
        for node_idx, node in enumerate(list(graph.node)):
            try:
                if self.matches_node(model, node):
                    if self.can_convert_node(model, node):
                        # Simple conversion - no optimization
                        new_node = self.convert_to_auto_hw_custom_op(model, node)
                        
                        # Replace in graph
                        graph.node.insert(node_idx, new_node)
                        graph.node.remove(node)
                        graph_modified = True
                        self._conversion_stats["converted"] += 1
                    else:
                        self._conversion_stats["skipped"] += 1
            except Exception as e:
                self._conversion_stats["failed"] += 1
                warnings.warn(f"Failed to convert {node.name}: {e}")
        
        self.log_conversion_summary()
        return (model, graph_modified)

    # Required abstract methods - minimal interface
    @abstractmethod
    def matches_node(self, model: ModelWrapper, node: NodeProto) -> bool:
        """Return True if this node type should be converted."""
        pass

    @abstractmethod 
    def get_auto_hw_custom_op_name(self) -> str:
        """Return the AutoHWCustomOp class name (e.g., 'ThresholdingAxi')."""
        pass

    @abstractmethod
    def get_domain_base(self) -> str:
        """Return domain base (e.g., 'brainsmith.hw_kernels.auto_thresholding')."""
        pass

    @abstractmethod
    def extract_node_attributes(self, model: ModelWrapper, node: NodeProto) -> Dict[str, Any]:
        """Extract necessary attributes from original node for AutoHWCustomOp."""
        pass

    # Optional hooks with sensible defaults
    def can_convert_node(self, model: ModelWrapper, node: NodeProto) -> bool:
        """Additional validation before conversion. Override if needed."""
        return True

    def convert_to_auto_hw_custom_op(self, model: ModelWrapper, node: NodeProto) -> NodeProto:
        """Convert node to AutoHWCustomOp - pure conversion, no optimization."""
        
        # Extract kernel-specific attributes
        kernel_attrs = self.extract_node_attributes(model, node)
        
        # Add required FINN attributes
        finn_attrs = self.create_basic_finn_attributes(model, node)
        
        # Combine attributes
        all_attrs = {**kernel_attrs, **finn_attrs}
        
        # Create new node
        new_node = helper.make_node(
            self.get_auto_hw_custom_op_name(),
            inputs=node.input,
            outputs=node.output,
            domain=self.get_target_domain(),
            name=f"Auto{self.get_auto_hw_custom_op_name()}_{node.name}",
            **all_attrs
        )
        
        # Build the KernelModel to ensure proper initialization
        self._build_kernel_model(model, new_node)
        
        return new_node

    def _build_kernel_model(self, model: ModelWrapper, node: NodeProto) -> None:
        """
        Build and configure the KernelModel for proper initialization.
        
        This method ensures the AutoHWCustomOp has a properly configured KernelModel
        by creating the custom op instance and initializing it with default SDIM settings.
        
        Args:
            model: ONNX model wrapper
            node: The newly created AutoHWCustomOp node
        """
        try:
            # Get the custom op instance
            custom_op = getCustomOp(node)
            
            # Trigger KernelModel creation by calling update_node_model
            if hasattr(custom_op, 'update_node_model'):
                custom_op.update_node_model(model)
            
            # Configure default SDIM for all inputs (set to 1 for all dimensions)
            if hasattr(custom_op, '_kernel_model') and custom_op._kernel_model:
                kernel_model = custom_op._kernel_model
                
                # Get SDIM parameters and configure with defaults
                sdim_params = kernel_model.get_sdim_parameters()
                if sdim_params:
                    default_config = {}
                    for intf_name, param_info in sdim_params.items():
                        # Set SDIM to 1 for all free dimensions (conservative default)
                        default_config[intf_name] = 1
                    
                    # Apply the default configuration
                    kernel_model.configure_sdim(default_config)
                    
                    # Recompute output rates after SDIM configuration
                    kernel_model.compute_output_rates()
        
        except Exception as e:
            # Log warning but don't fail the transformation
            warnings.warn(f"Failed to build KernelModel for {node.name}: {e}")

    def create_basic_finn_attributes(self, model: ModelWrapper, node: NodeProto) -> Dict[str, Any]:
        """Create minimal required FINN attributes."""
        attrs = {"backend": "fpgadataflow"}
        
        # Add datatype attributes from tensors
        if node.input:
            attrs["inputDataType"] = model.get_tensor_datatype(node.input[0]).name
        if len(node.input) > 1:
            attrs["weightDataType"] = model.get_tensor_datatype(node.input[1]).name
        if node.output:
            attrs["outputDataType"] = model.get_tensor_datatype(node.output[0]).name
        
        return attrs

    def get_target_domain(self) -> str:
        """Construct target domain."""
        return f"{self.get_domain_base()}.{self.target_domain}"

    def log_conversion_summary(self):
        """Simple conversion statistics."""
        stats = self._conversion_stats
        if stats["converted"] > 0 or stats["failed"] > 0:
            print(f"AutoHWCustomOp conversion: {stats['converted']} converted, "
                  f"{stats['skipped']} skipped, {stats['failed']} failed")

    # Utility helper methods
    def extract_channels_from_shape(self, model: ModelWrapper, tensor_name: str, 
                                   channels_last: bool = True) -> int:
        """Simple channel extraction from tensor shape."""
        shape = model.get_tensor_shape(tensor_name)
        if not shape:
            raise ValueError(f"No shape available for tensor {tensor_name}")
        
        if channels_last:
            return shape[-1]
        else:
            return shape[1] if len(shape) > 1 else shape[0]

    def get_initializer_info(self, model: ModelWrapper, tensor_name: str) -> Dict[str, Any]:
        """Get basic initializer information."""
        initializer = model.get_initializer(tensor_name)
        shape = model.get_tensor_shape(tensor_name)
        
        return {
            "has_initializer": initializer is not None,
            "shape": shape,
            "datatype": model.get_tensor_datatype(tensor_name)
        }

    def validate_integer_datatypes(self, model: ModelWrapper, node: NodeProto) -> bool:
        """Check that all node inputs/outputs have integer datatypes."""
        all_tensors = list(node.input) + list(node.output)
        for tensor_name in all_tensors:
            dt = model.get_tensor_datatype(tensor_name)
            if not dt.is_integer():
                return False
        return True