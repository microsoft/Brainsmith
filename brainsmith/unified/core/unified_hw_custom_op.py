############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Unified HWCustomOp Base Class

Provides clean integration between FINN and the Unified Kernel Modeling Framework.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType

from brainsmith.core.dataflow import (
    Kernel, DataflowGraph, Interface, Shape, InterfaceDirection, DataType as CoreDataType
)
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.core.dataflow.dse import (
    ParallelismConfig, ConfigurationSpace, PerformanceEvaluator,
    DSEConstraints
)

from .kernel_definition import KernelDefinition


class UnifiedHWCustomOp(HWCustomOp, ABC):
    """
    Base class for unified hardware custom operators.
    Provides clean integration between FINN and Unified Kernel Framework.
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize unified operator."""
        super().__init__(onnx_node, **kwargs)
        
        # Get kernel definition (subclasses override)
        self.kernel_def = self.get_kernel_definition()
        
        # Validate kernel definition
        errors = self.kernel_def.validate()
        if errors:
            raise ValueError(f"Invalid kernel definition: {errors}")
        
        # Create kernel instance
        self.kernel = self._create_kernel_from_definition()
        
        # Create dataflow graph with single kernel
        self.graph = DataflowGraph()
        self.graph.add_kernel(self.kernel)
        
        # Initialize DSE components
        self.dse_constraints = DSEConstraints()
        self.config_space = None  # Created after shape initialization
        self.evaluator = PerformanceEvaluator()
        
        # State management
        self._current_config = None
        self._optimized = False
        self._shapes_initialized = False
        
        # Initialize from ONNX attributes
        self._initialize_from_onnx()
    
    @abstractmethod
    def get_kernel_definition(self) -> KernelDefinition:
        """
        Return kernel definition for this operator.
        Subclasses must implement this method.
        """
        pass
    
    def _map_interface_type_to_direction(self, intf_type: InterfaceType) -> InterfaceDirection:
        """Map InterfaceType to InterfaceDirection."""
        mapping = {
            InterfaceType.INPUT: InterfaceDirection.INPUT,
            InterfaceType.OUTPUT: InterfaceDirection.OUTPUT,
            InterfaceType.WEIGHT: InterfaceDirection.WEIGHT,
            InterfaceType.CONFIG: InterfaceDirection.CONFIG,
        }
        return mapping.get(intf_type, InterfaceDirection.INPUT)
    
    def _create_kernel_from_definition(self) -> Kernel:
        """Create kernel instance from definition."""
        # Create interfaces from definitions
        interfaces = []
        for intf_def in self.kernel_def.interfaces:
            # Create Interface object
            # Note: Shapes will be set later from ONNX
            intf = Interface(
                name=intf_def.name,
                direction=self._map_interface_type_to_direction(intf_def.type),
                dtype=CoreDataType(name="INT8", bits=8),  # Default, will be set later
                tensor_dims=tuple([1]),  # Placeholder, set from ONNX
                block_dims=tuple([1]),   # Placeholder, set from attributes
            )
            # Store additional metadata
            intf.metadata = {
                'protocol': intf_def.protocol,
                'protocol_config': intf_def.protocol_config,
                'datatype_constraints': intf_def.datatype_constraints,
                'parameter_links': intf_def.parameter_links,
                'definition_metadata': intf_def.metadata
            }
            interfaces.append(intf)
        
        # Create kernel
        kernel = Kernel(name=self.kernel_def.name, interfaces=interfaces)
        
        # Constraints are handled as pragmas in the kernel
        # They will be converted when needed
        
        return kernel
    
    def _initialize_from_onnx(self):
        """Initialize kernel from ONNX node attributes."""
        # Set tensor shapes from ONNX inputs/outputs
        self._set_tensor_shapes()
        
        # Set block shapes from attributes or defaults
        self._set_block_shapes()
        
        # Initialize parallelism
        self._initialize_parallelism()
        
        # Create configuration space after shapes are set
        self.config_space = ConfigurationSpace(self.graph)
        
        # Mark shapes as initialized
        self._shapes_initialized = True
    
    def _set_tensor_shapes(self):
        """Set tensor shapes from ONNX model."""
        # Map ONNX inputs to kernel input interfaces
        input_idx = 0
        for intf in self.kernel.interfaces:
            if intf.direction == InterfaceDirection.INPUT:
                if input_idx < len(self.onnx_node.input):
                    # Get shape from model
                    model_shape = self.get_normal_input_shape(input_idx)
                    intf.tensor_dims = tuple(model_shape)
                    input_idx += 1
        
        # Map ONNX outputs to kernel output interfaces
        output_idx = 0
        for intf in self.kernel.interfaces:
            if intf.direction == InterfaceDirection.OUTPUT:
                if output_idx < len(self.onnx_node.output):
                    # Get shape from model
                    model_shape = self.get_normal_output_shape(output_idx)
                    intf.tensor_dims = tuple(model_shape)
                    output_idx += 1
        
        # Weight interfaces get shapes from attributes or initialization
        for intf in self.kernel.interfaces:
            if intf.direction == InterfaceDirection.WEIGHT:
                # Check for weight shape attribute
                weight_shape_attr = f"{intf.name}_shape"
                if self.has_nodeattr(weight_shape_attr):
                    shape = self.get_nodeattr(weight_shape_attr)
                    intf.tensor_dims = tuple(shape)
    
    def _set_block_shapes(self):
        """Set block shapes from node attributes or compute defaults."""
        for intf in self.kernel.interfaces:
            block_shape_attr = f"{intf.name}_block_shape"
            if self.has_nodeattr(block_shape_attr):
                block_shape = self.get_nodeattr(block_shape_attr)
                intf.block_dims = tuple(block_shape)
            else:
                # Use tensor shape as default block shape
                intf.block_dims = intf.tensor_dims
    
    def _initialize_parallelism(self):
        """Initialize parallelism from attributes or defaults."""
        config = ParallelismConfig()
        
        # Build interface parallelism dictionary
        interface_pars = {}
        for intf in self.kernel.interfaces:
            par_attr = f"{intf.name}_parallelism"
            if self.has_nodeattr(par_attr):
                parallelism = self.get_nodeattr(par_attr)
            else:
                parallelism = 1  # Default
            
            # ParallelismConfig expects (kernel_name, interface_name) as key
            interface_pars[(self.kernel.name, intf.name)] = parallelism
        
        config.interface_pars = interface_pars
        
        # Apply configuration to kernel
        try:
            configured_kernel = config.apply_to_kernel(self.kernel, self.kernel.name)
            # Update our kernel with the configured one
            self.kernel = configured_kernel
            self._current_config = config
        except Exception as e:
            # If initial config fails, keep default
            self._current_config = ParallelismConfig(global_par=1)
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """Define node attributes dynamically from kernel definition."""
        attrs = super().get_nodeattr_types()
        
        # Add interface-specific attributes
        for intf_def in self.kernel_def.interfaces:
            # Parallelism attribute
            attrs[f"{intf_def.name}_parallelism"] = ("i", False, 1)
            
            # Block shape attribute
            attrs[f"{intf_def.name}_block_shape"] = ("ints", False, [])
            
            # Datatype attribute if interface has datatype parameter link
            if "datatype" in intf_def.parameter_links:
                attrs[f"{intf_def.name}_datatype"] = ("s", False, "")
        
        # Add exposed RTL parameters
        for param in self.kernel_def.exposed_parameters:
            # Determine type from default value
            default = self.kernel_def.parameter_defaults.get(param, 0)
            if isinstance(default, int):
                param_range = self.kernel_def.parameter_ranges.get(param, None)
                if param_range:
                    attrs[param] = ("i", False, default, set(range(param_range[0], param_range[1] + 1)))
                else:
                    attrs[param] = ("i", False, default)
            elif isinstance(default, float):
                attrs[param] = ("f", False, default)
            elif isinstance(default, str):
                attrs[param] = ("s", False, default)
            else:
                attrs[param] = ("s", False, str(default))
        
        # Add optimization attributes
        attrs["optimization_objective"] = ("s", False, "balanced", {"throughput", "latency", "balanced", "resources"})
        attrs["auto_optimize"] = ("i", False, 0, {0, 1})
        attrs["target_throughput"] = ("f", False, 0.0)
        attrs["max_luts"] = ("i", False, 0)
        attrs["max_brams"] = ("i", False, 0)
        attrs["max_dsps"] = ("i", False, 0)
        
        return attrs
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self._current_config:
            return {
                'throughput': 0.0,
                'latency': 0,
                'resource_usage': {},
                'bandwidth_requirements': {}
            }
        
        # Evaluate current configuration
        metrics = self.evaluator.evaluate(self._current_config, self.graph)
        
        # Get bandwidth requirements
        bandwidth = self.kernel.bandwidth_requirements()
        
        return {
            'throughput': metrics.throughput,
            'latency': metrics.latency,
            'resource_usage': metrics.resource_usage,
            'bandwidth_requirements': bandwidth,
            'initiation_interval': metrics.initiation_interval,
            'is_schedulable': metrics.is_schedulable
        }
    
    def get_input_datatype(self, ind=0):
        """Get input datatype from attributes or defaults."""
        input_intfs = self.kernel.input_interfaces
        if ind < len(input_intfs):
            intf = input_intfs[ind]
            dt_attr = f"{intf.name}_datatype"
            if self.has_nodeattr(dt_attr):
                return DataType[self.get_nodeattr(dt_attr)]
            # Return default based on constraints
            if intf.metadata.get('datatype_constraints'):
                constraint = intf.metadata['datatype_constraints'][0]
                return DataType[f"{constraint.base_type}{constraint.min_width}"]
        return DataType["UINT8"]  # Default
    
    def get_output_datatype(self, ind=0):
        """Get output datatype from attributes or defaults."""
        output_intfs = self.kernel.output_interfaces
        if ind < len(output_intfs):
            intf = output_intfs[ind]
            dt_attr = f"{intf.name}_datatype"
            if self.has_nodeattr(dt_attr):
                return DataType[self.get_nodeattr(dt_attr)]
            # Return default based on constraints
            if intf.metadata.get('datatype_constraints'):
                constraint = intf.metadata['datatype_constraints'][0]
                return DataType[f"{constraint.base_type}{constraint.min_width}"]
        return DataType["UINT8"]  # Default
    
    def get_instream_width(self, ind=0):
        """Calculate input stream width."""
        input_intfs = self.kernel.input_interfaces
        if ind < len(input_intfs):
            intf = input_intfs[ind]
            dt = self.get_input_datatype(ind)
            # Width = parallelism * datatype_width
            parallelism = self.get_nodeattr(f"{intf.name}_parallelism")
            return parallelism * dt.bitwidth()
        return 8  # Default
    
    def get_outstream_width(self, ind=0):
        """Calculate output stream width."""
        output_intfs = self.kernel.output_interfaces
        if ind < len(output_intfs):
            intf = output_intfs[ind]
            dt = self.get_output_datatype(ind)
            # Width = parallelism * datatype_width
            parallelism = self.get_nodeattr(f"{intf.name}_parallelism")
            return parallelism * dt.bitwidth()
        return 8  # Default
    
    def get_folded_input_shape(self, ind=0):
        """Get folded input shape based on parallelism."""
        input_intfs = self.kernel.input_interfaces
        if ind < len(input_intfs):
            intf = input_intfs[ind]
            if self._current_config:
                kernel_config = self._current_config.kernel_configs[self.kernel.name]
                stream_shape = kernel_config.stream_shapes[intf.name]
                # Calculate folded shape
                tensor_shape = list(intf.tensor_dims)
                block_shape = list(intf.block_dims)
                folded_shape = []
                for t, b in zip(tensor_shape, block_shape):
                    folded_shape.append(t // b)
                folded_shape.append(stream_shape[0])  # Add stream dimension
                return folded_shape
        return [1]
    
    def get_folded_output_shape(self, ind=0):
        """Get folded output shape based on parallelism."""
        output_intfs = self.kernel.output_interfaces
        if ind < len(output_intfs):
            intf = output_intfs[ind]
            if self._current_config:
                kernel_config = self._current_config.kernel_configs[self.kernel.name]
                stream_shape = kernel_config.stream_shapes[intf.name]
                # Calculate folded shape
                tensor_shape = list(intf.tensor_dims)
                block_shape = list(intf.block_dims)
                folded_shape = []
                for t, b in zip(tensor_shape, block_shape):
                    folded_shape.append(t // b)
                folded_shape.append(stream_shape[0])  # Add stream dimension
                return folded_shape
        return [1]
    
    def get_number_output_values(self):
        """Calculate total number of output values."""
        output_intfs = self.kernel.output_interfaces
        if output_intfs:
            intf = output_intfs[0]
            return int(np.prod(intf.tensor_dims))
        return 1
    
    def execute_node(self, context, graph):
        """Execute node - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute_node")
    
    # FINN abstract method implementations
    def get_normal_input_shape(self, ind=0):
        """Get normal (non-folded) input shape."""
        # Default implementation - can be overridden by subclasses
        model = self.onnx_node.graph if hasattr(self.onnx_node, 'graph') else None
        if model and ind < len(self.onnx_node.input):
            input_name = self.onnx_node.input[ind]
            # Try to get shape from model
            # This is a simplified version - real implementation would query the model
            return [1, 32]  # Default shape
        return [1]
    
    def get_normal_output_shape(self, ind=0):
        """Get normal (non-folded) output shape."""
        # Default implementation - can be overridden by subclasses
        model = self.onnx_node.graph if hasattr(self.onnx_node, 'graph') else None
        if model and ind < len(self.onnx_node.output):
            output_name = self.onnx_node.output[ind]
            # Try to get shape from model
            # This is a simplified version - real implementation would query the model
            return [1, 32]  # Default shape
        return [1]
    
    def infer_node_datatype(self, model):
        """Infer node datatypes - required by FINN."""
        # This is typically used to propagate datatypes through the graph
        # For unified implementation, datatypes are managed through attributes
        pass
    
    def code_generation_ipi(self):
        """Generate IP integrator commands - required by RTLBackend."""
        # Default implementation for IPI integration
        return []
    
    # Helper methods for node attributes
    def has_nodeattr(self, name: str) -> bool:
        """Check if node attribute exists."""
        for attr in self.onnx_node.attribute:
            if attr.name == name:
                return True
        return False