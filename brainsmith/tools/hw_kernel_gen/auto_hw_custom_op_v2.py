############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
AutoHWCustomOp base class using Kernel Modeling system.

This module provides a modern implementation of AutoHWCustomOp that leverages
the Kernel Modeling system for all shape, datatype, and performance calculations.
It serves as a base class for auto-generated HWCustomOp implementations,
providing automatic implementation of all FINN-required methods through
delegation to KernelModel interfaces.

Key Features:
- Clean integration with Kernel Modeling system
- Automatic shape and stream width calculations
- SDIM-based parallelism configuration
- Legacy SIMD/PE compatibility
- Resource estimation from performance metrics
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union

# FINN imports
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

# Kernel Modeling imports
from brainsmith.core.dataflow import (
    KernelDefinition,
    KernelModel,
    InputInterface,
    OutputInterface
)
from brainsmith.core.dataflow.base import ParameterBinding


class AutoHWCustomOp(HWCustomOp, ABC):
    """
    Base class for HWCustomOp implementations using Kernel Modeling.
    
    This class bridges FINN's HWCustomOp interface with the Kernel Modeling
    system, providing automatic implementation of all required methods through
    delegation to KernelModel interfaces.
    
    The class implements a three-tier architecture:
    1. Static: KernelDefinition defines interfaces and constraints
    2. Runtime: KernelModel instantiated with concrete types and shapes
    3. Dynamic: SDIM configuration for parallelism control
    
    Attributes:
        _kernel_def: KernelDefinition providing static schema
        _kernel_model: KernelModel with runtime instances (created lazily)
        _sdim_config: Cached SDIM configuration
    """
    
    def __init__(self, onnx_node, kernel_definition: KernelDefinition, **kwargs):
        """
        Initialize AutoHWCustomOp with a KernelDefinition.
        
        Args:
            onnx_node: ONNX node containing attributes and connections
            kernel_definition: Static kernel schema with interface definitions
            **kwargs: Additional arguments passed to HWCustomOp
        """
        super().__init__(onnx_node, **kwargs)
        
        # Store the kernel definition
        self._kernel_def = kernel_definition
        
        # KernelModel created lazily when runtime info available
        self._kernel_model = None
        
        # Cache for SDIM configuration
        self._sdim_config = {}
    
    def _ensure_kernel_model(self):
        """
        Create KernelModel if not already created.
        
        This method extracts runtime information from the ONNX node
        and creates a KernelModel instance. It's called lazily because
        some runtime information may not be available during construction.
        """
        if self._kernel_model is not None:
            return
        
        # Extract runtime specifications
        input_specs = self._extract_input_specs()
        output_specs = self._extract_output_specs()
        param_binding = self._extract_parameter_binding()
        
        # Create kernel model
        self._kernel_model = self._kernel_def.create_model(
            input_specs=input_specs,
            output_specs=output_specs,
            parameter_binding=param_binding
        )
        
        # Apply SDIM configuration
        self._apply_sdim_configuration()
        
        # Apply any legacy attribute mappings
        self._apply_legacy_attributes()
    
    @abstractmethod
    def _extract_input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract input specifications (shape and datatype) from ONNX context.
        
        Subclasses must implement this to extract the actual tensor shapes
        and datatypes for all inputs defined in the KernelDefinition.
        
        Returns:
            Dictionary mapping input names to (shape, datatype) tuples
            
        Notes:
            - Keys should match input names from KernelDefinition.input_definitions
            - Shapes should be extracted from ONNX graph context or node attributes
            - Datatypes can use the _get_interface_datatype() helper method
        """
        pass
    
    @abstractmethod
    def _extract_output_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract output specifications (shape and datatype) from ONNX context.
        
        Subclasses must implement this to extract the actual tensor shapes
        and datatypes for all outputs defined in the KernelDefinition.
        
        Returns:
            Dictionary mapping output names to (shape, datatype) tuples
            
        Notes:
            - Keys should match output names from KernelDefinition.output_definitions
            - Shapes are often derived from input shapes and kernel parameters
            - Datatypes can use the _get_interface_datatype() helper method
        """
        pass
    
    def _extract_parameter_binding(self) -> ParameterBinding:
        """
        Extract kernel parameters from ONNX node attributes.
        
        Only extracts parameters that are defined as node attributes.
        Shape parameters (BDIM/SDIM) are handled by KernelModel configuration,
        not through parameter binding.
        
        Returns:
            ParameterBinding with parameter name->value mappings
        """
        params = {}
        
        # Get all defined node attributes for this kernel
        nodeattr_types = self.get_nodeattr_types()
        
        # Extract only the parameters that are actually node attributes
        # Skip interface datatypes and FINN legacy parameters
        skip_attrs = {'SIMD', 'PE', 'ram_style'}
        
        for attr_name in nodeattr_types:
            # Skip interface datatype attributes (end with DataType)
            if attr_name.endswith('DataType'):
                continue
                
            # Skip FINN legacy attributes
            if attr_name in skip_attrs:
                continue
                
            # Extract the parameter value
            try:
                value = self.get_nodeattr(attr_name)
                if value is not None:
                    params[attr_name] = value
            except AttributeError:
                # Parameter not found - skip it
                continue
        
        return ParameterBinding(params) if params else None
    
    def _get_finn_attribute_mapping(self) -> Dict[str, str]:
        """
        Generate FINN attribute names for all interfaces.
        
        Returns:
            Dictionary mapping interface names to FINN attribute names
        """
        attrs = {}
        
        # Get categorized inputs
        regular_inputs = self._kernel_def.get_regular_inputs()
        weight_inputs = self._kernel_def.get_weight_inputs()
        outputs = self._kernel_def.output_definitions
        
        # Map regular inputs
        if len(regular_inputs) == 1:
            attrs[regular_inputs[0].name] = "inputDataType"
        else:
            for i, inp in enumerate(regular_inputs):
                attrs[inp.name] = f"input{i}DataType"
        
        # Map weights
        if len(weight_inputs) == 1:
            attrs[weight_inputs[0].name] = "weightDataType"
        else:
            for i, weight in enumerate(weight_inputs):
                attrs[weight.name] = f"weight{i}DataType"
        
        # Map outputs
        if len(outputs) == 1:
            attrs[outputs[0].name] = "outputDataType"
        else:
            for i, out in enumerate(outputs):
                attrs[out.name] = f"output{i}DataType"
        
        return attrs
    
    def _get_interface_datatype(self, interface_name: str, is_input: bool = True) -> DataType:
        """
        Helper method to extract datatype using FINN-standard attribute pattern.
        """
        # Get the mapping
        finn_attrs = self._get_finn_attribute_mapping()
        
        # Find the attribute name for this interface
        if interface_name not in finn_attrs:
            raise ValueError(f"Interface '{interface_name}' not found in kernel definition")
        
        attr_name = finn_attrs[interface_name]
        
        # Get datatype
        dtype_str = self.get_nodeattr(attr_name)
        if not dtype_str:
            raise ValueError(f"No datatype specified. Expected node attribute '{attr_name}'.")
        
        return DataType[dtype_str]
    
    def _extract_sdim_configuration(self) -> Dict[str, Union[int, List[int]]]:
        """
        Extract SDIM configuration from node attributes.
        
        Checks for interface-specific SDIM attributes first, then falls
        back to legacy SIMD attribute for backward compatibility.
        
        Returns:
            Dictionary mapping interface names to SDIM values
        """
        sdim_config = {}
        
        # Try interface-specific SDIM attributes
        for inp_def in self._kernel_def.input_definitions:
            sdim_attr = f"{inp_def.name}_sdim"
            if sdim_attr in [a.name for a in self.onnx_node.attribute]:
                value = self.get_nodeattr(sdim_attr)
                if value is not None:
                    # Handle both scalar and list values
                    if isinstance(value, (list, tuple)) or (isinstance(value, int) and value > 0):
                        sdim_config[inp_def.name] = value
        
        # Fall back to legacy SIMD for all inputs
        if not sdim_config:
            simd = self.get_nodeattr("SIMD")
            if simd and simd > 0:
                for inp_def in self._kernel_def.input_definitions:
                    sdim_config[inp_def.name] = simd
        
        return sdim_config
    
    def _apply_sdim_configuration(self):
        """
        Apply SDIM configuration to KernelModel.
        
        Extracts SDIM values from node attributes and configures
        the kernel model's streaming dimensions.
        """
        sdim_config = self._extract_sdim_configuration()
        if sdim_config:
            self._kernel_model.configure_sdim(sdim_config)
            self._sdim_config = sdim_config
    
    # FINN Abstract Method Implementations
    
    def get_input_datatype(self, ind=0) -> DataType:
        """
        Get FINN DataType of input stream.
        
        Args:
            ind: Input index (default 0)
            
        Returns:
            QONNX DataType for the specified input
            
        Raises:
            IndexError: If index exceeds available inputs
        """
        self._ensure_kernel_model()
        input_models = list(self._kernel_model.input_models)
        
        if ind >= len(input_models):
            raise IndexError(
                f"Input index {ind} exceeds available inputs ({len(input_models)})"
            )
        
        return input_models[ind].datatype
    
    def get_output_datatype(self, ind=0) -> DataType:
        """
        Get FINN DataType of output stream.
        
        Args:
            ind: Output index (default 0)
            
        Returns:
            QONNX DataType for the specified output
            
        Raises:
            IndexError: If index exceeds available outputs
        """
        self._ensure_kernel_model()
        output_models = list(self._kernel_model.output_models)
        
        if ind >= len(output_models):
            raise IndexError(
                f"Output index {ind} exceeds available outputs ({len(output_models)})"
            )
        
        return output_models[ind].datatype
    
    def get_normal_input_shape(self, ind=0) -> List[int]:
        """
        Get normal (tensor) shape of input.
        
        Args:
            ind: Input index (default 0)
            
        Returns:
            List representing tensor dimensions
        """
        self._ensure_kernel_model()
        input_models = list(self._kernel_model.input_models)
        
        if ind >= len(input_models):
            raise IndexError(
                f"Input index {ind} exceeds available inputs ({len(input_models)})"
            )
        
        return list(input_models[ind].tensor_dims)
    
    def get_normal_output_shape(self, ind=0) -> List[int]:
        """
        Get normal (tensor) shape of output.
        
        Args:
            ind: Output index (default 0)
            
        Returns:
            List representing tensor dimensions
        """
        self._ensure_kernel_model()
        output_models = list(self._kernel_model.output_models)
        
        if ind >= len(output_models):
            raise IndexError(
                f"Output index {ind} exceeds available outputs ({len(output_models)})"
            )
        
        return list(output_models[ind].tensor_dims)
    
    def get_folded_input_shape(self, ind=0) -> List[int]:
        """
        Get folded shape for hardware implementation.
        
        Folded shape represents how data is organized for streaming:
        [num_blocks_dim0, num_blocks_dim1, ..., folded_block_dim0, folded_block_dim1, ...]
        
        Where folded_block_dim = block_dim / sdim
        
        Args:
            ind: Input index (default 0)
            
        Returns:
            List representing folded dimensions
        """
        self._ensure_kernel_model()
        input_models = list(self._kernel_model.input_models)
        
        if ind >= len(input_models):
            raise IndexError(
                f"Input index {ind} exceeds available inputs ({len(input_models)})"
            )
        
        iface = input_models[ind]
        
        # Calculate number of blocks in each dimension
        num_blocks = []
        for t, b in zip(iface.tensor_dims, iface.block_dims):
            num_blocks.append(math.ceil(t / b))
        
        # Calculate folded block dimensions (block_dims / sdim)
        folded_block = []
        for bd, sd in zip(iface.block_dims, iface.sdim):
            if bd % sd != 0:
                raise ValueError(
                    f"Block dimension {bd} not divisible by SDIM {sd} "
                    f"for input '{iface.definition.name}'"
                )
            folded_block.append(bd // sd)
        
        # Concatenate: [num_blocks..., folded_block_dims...]
        return num_blocks + folded_block
    
    def get_folded_output_shape(self, ind=0) -> List[int]:
        """
        Get folded shape for hardware output.
        
        Similar to input folding but uses output streaming rates.
        
        Args:
            ind: Output index (default 0)
            
        Returns:
            List representing folded dimensions
        """
        self._ensure_kernel_model()
        output_models = list(self._kernel_model.output_models)
        
        if ind >= len(output_models):
            raise IndexError(
                f"Output index {ind} exceeds available outputs ({len(output_models)})"
            )
        
        iface = output_models[ind]
        
        # Calculate number of blocks in each dimension
        num_blocks = []
        for t, b in zip(iface.tensor_dims, iface.block_dims):
            num_blocks.append(math.ceil(t / b))
        
        # For outputs, use streaming_rate instead of sdim
        # If not set, assume no folding (streaming_rate = block_dims)
        if hasattr(iface, '_streaming_rate') and iface._streaming_rate is not None:
            # Streaming rate is a single value, apply where it makes sense
            sr = iface.streaming_rate
            folded_block = []
            for bd in iface.block_dims:
                if sr > bd:
                    # Streaming rate exceeds block dim - no folding on this dimension
                    folded_block.append(1)
                elif bd % sr != 0:
                    # Try to find a divisor that works
                    folded_block.append(max(1, bd // sr))
                else:
                    folded_block.append(bd // sr)
        else:
            # No streaming rate set - no folding
            folded_block = [1] * len(iface.block_dims)
        
        return num_blocks + folded_block
    
    def get_instream_width(self, ind=0) -> int:
        """
        Get input stream width in bits.
        
        Stream width = datatype_bits * product(sdim)
        
        Args:
            ind: Input index (default 0)
            
        Returns:
            Width in bits
        """
        self._ensure_kernel_model()
        input_models = list(self._kernel_model.input_models)
        
        if ind >= len(input_models):
            raise IndexError(
                f"Input index {ind} exceeds available inputs ({len(input_models)})"
            )
        
        iface = input_models[ind]
        
        # Width = datatype bits * streaming elements
        datatype_bits = iface.datatype.bitwidth()
        streaming_elements = int(np.prod(iface.sdim))
        
        return datatype_bits * streaming_elements
    
    def get_outstream_width(self, ind=0) -> int:
        """
        Get output stream width in bits.
        
        Stream width = datatype_bits * product(streaming_rate)
        
        Args:
            ind: Output index (default 0)
            
        Returns:
            Width in bits
        """
        self._ensure_kernel_model()
        output_models = list(self._kernel_model.output_models)
        
        if ind >= len(output_models):
            raise IndexError(
                f"Output index {ind} exceeds available outputs ({len(output_models)})"
            )
        
        iface = output_models[ind]
        
        # Width = datatype bits * streaming elements
        datatype_bits = iface.datatype.bitwidth()
        
        # Use streaming_rate if available, otherwise assume 1
        if hasattr(iface, 'streaming_rate') and iface.streaming_rate:
            streaming_elements = int(np.prod(iface.streaming_rate))
        else:
            streaming_elements = 1
        
        return datatype_bits * streaming_elements
    
    def get_number_output_values(self) -> int:
        """
        Get total number of output values.
        
        Returns:
            Total elements across all outputs
        """
        self._ensure_kernel_model()
        
        total = 0
        for out_model in self._kernel_model.output_models:
            total += int(np.prod(out_model.tensor_dims))
        
        return total
    
    # Node Attribute Management
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """
        Define node attributes including auto-generated datatype attributes.
        
        If CodegenBinding is present, it will auto-generate algorithm parameters
        and other RTL-specific attributes.
        
        Returns:
            Dictionary of attribute definitions
        """
        # Start with parent class attributes
        attrs = super().get_nodeattr_types()
        
        # Use new KernelModeling architecture for attribute mapping
        finn_attrs = self._get_finn_attribute_mapping()
        for interface_name, attr_name in finn_attrs.items():
            # All datatype attributes are required
            attrs[attr_name] = ("s", True, "")
        
        # Add standard attributes
        attrs["SIMD"] = ("i", False, 1)  # Maps to input SDIM
        attrs["PE"] = ("i", False, 1)    # Maps to weight/output SDIM
        
        # Memory style for operations with weights
        if self._has_weight_inputs():
            attrs["ram_style"] = ("s", False, "auto", {"auto", "block", "distributed", "ultra"})
        
        return attrs
    
    def _has_weight_inputs(self) -> bool:
        """
        Check if kernel has weight inputs.
        
        Returns:
            True if any input has is_weight=True
        """
        return self._kernel_def.has_weights()
    
    def _apply_legacy_attributes(self):
        """
        Apply operation-specific legacy attribute mappings.
        
        This method is called after model creation to allow subclasses
        to interpret legacy FINN attributes (like numInputVectors, PE, SIMD)
        and apply them to the kernel model (e.g., modifying shapes or SDIM).
        
        Default implementation does nothing. Subclasses should override
        to handle their specific legacy attributes.
        """
        pass
    
    def _get_interface_model(self, interface_name: str) -> Union[InputInterface, OutputInterface, None]:
        """
        Get interface model by name.
        
        Args:
            interface_name: Name of the interface (compiler name)
            
        Returns:
            Interface model or None if not found
        """
        self._ensure_kernel_model()
        
        # Check inputs
        for inp in self._kernel_model.input_models:
            if inp.definition.name == interface_name:
                return inp
        
        # Check outputs
        for out in self._kernel_model.output_models:
            if out.definition.name == interface_name:
                return out
        
        return None
    
    def _get_interface_datatype(self, interface_name: str) -> Optional[DataType]:
        """
        Get the datatype of an interface by name.
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            QONNX DataType or None if not found
        """
        interface = self._get_interface_model(interface_name)
        if interface and hasattr(interface, 'datatype'):
            return interface.datatype
        return None
    
    # Resource Estimation
    
    def bram_estimation(self) -> int:
        """
        Estimate BRAM usage from memory requirements.
        
        Returns:
            Number of 18Kb BRAM blocks
        """
        self._ensure_kernel_model()
        
        try:
            metrics = self._kernel_model.calculate_performance_metrics()
            
            # Extract memory requirements
            memory_bits = 0
            if "inputs" in metrics:
                for inp_name, inp_metrics in metrics["inputs"].items():
                    if "memory_bits" in inp_metrics:
                        memory_bits += inp_metrics["memory_bits"]
            
            # Check aggregate metrics
            if "aggregate" in metrics and "memory_bits" in metrics["aggregate"]:
                memory_bits = max(memory_bits, metrics["aggregate"]["memory_bits"])
            
            # Convert to BRAM blocks (18Kb each)
            if memory_bits > 0:
                bram_bits = 18 * 1024
                return (memory_bits + bram_bits - 1) // bram_bits
        
        except Exception:
            # If metrics calculation fails, return conservative estimate
            pass
        
        return 0
    
    def lut_estimation(self) -> int:
        """
        Estimate LUT usage based on stream widths.
        
        Returns:
            Estimated LUT count
        """
        self._ensure_kernel_model()
        
        # Simple heuristic: LUTs proportional to total stream width
        total_width = 0
        
        # Sum input stream widths
        for i in range(len(list(self._kernel_model.input_models))):
            try:
                total_width += self.get_instream_width(i)
            except:
                pass
        
        # Sum output stream widths
        for i in range(len(list(self._kernel_model.output_models))):
            try:
                total_width += self.get_outstream_width(i)
            except:
                pass
        
        # Rough estimate: 10 LUTs per bit of stream width
        return max(100, total_width * 10)
    
    def dsp_estimation(self, fpgapart: str = "xczu7ev") -> int:
        """
        Estimate DSP usage.
        
        Args:
            fpgapart: FPGA part name
            
        Returns:
            Estimated DSP count
        """
        # Simple default - subclasses should override for accuracy
        return 0
    
    def uram_estimation(self) -> int:
        """
        Estimate UltraRAM usage.
        
        Returns:
            Number of URAM blocks
        """
        # Check if using ultra RAM
        if self.get_nodeattr("ram_style") == "ultra":
            # Similar to BRAM but with 288Kb blocks
            bram_count = self.bram_estimation()
            if bram_count > 0:
                # Rough conversion from BRAM to URAM
                bram_bits = bram_count * 18 * 1024
                uram_bits = 288 * 1024
                return (bram_bits + uram_bits - 1) // uram_bits
        
        return 0
    
    def get_exp_cycles(self) -> int:
        """
        Get expected execution cycles.
        
        Returns:
            Expected cycles for one inference
        """
        self._ensure_kernel_model()
        
        try:
            metrics = self._kernel_model.calculate_performance_metrics()
            
            # Look for initiation interval in aggregate metrics
            if "aggregate" in metrics:
                if "initiation_interval" in metrics["aggregate"]:
                    return int(metrics["aggregate"]["initiation_interval"])
                elif "latency" in metrics["aggregate"]:
                    return int(metrics["aggregate"]["latency"])
        
        except Exception:
            # If metrics fail, return conservative estimate
            pass
        
        # Default: assume one cycle per output element
        return self.get_number_output_values()
    
    # Optional Method Implementations
    
    def verify_node(self) -> List[str]:
        """
        Verify node configuration.
        
        Returns:
            List of verification messages
        """
        info_messages = []
        
        # Check backend
        backend = self.get_nodeattr("backend")
        if backend == "fpgadataflow":
            info_messages.append("✓ Backend set correctly to 'fpgadataflow'")
        else:
            info_messages.append(f"✗ Backend '{backend}' should be 'fpgadataflow'")
        
        # Check required datatypes
        self._ensure_kernel_model()
        
        for i, inp_model in enumerate(self._kernel_model.input_models):
            dtype = inp_model.datatype
            if dtype:
                info_messages.append(f"✓ Input {i} has datatype {dtype}")
            else:
                info_messages.append(f"✗ Input {i} missing datatype")
        
        for i, out_model in enumerate(self._kernel_model.output_models):
            dtype = out_model.datatype
            if dtype:
                info_messages.append(f"✓ Output {i} has datatype {dtype}")
            else:
                info_messages.append(f"✗ Output {i} missing datatype")
        
        # Check SDIM configuration
        if self._sdim_config:
            info_messages.append(f"✓ SDIM configured: {self._sdim_config}")
        else:
            info_messages.append("ℹ No SDIM configuration (using defaults)")
        
        return info_messages
    
    def execute_node(self, context, graph):
        """
        Execute node in simulation.
        
        This is a basic pass-through implementation. Subclasses should
        override for actual computation.
        
        Args:
            context: Execution context with tensors
            graph: ONNX graph
        """
        # Simple pass-through: copy first input to first output
        node = self.onnx_node
        
        if len(node.input) > 0 and len(node.output) > 0:
            if node.input[0] in context:
                # Basic copy - subclasses should implement actual logic
                context[node.output[0]] = context[node.input[0]].copy()
    
    def infer_node_datatype(self, model, node):
        """
        Infer node datatypes during graph transformation.
        
        This is required by FINN's HWCustomOp abstract interface.
        For AutoHWCustomOp, datatypes are already specified via
        node attributes, so this is a no-op.
        
        Args:
            model: ONNX model
            node: ONNX node
        """
        # Datatypes are already specified via attributes
        pass