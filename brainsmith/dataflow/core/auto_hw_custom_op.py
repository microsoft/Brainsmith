"""
AutoHWCustomOp base class for auto-generated hardware custom operators.

This module provides the base class for all auto-generated HWCustomOp classes,
implementing standardized methods that can be fully determined from the
dataflow interface metadata.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from .class_naming import generate_class_name
from .dataflow_model import DataflowModel, ParallelismConfiguration
from .dataflow_interface import DataflowInterface, DataflowInterfaceType, DataflowDataType
from .interface_metadata import InterfaceMetadata, InterfaceMetadataCollection, DataTypeConstraint

# Try to import FINN, but make it optional for development
try:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
    from finn.util.data_packing import numpy_to_hls_code
    from qonnx.core.datatype import DataType
    FINN_AVAILABLE = True
except ImportError:
    FINN_AVAILABLE = False
    # Create minimal stub base class for standalone operation
    class HWCustomOp:
        def __init__(self, onnx_node, **kwargs):
            self.onnx_node = onnx_node
            
        def get_nodeattr_types(self):
            return {}
            
        def get_nodeattr(self, name):
            return None
            
    class DataType:
        """Stub DataType for development without FINN"""
        @staticmethod
        def bitwidth():
            return 8
            
    def numpy_to_hls_code(*args, **kwargs):
        pass


class AutoHWCustomOp(HWCustomOp):
    """
    Base class for auto-generated HWCustomOp implementations.
    
    This class provides all standardized method implementations that can be
    fully determined from dataflow interface metadata, dramatically reducing
    the amount of generated code needed in templates.
    
    Key features:
    - Interface metadata driven: purely based on InterfaceMetadata objects
    - Lazy DataflowModel building: compatible with FINN's node creation → parallelism setting workflow
    - Automatic tensor shape extraction: eliminates manual configuration burden
    """
    
    def __init__(self, onnx_node, interface_metadata: List[InterfaceMetadata], **kwargs):
        """
        Initialize AutoHWCustomOp with interface metadata.
        
        Args:
            onnx_node: ONNX node for this operation
            interface_metadata: List of interface metadata defining the operation
            **kwargs: Additional arguments
        """
        super().__init__(onnx_node, **kwargs)
        
        # Validate interface metadata
        if not interface_metadata:
            raise ValueError("No interface metadata available for building DataflowModel")
        
        # Interface-driven initialization
        self._interface_metadata_collection = InterfaceMetadataCollection(interface_metadata)
        self._dataflow_model = None
        self._model_built = False
        self._model_wrapper = None
    
    def _ensure_dataflow_model_built(self):
        """Build DataflowModel on first access if not already built."""
        if not self._model_built:
            self._build_dataflow_model()
            self._model_built = True
    
    def _build_dataflow_model(self):
        """Build DataflowModel from interface metadata with runtime shape extraction."""
        # Import here to avoid circular imports
        from .tensor_chunking import TensorChunking
        
        # Initialize chunker for tensor shape extraction
        chunker = TensorChunking()
        
        # Pass ModelWrapper to chunker for shape extraction
        if self._model_wrapper:
            chunker.set_model_wrapper(self._model_wrapper)
        
        interfaces = []
        
        for metadata in self._interface_metadata_collection.interfaces:
            # Extract runtime tensor shape from ModelWrapper if available
            try:
                if self._model_wrapper:
                    tensor_shape = self._extract_runtime_tensor_shape(metadata.name)
                else:
                    # Use chunker for fallback shape extraction
                    tensor_shape = chunker.extract_tensor_shape_from_input(metadata.name, self.onnx_node)
            except Exception:
                # Final fallback to default shape from metadata
                tensor_shape = self._get_default_shape_for_interface(metadata.name)
            
            # Use the interface's own chunking strategy with runtime shape
            num_tensors, tDim = self._compute_runtime_chunking(metadata, tensor_shape)
            
            # Infer layout for better defaults
            layout = self._infer_layout_from_shape(tensor_shape)
            
            # Convert InterfaceMetadata datatype to DataflowDataType
            dtype_constraint = metadata.get_default_datatype()
            dataflow_dtype = self._convert_metadata_datatype(dtype_constraint)
            
            # Create DataflowInterface with runtime-extracted information
            interface = DataflowInterface(
                name=metadata.name,
                interface_type=metadata.interface_type,
                num_tensors=num_tensors,
                tDim=tDim,
                sDim=tDim.copy(),  # Initialize sDim same as tDim
                dtype=dataflow_dtype
            )
            
            # Store runtime metadata for debugging/introspection
            interface._runtime_tensor_shape = tensor_shape
            interface._inferred_layout = layout
            interface._chunking_source = "runtime_extraction" if self._model_wrapper else "fallback"
            
            interfaces.append(interface)
        
        # Create DataflowModel
        self._dataflow_model = DataflowModel(interfaces, {})
    
    def _invalidate_dataflow_model(self):
        """Mark model as needing rebuild after attribute changes."""
        self._model_built = False
        self._dataflow_model = None
    
    def _extract_runtime_tensor_shape(self, interface_name: str) -> List[int]:
        """
        Extract actual tensor shape from ModelWrapper at runtime.
        
        This is the key method that addresses the static vs runtime configuration issue.
        Instead of using hardcoded static dimensions, we extract actual tensor shapes
        from the FINN ModelWrapper when the HWCustomOp is instantiated at runtime.
        
        Args:
            interface_name: Name of the interface to extract shape for
            
        Returns:
            List[int]: Actual tensor shape from runtime model
        """
        if not self._model_wrapper:
            raise RuntimeError("ModelWrapper not available for runtime shape extraction")
        
        try:
            # For input interfaces, extract shape from model graph
            if interface_name in [iface.name for iface in self._interface_metadata_collection.get_input_interfaces()]:
                return self._extract_input_shape_from_model(interface_name)
            
            # For output interfaces, extract shape from model graph
            elif interface_name in [iface.name for iface in self._interface_metadata_collection.get_output_interfaces()]:
                return self._extract_output_shape_from_model(interface_name)
            
            # For weight interfaces, extract shape from node attributes or initializers
            elif interface_name in [iface.name for iface in self._interface_metadata_collection.get_weight_interfaces()]:
                return self._extract_weight_shape_from_model(interface_name)
            
            else:
                # Fallback to default shape
                return self._get_default_shape_for_interface(interface_name)
                
        except Exception as e:
            # Log the error but continue with fallback
            print(f"Warning: Failed to extract runtime shape for {interface_name}: {e}")
            return self._get_default_shape_for_interface(interface_name)
    
    def _extract_input_shape_from_model(self, interface_name: str) -> List[int]:
        """Extract input tensor shape from FINN ModelWrapper."""
        try:
            # Get input shape from model graph
            graph_input = None
            for inp in self._model_wrapper.graph.input:
                if inp.name == interface_name or interface_name in inp.name:
                    graph_input = inp
                    break
            
            if graph_input and graph_input.type.tensor_type.shape:
                shape = []
                for dim in graph_input.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(int(dim.dim_value))
                    else:
                        # Handle dynamic dimensions with reasonable defaults
                        shape.append(1)
                return shape
            
            # Fallback: try to get shape from node inputs
            if hasattr(self.onnx_node, 'input') and self.onnx_node.input:
                input_name = self.onnx_node.input[0]  # First input
                value_info = self._model_wrapper.get_tensor_shape(input_name)
                if value_info:
                    return list(value_info)
            
            # No fallback - must have valid dimensions
            raise RuntimeError(f"Cannot extract input shape for {interface_name}: no valid tensor information found")
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract input shape for {interface_name}: {e}")
    
    def _extract_output_shape_from_model(self, interface_name: str) -> List[int]:
        """Extract output tensor shape from FINN ModelWrapper."""
        try:
            # Get output shape from model graph
            graph_output = None
            for out in self._model_wrapper.graph.output:
                if out.name == interface_name or interface_name in out.name:
                    graph_output = out
                    break
            
            if graph_output and graph_output.type.tensor_type.shape:
                shape = []
                for dim in graph_output.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(int(dim.dim_value))
                    else:
                        # Handle dynamic dimensions with reasonable defaults
                        shape.append(1)
                return shape
            
            # Fallback: try to get shape from node outputs
            if hasattr(self.onnx_node, 'output') and self.onnx_node.output:
                output_name = self.onnx_node.output[0]  # First output
                value_info = self._model_wrapper.get_tensor_shape(output_name)
                if value_info:
                    return list(value_info)
            
            # No fallback - must have valid dimensions
            raise RuntimeError(f"Cannot extract output shape for {interface_name}: no valid tensor information found")
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract output shape for {interface_name}: {e}")
    
    def _extract_weight_shape_from_model(self, interface_name: str) -> List[int]:
        """Extract weight tensor shape from FINN ModelWrapper."""
        try:
            # Look for initializers in the model
            for init in self._model_wrapper.graph.initializer:
                if init.name == interface_name or interface_name in init.name:
                    return list(init.dims)
            
            # Look for node attributes that might contain weight shapes
            if hasattr(self.onnx_node, 'attribute'):
                for attr in self.onnx_node.attribute:
                    if 'weight' in attr.name.lower() or 'param' in attr.name.lower():
                        if hasattr(attr, 'ints') and attr.ints:
                            return list(attr.ints)
            
            # No fallback - must have valid dimensions
            raise RuntimeError(f"Cannot extract weight shape for {interface_name}: no valid tensor information found")
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract weight shape for {interface_name}: {e}")
    
    def _compute_runtime_chunking(self, metadata, tensor_shape: List[int]) -> Tuple[List[int], List[int]]:
        """
        Compute chunking based on runtime tensor shape and interface chunking strategy.
        
        Args:
            metadata: InterfaceMetadata with chunking strategy
            tensor_shape: Actual runtime tensor shape
            
        Returns:
            Tuple of (num_tensors, tDim)
        """
        # Use the interface's chunking strategy if available
        if hasattr(metadata, 'chunking_strategy') and metadata.chunking_strategy:
            return metadata.chunking_strategy.compute_chunking(tensor_shape, metadata.name)
        
        # Fallback: minimal chunking (no chunking)
        num_tensors = [1] * len(tensor_shape)
        tDim = list(tensor_shape)
        return num_tensors, tDim
    
    def _get_default_shape_for_interface(self, interface_name: str) -> List[int]:
        """
        Get default tensor shape when runtime extraction fails.
        
        WARNING: This should only be used during development/testing.
        In production, the FINN compiler must provide a valid ModelWrapper
        for proper runtime dimension extraction.
        """
        raise RuntimeError(
            f"Cannot determine tensor shape for interface '{interface_name}': "
            f"No ModelWrapper available for runtime shape extraction. "
            f"The HWCustomOp must be instantiated by the FINN compiler with "
            f"a valid ModelWrapper containing actual tensor shapes."
        )
    
    def _infer_layout_from_shape(self, tensor_shape: List[int]) -> str:
        """Infer tensor layout from shape."""
        if len(tensor_shape) == 4:
            return "NCHW"
        elif len(tensor_shape) == 3:
            return "CHW"
        elif len(tensor_shape) == 2:
            return "NC"
        elif len(tensor_shape) == 1:
            return "C"
        else:
            return "UNKNOWN"
    
    def _convert_metadata_datatype(self, dtype_constraint) -> DataflowDataType:
        """Convert InterfaceMetadata datatype constraint to DataflowDataType."""
        return DataflowDataType(
            base_type=dtype_constraint.finn_type.replace('8', '').replace('16', '').replace('32', ''),
            bitwidth=dtype_constraint.bit_width,
            signed=dtype_constraint.signed,
            finn_type=dtype_constraint.finn_type
        )
    
    @property
    def dataflow_model(self) -> DataflowModel:
        """Get DataflowModel, building it lazily if needed."""
        self._ensure_dataflow_model_built()
        return self._dataflow_model
    
    @property
    def interface_metadata(self) -> InterfaceMetadataCollection:
        """Get interface metadata collection."""
        return self._interface_metadata_collection
    
    def set_model_wrapper(self, model_wrapper):
        """Set ModelWrapper for accurate tensor shape extraction."""
        self._model_wrapper = model_wrapper
        self._invalidate_dataflow_model()  # Rebuild with new shape information
    
    def get_model_wrapper(self):
        """Get current ModelWrapper."""
        return self._model_wrapper

    @property
    def input_interfaces(self) -> List[str]:
        """Get input interface names from DataflowModel."""
        return [iface.name for iface in self.dataflow_model.input_interfaces]

    @property
    def output_interfaces(self) -> List[str]:
        """Get output interface names from DataflowModel."""
        return [iface.name for iface in self.dataflow_model.output_interfaces]

    @property
    def weight_interfaces(self) -> List[str]:
        """Get weight interface names from DataflowModel."""
        return [iface.name for iface in self.dataflow_model.weight_interfaces]

    @property
    def config_interfaces(self) -> List[str]:
        """Get config interface names from DataflowModel."""
        return [iface.name for iface in self.dataflow_model.config_interfaces]
    
    def get_enhanced_nodeattr_types(self) -> Dict[str, Tuple[str, bool, Any]]:
        """
        Get enhanced node attribute types with dataflow modeling support.
        
        This method adds dataflow-specific attributes to the base FINN attributes.
        Subclasses should override get_nodeattr_types() and call this method.
        """
        attrs = {}
        
        # Add parallelism configuration for input and output interfaces
        for iface in self.dataflow_model.input_interfaces:
            attrs[f"{iface.name}_parallel"] = ("i", False, 1)
            
        for iface in self.dataflow_model.output_interfaces:
            attrs[f"{iface.name}_parallel"] = ("i", False, 1)
                    
        # Add datatype configuration for all interfaces
        for iface in self.dataflow_model.interfaces.values():
            default_dtype = iface.dtype.finn_type
            attrs[f"{iface.name}_dtype"] = ("s", False, default_dtype)
            
        # Add resource estimation and validation configuration
        attrs.update({
            "resource_estimation_mode": ("s", False, "automatic", {"automatic", "conservative", "optimistic"}),
            "enable_constraint_validation": ("b", False, True),
        })
        
        return attrs
        
    def get_input_datatype(self, ind: int = 0) -> Any:
        """Get input datatype from DataflowModel interface."""
        input_ifaces = self.dataflow_model.input_interfaces
        if ind >= len(input_ifaces):
            raise IndexError(f"Input index {ind} exceeds available inputs")
        
        interface = input_ifaces[ind]
        
        # Check for runtime configuration override
        configured_dtype = self.get_nodeattr(f"{interface.name}_dtype")
        if configured_dtype:
            if not interface.validate_datatype_string(configured_dtype):
                raise ValueError(f"Configured datatype {configured_dtype} violates constraints")
            return DataType[configured_dtype] if FINN_AVAILABLE else configured_dtype
        
        # Use interface's default datatype
        return DataType[interface.dtype.finn_type] if FINN_AVAILABLE else interface.dtype.finn_type
            
    def get_output_datatype(self, ind: int = 0) -> Any:
        """Get output datatype from DataflowModel interface."""
        output_ifaces = self.dataflow_model.output_interfaces
        if ind >= len(output_ifaces):
            raise IndexError(f"Output index {ind} exceeds available outputs")
        
        interface = output_ifaces[ind]
        
        # Check for runtime configuration override
        configured_dtype = self.get_nodeattr(f"{interface.name}_dtype")
        if configured_dtype:
            if not interface.validate_datatype_string(configured_dtype):
                raise ValueError(f"Configured datatype {configured_dtype} violates constraints")
            return DataType[configured_dtype] if FINN_AVAILABLE else configured_dtype
        
        # Use interface's default datatype
        return DataType[interface.dtype.finn_type] if FINN_AVAILABLE else interface.dtype.finn_type
            
    def get_normal_input_shape(self, ind: int = 0) -> List[int]:
        """Get normal input shape from DataflowModel interface."""
        input_ifaces = self.dataflow_model.input_interfaces
        if ind >= len(input_ifaces):
            raise IndexError(f"Input index {ind} exceeds available inputs")
        
        interface = input_ifaces[ind]
        return interface.reconstruct_tensor_shape()
        
    def get_normal_output_shape(self, ind: int = 0) -> List[int]:
        """Get normal output shape from DataflowModel interface."""
        output_ifaces = self.dataflow_model.output_interfaces
        if ind >= len(output_ifaces):
            raise IndexError(f"Output index {ind} exceeds available outputs")
        
        interface = output_ifaces[ind]
        return interface.reconstruct_tensor_shape()
        
    def get_folded_input_shape(self, ind: int = 0) -> List[int]:
        """Get folded input shape considering parallelism configuration."""
        input_ifaces = self.dataflow_model.input_interfaces
        if ind >= len(input_ifaces):
            raise IndexError(f"Input index {ind} exceeds available inputs")
        
        interface = input_ifaces[ind]
        normal_shape = interface.reconstruct_tensor_shape()
        
        # Apply folding based on parallelism configuration
        parallel_factor = self.get_nodeattr(f"{interface.name}_parallel") or 1
        if parallel_factor > 1 and normal_shape:
            folded_shape = normal_shape.copy()
            folded_shape[-1] = folded_shape[-1] // parallel_factor
            return folded_shape
        
        return normal_shape
        
    def get_folded_output_shape(self, ind: int = 0) -> List[int]:
        """Get folded output shape considering parallelism configuration."""
        output_ifaces = self.dataflow_model.output_interfaces
        if ind >= len(output_ifaces):
            raise IndexError(f"Output index {ind} exceeds available outputs")
        
        interface = output_ifaces[ind]
        normal_shape = interface.reconstruct_tensor_shape()
        
        # Apply folding based on parallelism configuration
        parallel_factor = self.get_nodeattr(f"{interface.name}_parallel") or 1
        if parallel_factor > 1 and normal_shape:
            folded_shape = normal_shape.copy()
            folded_shape[-1] = folded_shape[-1] // parallel_factor
            return folded_shape
        
        return normal_shape
        
    def get_instream_width(self, ind: int = 0) -> int:
        """Get input stream width in bits."""
        input_dtype = self.get_input_datatype(ind)
        folded_shape = self.get_folded_input_shape(ind)
        
        # Width = datatype bits * folded elements in last dimension
        elements_per_cycle = folded_shape[-1] if folded_shape else 1
        
        if FINN_AVAILABLE:
            return input_dtype.bitwidth() * elements_per_cycle
        else:
            # Default to 8 bits for development
            return 8 * elements_per_cycle
            
    def get_outstream_width(self, ind: int = 0) -> int:
        """Get output stream width in bits."""
        output_dtype = self.get_output_datatype(ind)
        folded_shape = self.get_folded_output_shape(ind)
        
        # Width = datatype bits * folded elements in last dimension
        elements_per_cycle = folded_shape[-1] if folded_shape else 1
        
        if FINN_AVAILABLE:
            return output_dtype.bitwidth() * elements_per_cycle
        else:
            # Default to 8 bits for development
            return 8 * elements_per_cycle
            
    def get_number_output_values(self) -> int:
        """Get total number of output values."""
        if not self.output_interfaces:
            return 0
            
        total_outputs = 0
        for ind in range(len(self.output_interfaces)):
            output_shape = self.get_normal_output_shape(ind)
            total_outputs += np.prod(output_shape)
        return int(total_outputs)
        
    def get_exp_cycles(self) -> int:
        """Get expected cycles using DataflowModel's unified computational model."""
        # Extract current parallelism configuration
        iPar = {}
        wPar = {}
        
        for iface in self.dataflow_model.input_interfaces:
            iPar[iface.name] = self.get_nodeattr(f"{iface.name}_parallel") or 1
        
        for iface in self.dataflow_model.weight_interfaces:
            wPar[iface.name] = self.get_nodeattr(f"{iface.name}_parallel") or 1
        
        # Use unified calculation
        intervals = self.dataflow_model.calculate_initiation_intervals(iPar, wPar)
        return intervals.L
        
    def get_op_and_param_counts(self) -> Dict[str, int]:
        """
        Get operation and parameter counts.
        
        Returns:
            Dict with operation and parameter counts
        """
        counts = {
            "ops": 0,
            "params": 0,
            "weight_params": 0,
            "config_params": 0
        }
        
        # Count weight parameters
        for iface in self.dataflow_model.weight_interfaces:
            params = np.prod(iface.num_tensors) * np.prod(iface.tDim)
            counts["weight_params"] += params
            counts["params"] += params
            
        # Count config parameters
        for iface in self.dataflow_model.config_interfaces:
            params = np.prod(iface.num_tensors) * np.prod(iface.tDim)
            counts["config_params"] += params
            counts["params"] += params
            
        # Estimate operations based on input/output sizes
        if self.dataflow_model.input_interfaces and self.dataflow_model.output_interfaces:
            input_size = sum(
                np.prod(iface.num_tensors) * np.prod(iface.tDim)
                for iface in self.dataflow_model.input_interfaces
            )
            output_size = sum(
                np.prod(iface.num_tensors) * np.prod(iface.tDim)
                for iface in self.dataflow_model.output_interfaces
            )
            # Simple estimation: operations proportional to input*output
            counts["ops"] = input_size * output_size
            
        return counts
        
    def derive_characteristic_fxns(self) -> Dict[str, Any]:
        """
        Derive characteristic functions for the operation.
        
        Returns:
            Dict with characteristic functions
        """
        return {
            "compute_cycles": self.get_exp_cycles(),
            "bram_usage": self.estimate_bram_usage(),
            "lut_usage": self.estimate_lut_usage(),
            "dsp_usage": self.estimate_dsp_usage("xczu7ev"),  # Default FPGA part
            "stream_widths": {
                "input": [self.get_instream_width(i) for i in range(len(self.input_interfaces))],
                "output": [self.get_outstream_width(i) for i in range(len(self.output_interfaces))]
            }
        }
        
    def generate_params(self, model, path):
        """
        Generate parameter files for weight interfaces.
        
        Args:
            model: ONNX model containing weights
            path: Output path for parameter files
        """
        if not self.dataflow_model.weight_interfaces:
            # No weight interfaces - nothing to generate
            return
            
        import os
        os.makedirs(path, exist_ok=True)
        
        for iface in self.dataflow_model.weight_interfaces:
            # Generate parameters for each weight interface
            shape = (
                np.prod(iface.num_tensors),
                np.prod(iface.tDim)
            )
            
            # Extract weights from model (placeholder - depends on actual weight source)
            # This would need to be customized based on how weights are stored
            weights = np.random.randn(*shape).astype(np.float32)
            
            # Convert to appropriate format and save
            dtype_name = self.get_nodeattr(f"{iface.name}_dtype") or iface.dtype.finn_type
            weight_file = os.path.join(path, f"{iface.name}_weights.dat")
            
            if FINN_AVAILABLE:
                # Use FINN's data packing utilities
                with open(weight_file, "w") as f:
                    numpy_to_hls_code(
                        weights,
                        DataType[dtype_name],
                        f,
                        dtype_name
                    )
            else:
                # Simple text output for development
                with open(weight_file, "w") as f:
                    for w in weights.flatten():
                        f.write(f"{w}\n")

    def _get_current_parallelism_config(self) -> Dict[str, int]:
        """Get current parallelism configuration for all interfaces."""
        config = {}
        # Get all interfaces from DataflowModel properties
        all_interfaces = (
            self.dataflow_model.input_interfaces +
            self.dataflow_model.output_interfaces +
            self.dataflow_model.weight_interfaces +
            self.dataflow_model.config_interfaces
        )
        
        for iface in all_interfaces:
            parallel_attr = f"{iface.name}_parallel"
            config[iface.name] = self.get_nodeattr(parallel_attr) or 1
        return config
                        
    def estimate_bram_usage(self) -> int:
        """Estimate BRAM usage using DataflowModel resource requirements."""
        try:
            # Create proper ParallelismConfiguration object
            iPar = {iface.name: self.get_nodeattr(f"{iface.name}_parallel") or 1
                   for iface in self.dataflow_model.input_interfaces}
            wPar = {iface.name: self.get_nodeattr(f"{iface.name}_parallel") or 1
                   for iface in self.dataflow_model.weight_interfaces}
            
            parallelism_config = ParallelismConfiguration(
                iPar=iPar,
                wPar=wPar,
                derived_sDim={}  # Will be computed
            )
            
            resources = self.dataflow_model.get_resource_requirements(parallelism_config)
            memory_bits = resources["memory_bits"]
            bram_capacity = 18 * 1024  # BRAM18K
            
            # Apply estimation mode scaling
            estimation_mode = self.get_nodeattr("resource_estimation_mode")
            scale_factor = {"conservative": 1.5, "optimistic": 0.7, "automatic": 1.0}.get(estimation_mode, 1.0)
            
            return int(np.ceil((memory_bits * scale_factor) / bram_capacity))
        except Exception:
            # Fallback to simple estimation
            total_memory = sum(iface.get_memory_footprint() for iface in
                              self.dataflow_model.input_interfaces + self.dataflow_model.weight_interfaces)
            return max(1, total_memory // (18 * 1024))
        
    def estimate_lut_usage(self) -> int:
        """Estimate LUT usage using DataflowModel resource requirements."""
        try:
            # Simple estimation based on interface complexity
            total_width = sum(iface.calculate_stream_width() for iface in
                             self.dataflow_model.input_interfaces + self.dataflow_model.output_interfaces)
            
            # Apply estimation mode scaling
            estimation_mode = self.get_nodeattr("resource_estimation_mode")
            scale_factor = {"conservative": 1.5, "optimistic": 0.7, "automatic": 1.0}.get(estimation_mode, 1.0)
            
            # Simple heuristic: ~10 LUTs per bit of stream width
            lut_estimate = int((total_width * 10) * scale_factor)
            return max(1, lut_estimate)
        except Exception:
            # Fallback to fixed estimate
            return 100
        
    def estimate_dsp_usage(self, fpgapart: str = "xczu7ev") -> int:
        """Estimate DSP usage using DataflowModel resource requirements."""
        try:
            # Simple estimation: assume some operations require DSPs
            total_ops = sum(np.prod(iface.num_tensors) for iface in self.dataflow_model.input_interfaces)
            
            # Apply estimation mode scaling
            estimation_mode = self.get_nodeattr("resource_estimation_mode")
            scale_factor = {"conservative": 1.2, "optimistic": 0.8, "automatic": 1.0}.get(estimation_mode, 1.0)
            
            # Simple heuristic: 1 DSP per 1000 operations
            dsp_estimate = int(np.ceil((total_ops / 1000) * scale_factor))
            return max(0, dsp_estimate)
        except Exception:
            # Fallback to minimal DSP usage
            return 0
        
    def get_interface_config(self, interface_name: str) -> Dict[str, Any]:
        """Get configuration for a specific interface using DataflowModel."""
        # Find interface by name
        all_interfaces = (
            self.dataflow_model.input_interfaces +
            self.dataflow_model.output_interfaces +
            self.dataflow_model.weight_interfaces +
            self.dataflow_model.config_interfaces
        )
        
        iface = None
        for interface in all_interfaces:
            if interface.name == interface_name:
                iface = interface
                break
        
        if not iface:
            raise KeyError(f"Interface '{interface_name}' not found in dataflow model")
            
        config = {
            "interface_type": iface.interface_type.value,  # Use .value for enum
            "dtype": {
                "finn_type": iface.dtype.finn_type,
                "signed": iface.dtype.signed
            },
            "num_tensors": list(iface.num_tensors),  # Correct attribute name
            "tDim": list(iface.tDim),  # Correct attribute name
            "parallel": self.get_nodeattr(f"{interface_name}_parallel") or 1,
            "runtime_dtype": self.get_nodeattr(f"{interface_name}_dtype") or iface.dtype.finn_type
        }
        
        return config
        
    @staticmethod
    def generate_class_name(kernel_name: str) -> str:
        """
        Generate proper CamelCase class name from kernel name.
        
        Args:
            kernel_name: Underscore-separated kernel name
            
        Returns:
            CamelCase class name
        """
        return generate_class_name(kernel_name)
        
    def _validate_datatype_constraints(self, interface_name: str, datatype: str) -> bool:
        """
        Validate datatype against interface constraints.
        
        Args:
            interface_name: Name of the interface
            datatype: Proposed datatype string
            
        Returns:
            bool: True if datatype is valid for the interface
        """
        if not self.get_nodeattr("enable_constraint_validation"):
            return True
            
        interface_config = self.dataflow_interfaces.get(interface_name)
        if not interface_config:
            return True
            
        dtype_config = interface_config["dtype"]
        
        # Check if datatype is in allowed base types
        if dtype_config.get("base_types"):
            if FINN_AVAILABLE:
                dt = DataType[datatype]
                base_type = dt.name.split("_")[0] if "_" in dt.name else dt.name
            else:
                # Simple parsing for development
                base_type = datatype.split("_")[0] if "_" in datatype else datatype
            if base_type not in dtype_config["base_types"]:
                return False
                
        # Check bitwidth constraints
        if FINN_AVAILABLE:
            dt_bitwidth = DataType[datatype].bitwidth()
        else:
            # Extract bitwidth from datatype name (e.g., UINT8 -> 8)
            import re
            match = re.search(r'\d+', datatype)
            dt_bitwidth = int(match.group()) if match else 8
            
        min_bits = dtype_config.get("min_bits")
        max_bits = dtype_config.get("max_bits")
        
        if min_bits is not None and dt_bitwidth < min_bits:
            return False
        if max_bits is not None and dt_bitwidth > max_bits:
            return False
            
        return True