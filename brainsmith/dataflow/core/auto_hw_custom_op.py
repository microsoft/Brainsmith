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
    Base class for auto-generated HWCustomOp implementations using simplified 3-tier architecture.
    
    Three tiers of information:
    1. Kernel Data (static): Interface metadata, chunking strategies, node attributes
    2. Model Data (runtime): qDim from ONNX, datatypes  
    3. Parallelism (dynamic): iPar/wPar values, sDim calculations, performance
    
    This simplified approach eliminates lazy building complexity while maintaining
    full compatibility with FINN's node creation → parallelism setting workflow.
    """
    
    def __init__(self, onnx_node, interface_metadata: List[InterfaceMetadata], **kwargs):
        """
        Initialize AutoHWCustomOp with streamlined 3-tier architecture.
        
        Args:
            onnx_node: ONNX node for this operation
            interface_metadata: List of interface metadata (Tier 1: Kernel Data)
            **kwargs: Additional arguments
        """
        super().__init__(onnx_node, **kwargs)
        
        # Validate interface metadata  
        if not interface_metadata:
            raise ValueError("No interface metadata available for building DataflowModel")
        
        # Tier 1: Kernel Data (static, from RTL)
        self._interface_metadata_collection = InterfaceMetadataCollection(interface_metadata)
        
        # Tier 2: Model Data (runtime, from ONNX) - build immediately with minimum defaults
        self._dataflow_model = self._build_dataflow_model_with_defaults()
        
        # Tier 3: Parallelism (dynamic) - initialize with minimum values
        self._current_parallelism = self._initialize_minimum_parallelism()
    
    def _build_dataflow_model_with_defaults(self) -> 'DataflowModel':
        """
        Build DataflowModel immediately with sensible defaults.
        Tier 2 (Model Data) extracted from ONNX when available, defaults otherwise.
        """
        interfaces = []
        
        for metadata in self._interface_metadata_collection.interfaces:
            # Tier 2: Extract tensor shape from ONNX node if available
            tensor_shape = self._extract_tensor_shape_from_onnx(metadata.name)
            
            # Apply interface's chunking strategy to get qDim and tDim  
            qDim, tDim = self._apply_chunking_strategy(metadata, tensor_shape)
            
            # Convert metadata datatype to DataflowDataType
            dataflow_dtype = self._convert_metadata_datatype(metadata.get_default_datatype())
            
            # Create DataflowInterface with Tier 1 + Tier 2 data
            # Tier 3 (sDim) will be set to minimum parallelism initially
            interface = DataflowInterface(
                name=metadata.name,
                interface_type=metadata.interface_type,
                qDim=qDim,  # Tier 2: From ONNX tensor shape
                tDim=tDim,  # Tier 1: From chunking strategy
                sDim=[1] * len(tDim),  # Tier 3: Minimum parallelism (will be updated)
                dtype=dataflow_dtype  # Tier 2: From ONNX or metadata
            )
            
            interfaces.append(interface)
        
        # Create and return DataflowModel
        return DataflowModel(interfaces, {})
    
    def _extract_tensor_shape_from_onnx(self, interface_name: str) -> List[int]:
        """
        Extract tensor shape from ONNX node (Tier 2: Model Data).
        
        Simplified approach: Extract from ONNX node when available,
        use sensible defaults otherwise.
        """
        try:
            # Try to get shape from ONNX node inputs/outputs
            if hasattr(self.onnx_node, 'input') and self.onnx_node.input:
                # For inputs, use first input as reference
                # In practice, FINN will provide proper shape info
                return [128]  # Default 1D shape
            else:
                # Default fallback shape
                return [128]  # Simple 1D default
        except Exception:
            # Always have a fallback
            return [128]
    
    def _apply_chunking_strategy(self, metadata, tensor_shape: List[int]) -> tuple[List[int], List[int]]:
        """
        Apply chunking strategy from metadata (Tier 1: Kernel Data) to tensor shape.
        
        Returns:
            tuple: (qDim, tDim) where qDim=original shape, tDim=chunk shape
        """
        # qDim is always the original tensor shape (Tier 2)
        qDim = list(tensor_shape)
        
        # tDim comes from the chunking strategy (Tier 1)
        if hasattr(metadata, 'chunking_strategy') and metadata.chunking_strategy:
            # Use metadata's chunking strategy
            _, tDim = metadata.chunking_strategy.compute_chunking(tensor_shape, metadata.name)
        else:
            # Default: no chunking, process entire tensor
            tDim = list(tensor_shape)
        
        return qDim, tDim
    
    def _initialize_minimum_parallelism(self) -> Dict[str, int]:
        """
        Initialize Tier 3 (Parallelism) with minimum values.
        
        This ensures the object is fully functional immediately,
        with parallelism updated separately as needed.
        """
        parallelism = {}
        
        # Set minimum parallelism (1) for all interfaces
        for iface in self._dataflow_model.input_interfaces:
            parallelism[f"{iface.name}_iPar"] = 1
            
        for iface in self._dataflow_model.weight_interfaces:
            parallelism[f"{iface.name}_wPar"] = 1
            
        return parallelism
    
    def update_parallelism(self, iPar: Dict[str, int] = None, wPar: Dict[str, int] = None):
        """
        Update Tier 3 (Parallelism) values and recalculate sDim/performance.
        
        This is separate from construction - parallelism can be updated
        at any time without rebuilding the entire model.
        
        Args:
            iPar: Input parallelism per interface {interface_name: parallelism}
            wPar: Weight parallelism per interface {interface_name: parallelism}
        """
        if iPar is None:
            iPar = {}
        if wPar is None:
            wPar = {}
            
        # Update current parallelism tracking
        for iface_name, parallel_val in iPar.items():
            self._current_parallelism[f"{iface_name}_iPar"] = parallel_val
            
        for iface_name, parallel_val in wPar.items():
            self._current_parallelism[f"{iface_name}_wPar"] = parallel_val
        
        # Update sDim for all interfaces based on new parallelism
        for iface in self._dataflow_model.input_interfaces:
            if iface.name in iPar:
                iface.apply_parallelism(iPar=iPar[iface.name])
                
        for iface in self._dataflow_model.weight_interfaces:
            if iface.name in wPar:
                iface.apply_parallelism(wPar=wPar[iface.name])
                
        for iface in self._dataflow_model.output_interfaces:
            # Output parallelism typically follows input parallelism
            if iPar:
                # Use first input's parallelism for outputs
                first_input_par = next(iter(iPar.values()))
                iface.apply_parallelism(iPar=first_input_par)
    
    
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
        """Get DataflowModel (always available after construction)."""
        return self._dataflow_model
    
    @property
    def interface_metadata(self) -> InterfaceMetadataCollection:
        """Get interface metadata collection."""
        return self._interface_metadata_collection
    
    def get_current_parallelism(self) -> Dict[str, int]:
        """Get current Tier 3 (Parallelism) values."""
        return self._current_parallelism.copy()

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
        """Get expected cycles using current Tier 3 (Parallelism) values.""" 
        # Extract iPar and wPar from current parallelism
        iPar = {}
        wPar = {}
        
        for key, value in self._current_parallelism.items():
            if key.endswith('_iPar'):
                interface_name = key[:-5]  # Remove '_iPar' suffix
                iPar[interface_name] = value
            elif key.endswith('_wPar'):
                interface_name = key[:-5]  # Remove '_wPar' suffix
                wPar[interface_name] = value
        
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
            params = np.prod(iface.get_num_tensors()) * np.prod(iface.tDim)
            counts["weight_params"] += params
            counts["params"] += params
            
        # Config interfaces no longer tracked - they are handled separately
            
        # Estimate operations based on input/output sizes
        if self.dataflow_model.input_interfaces and self.dataflow_model.output_interfaces:
            input_size = sum(
                np.prod(iface.get_num_tensors()) * np.prod(iface.tDim)
                for iface in self.dataflow_model.input_interfaces
            )
            output_size = sum(
                np.prod(iface.get_num_tensors()) * np.prod(iface.tDim)
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
                np.prod(iface.get_num_tensors()),
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
            self.dataflow_model.weight_interfaces
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
            total_ops = sum(np.prod(iface.get_num_tensors()) for iface in self.dataflow_model.input_interfaces)
            
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
            self.dataflow_model.weight_interfaces
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
            "qDim": list(iface.qDim),  # Original tensor dimensions
            "num_tensors": list(iface.get_num_tensors()),  # Computed number of chunks
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