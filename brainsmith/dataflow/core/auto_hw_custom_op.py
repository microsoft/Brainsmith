"""
AutoHWCustomOp base class for auto-generated hardware custom operators.

This module provides the base class for all auto-generated HWCustomOp classes,
implementing standardized methods that can be fully determined from the
dataflow interface metadata.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .class_naming import generate_class_name
from .dataflow_model import DataflowModel, ParallelismConfiguration
from .dataflow_interface import DataflowInterface, DataflowInterfaceType

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
    """
    
    def __init__(self, onnx_node, dataflow_model: DataflowModel, **kwargs):
        super().__init__(onnx_node, **kwargs)
        
        # Store the DataflowModel instance - single source of truth
        self.dataflow_model = dataflow_model
        
        # No more interface storage or caching!
        # All interface access goes through dataflow_model
        
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
            params = np.prod(iface.qDim) * np.prod(iface.tDim)
            counts["weight_params"] += params
            counts["params"] += params
            
        # Count config parameters
        for iface in self.dataflow_model.config_interfaces:
            params = np.prod(iface.qDim) * np.prod(iface.tDim)
            counts["config_params"] += params
            counts["params"] += params
            
        # Estimate operations based on input/output sizes
        if self.dataflow_model.input_interfaces and self.dataflow_model.output_interfaces:
            input_size = sum(
                np.prod(iface.qDim) * np.prod(iface.tDim)
                for iface in self.dataflow_model.input_interfaces
            )
            output_size = sum(
                np.prod(iface.qDim) * np.prod(iface.tDim)
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
                np.prod(iface.qDim),
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
        for iface in self.dataflow_model.get_all_interfaces():
            parallel_attr = f"{iface.name}_parallel"
            config[iface.name] = self.get_nodeattr(parallel_attr) or 1
        return config
                        
    def estimate_bram_usage(self) -> int:
        """Estimate BRAM usage using DataflowModel resource requirements."""
        parallelism_config = self._get_current_parallelism_config()
        resources = self.dataflow_model.get_resource_requirements(parallelism_config)
        
        memory_bits = resources["memory_bits"]
        bram_capacity = 18 * 1024  # BRAM18K
        
        # Apply estimation mode scaling
        estimation_mode = self.get_nodeattr("resource_estimation_mode")
        scale_factor = {"conservative": 1.5, "optimistic": 0.7, "automatic": 1.0}.get(estimation_mode, 1.0)
        
        return int(np.ceil((memory_bits * scale_factor) / bram_capacity))
        
    def estimate_lut_usage(self) -> int:
        """Estimate LUT usage using DataflowModel resource requirements."""
        parallelism_config = self._get_current_parallelism_config()
        resources = self.dataflow_model.get_resource_requirements(parallelism_config)
        
        lut_ops = resources["lut_ops"]
        
        # Apply estimation mode scaling
        estimation_mode = self.get_nodeattr("resource_estimation_mode")
        scale_factor = {"conservative": 1.5, "optimistic": 0.7, "automatic": 1.0}.get(estimation_mode, 1.0)
        
        return int(lut_ops * scale_factor)
        
    def estimate_dsp_usage(self, fpgapart: str = "xczu7ev") -> int:
        """Estimate DSP usage using DataflowModel resource requirements."""
        parallelism_config = self._get_current_parallelism_config()
        resources = self.dataflow_model.get_resource_requirements(parallelism_config)
        
        dsp_ops = resources["dsp_ops"]
        
        # Apply estimation mode scaling
        estimation_mode = self.get_nodeattr("resource_estimation_mode")
        scale_factor = {"conservative": 1.2, "optimistic": 0.8, "automatic": 1.0}.get(estimation_mode, 1.0)
        
        return int(np.ceil(dsp_ops * scale_factor))
        
    def get_interface_config(self, interface_name: str) -> Dict[str, Any]:
        """Get configuration for a specific interface using DataflowModel."""
        iface = self.dataflow_model.get_interface(interface_name)
        if not iface:
            raise KeyError(f"Interface '{interface_name}' not found in dataflow model")
            
        config = {
            "interface_type": iface.interface_type.name,
            "dtype": {
                "finn_type": iface.dtype.finn_type,
                "signed": iface.dtype.signed
            },
            "qDim": list(iface.q_dim),
            "tDim": list(iface.t_dim),
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