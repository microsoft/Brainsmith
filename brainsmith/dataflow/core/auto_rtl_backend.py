"""
AutoRTLBackend base class for auto-generated RTL backend implementations.

This module provides the base class for all auto-generated RTLBackend classes,
implementing standardized methods for RTL code generation and interface management.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
from .class_naming import generate_class_name, generate_backend_class_name

# Try to import FINN, but make it optional for development
try:
    from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
    from qonnx.core.datatype import DataType
    FINN_AVAILABLE = True
except ImportError:
    FINN_AVAILABLE = False
    # Create minimal stub base class for standalone operation
    class RTLBackend:
        def __init__(self):
            pass
            
        def get_nodeattr_types(self):
            return {}
            
        def get_nodeattr(self, name):
            return None
            
    class DataType:
        """Stub DataType for development without FINN"""
        @staticmethod
        def bitwidth():
            return 8
            
        def __getitem__(self, key):
            return DataType()


class AutoRTLBackend(RTLBackend):
    """
    Base class for auto-generated RTLBackend implementations.
    
    This class provides all standardized method implementations for RTL
    code generation, dramatically reducing the amount of generated code
    needed in templates.
    """
    
    def __init__(self):
        super().__init__()
        
        # Dataflow interfaces must be set by subclass
        self.dataflow_interfaces = {}
        
        # Cache for interface lists
        self._input_interfaces = None
        self._output_interfaces = None
        self._weight_interfaces = None
        self._config_interfaces = None
        
    @property
    def input_interfaces(self) -> List[str]:
        """Get list of input interface names."""
        if self._input_interfaces is None:
            self._input_interfaces = [
                name for name, config in self.dataflow_interfaces.items()
                if config["interface_type"] == "INPUT"
            ]
        return self._input_interfaces
        
    @property
    def output_interfaces(self) -> List[str]:
        """Get list of output interface names."""
        if self._output_interfaces is None:
            self._output_interfaces = [
                name for name, config in self.dataflow_interfaces.items()
                if config["interface_type"] == "OUTPUT"
            ]
        return self._output_interfaces
        
    @property
    def weight_interfaces(self) -> List[str]:
        """Get list of weight interface names."""
        if self._weight_interfaces is None:
            self._weight_interfaces = [
                name for name, config in self.dataflow_interfaces.items()
                if config["interface_type"] == "WEIGHT"
            ]
        return self._weight_interfaces
        
    @property
    def config_interfaces(self) -> List[str]:
        """Get list of config interface names."""
        if self._config_interfaces is None:
            self._config_interfaces = [
                name for name, config in self.dataflow_interfaces.items()
                if config["interface_type"] == "CONFIG"
            ]
        return self._config_interfaces
        
    def get_enhanced_nodeattr_types(self) -> Dict[str, Any]:
        """
        Get enhanced node attribute types for RTL backend configuration.
        
        This method adds RTL-specific attributes to the base FINN attributes.
        Subclasses should override get_nodeattr_types() and call this method.
        """
        attrs = {
            # Clock and reset configuration
            "clk_name": ("s", False, "ap_clk"),
            "rst_name": ("s", False, "ap_rst_n"),
            "rst_active_low": ("b", False, True),
            
            # Parameter file configuration
            "param_file_format": ("s", False, "hex", {"hex", "binary", "decimal"}),
            "param_file_extension": ("s", False, ".dat"),
            
            # RTL generation options
            "generate_wrapper": ("b", False, True),
            "wrapper_name": ("s", False, "wrapper"),
            "top_module_name": ("s", False, "top"),
        }
        
        # Add interface signal naming configuration
        for iface_name in self.dataflow_interfaces:
            attrs[f"{iface_name}_signal_prefix"] = ("s", False, f"{iface_name}_")
            
        return attrs
        
    def generate_interface_definitions(self) -> List[Dict[str, Any]]:
        """Generate RTL interface definitions from dataflow interfaces."""
        interfaces = []
        
        for iface_name, iface_config in self.dataflow_interfaces.items():
            interface_def = {
                "name": iface_name,
                "type": iface_config["interface_type"],
                "signal_prefix": self.get_nodeattr(f"{iface_name}_signal_prefix") or f"{iface_name}_",
                "width": self.calculate_interface_width(iface_name),
                "direction": "input" if iface_config["interface_type"] in ["INPUT", "WEIGHT", "CONFIG"] else "output",
                "protocol": "axi_stream",  # Default protocol
                "qDim": iface_config["qDim"],
                "tDim": iface_config["tDim"],
                "sDim": iface_config["sDim"],
            }
            interfaces.append(interface_def)
            
        return interfaces
        
    def generate_signal_assignments(self) -> List[Dict[str, str]]:
        """Generate RTL signal assignments for interfaces."""
        assignments = []
        
        # Clock and reset assignments
        clk_name = self.get_nodeattr("clk_name") or "ap_clk"
        rst_name = self.get_nodeattr("rst_name") or "ap_rst_n"
        
        assignments.extend([
            {"target": "clk", "source": clk_name},
            {"target": "rst", "source": rst_name},
        ])
        
        # Interface signal assignments
        for iface_name in self.dataflow_interfaces:
            prefix = self.get_nodeattr(f"{iface_name}_signal_prefix") or f"{iface_name}_"
            assignments.extend([
                {"target": f"{iface_name}_tdata", "source": f"{prefix}tdata"},
                {"target": f"{iface_name}_tvalid", "source": f"{prefix}tvalid"},
                {"target": f"{iface_name}_tready", "source": f"{prefix}tready"},
            ])
            
        return assignments
        
    def generate_parameter_overrides(self) -> Dict[str, Any]:
        """Generate RTL parameter overrides based on current configuration."""
        params = {}
        
        for iface_name, iface_config in self.dataflow_interfaces.items():
            # Add interface parameters
            params.update({
                f"{iface_name.upper()}_WIDTH": self.calculate_interface_width(iface_name),
                f"{iface_name.upper()}_DEPTH": np.prod(iface_config["qDim"]) * np.prod(iface_config["tDim"]),
            })
            
        return params
        
    def generate_clock_assignments(self) -> List[str]:
        """Generate clock signal assignments."""
        clk_name = self.get_nodeattr("clk_name") or "ap_clk"
        return [f"assign clk = {clk_name};"]
        
    def generate_reset_assignments(self) -> List[str]:
        """Generate reset signal assignments."""
        rst_name = self.get_nodeattr("rst_name") or "ap_rst_n"
        rst_active_low = self.get_nodeattr("rst_active_low")
        
        if rst_active_low:
            return [f"assign rst = ~{rst_name};"]
        else:
            return [f"assign rst = {rst_name};"]
            
    def calculate_interface_width(self, interface_name: str) -> int:
        """
        Calculate bit width for an interface.
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            Interface width in bits
        """
        if interface_name not in self.dataflow_interfaces:
            raise KeyError(f"Interface {interface_name} not found")
            
        interface_config = self.dataflow_interfaces[interface_name]
        
        # Get datatype
        dtype_name = interface_config["dtype"].get("finn_type", "UINT8")
        if FINN_AVAILABLE:
            dtype = DataType[dtype_name]
            bitwidth = dtype.bitwidth()
        else:
            # Extract bitwidth from datatype name for development
            import re
            match = re.search(r'\d+', dtype_name)
            bitwidth = int(match.group()) if match else 8
            
        # Get parallelism factor
        parallel_factor = self.get_nodeattr(f"{interface_name}_parallel") or 1
        
        # Calculate width: datatype_bits * parallel_factor
        width = bitwidth * parallel_factor
        
        # Round up to byte boundary for AXI compliance
        return ((width + 7) // 8) * 8
        
    def generate_enhanced_code_dict(self) -> Dict[str, Any]:
        """
        Generate enhanced code generation dictionary with dataflow metadata.
        
        Returns:
            Dict: Enhanced code generation parameters for RTL templates
        """
        # Get base code generation dict if available
        if hasattr(super(), 'code_generation_dict'):
            codegen_dict = super().code_generation_dict()
        else:
            codegen_dict = {}
            
        # Add interface-specific code generation parameters
        interface_dict = {
            "interfaces": self.generate_interface_definitions(),
            "signals": self.generate_signal_assignments(),
            "parameters": self.generate_parameter_overrides(),
            "clocks": self.generate_clock_assignments(),
            "resets": self.generate_reset_assignments(),
        }
        
        codegen_dict.update(interface_dict)
        return codegen_dict
        
    def generate_params(self, model, path):
        """
        Generate RTL parameter files based on dataflow interface configuration.
        
        Args:
            model: ONNX model node
            path: Output directory for parameter files
        """
        os.makedirs(path, exist_ok=True)
        
        # Generate parameter files for weight interfaces
        for iface in self.weight_interfaces:
            self._generate_weight_params(iface, model, path)
            
        # Generate configuration parameter files
        for iface in self.config_interfaces:
            self._generate_config_params(iface, model, path)
            
    def _generate_weight_params(self, interface_name: str, model, path: str):
        """Generate parameter file for a weight interface."""
        interface_config = self.dataflow_interfaces[interface_name]
        
        # Calculate parameter dimensions
        qDim = interface_config["qDim"]
        tDim = interface_config["tDim"]
        total_elements = np.prod(qDim) * np.prod(tDim)
        
        # Get datatype configuration
        dtype_name = model.get_nodeattr(f"{interface_name}_dtype") or interface_config["dtype"]["finn_type"]
        
        # Extract weights from model (placeholder - customize based on actual source)
        weights = self._extract_weights_for_interface(interface_name, model)
        if weights is None:
            # Generate placeholder weights for testing
            weights = np.random.randn(total_elements).astype(np.float32)
            
        # Convert to target datatype
        if FINN_AVAILABLE:
            dtype = DataType[dtype_name]
            if dtype.is_integer():
                # Quantize floating point weights to integer type
                weights = self._quantize_weights(weights, dtype)
        
        # Write parameter file
        param_format = model.get_nodeattr("param_file_format") or "hex"
        param_ext = model.get_nodeattr("param_file_extension") or ".dat"
        param_file = os.path.join(path, f"{interface_name}_weights{param_ext}")
        
        self._write_param_file(weights, param_file, param_format, dtype_name)
        
    def _generate_config_params(self, interface_name: str, model, path: str):
        """Generate configuration file for a config interface."""
        interface_config = self.dataflow_interfaces[interface_name]
        
        # Generate configuration values based on interface constraints
        config_values = self._generate_config_values(interface_name, model)
        
        # Write configuration file
        config_file = os.path.join(path, f"{interface_name}_config.dat")
        with open(config_file, 'w') as f:
            for value in config_values:
                f.write(f"{value:08x}\n")
                
    def _extract_weights_for_interface(self, interface_name: str, model) -> Optional[np.ndarray]:
        """
        Extract weights for a specific interface from the model.
        
        Args:
            interface_name: Name of the weight interface
            model: ONNX model node
            
        Returns:
            numpy array of weights or None if not found
        """
        # This is a placeholder - in practice, this would extract weights
        # from the ONNX model's initializers based on interface mapping
        
        # For now, return None to trigger placeholder weight generation
        return None
        
    def _generate_config_values(self, interface_name: str, model) -> List[int]:
        """
        Generate configuration values for a config interface.
        
        Args:
            interface_name: Name of the config interface
            model: ONNX model node
            
        Returns:
            List of configuration values
        """
        interface_config = self.dataflow_interfaces[interface_name]
        
        # Generate configuration based on interface constraints and current settings
        config_values = []
        
        # Add parallelism configuration
        parallel_factor = model.get_nodeattr(f"{interface_name}_parallel") or 1
        config_values.append(parallel_factor)
        
        # Add datatype configuration
        dtype_name = model.get_nodeattr(f"{interface_name}_dtype") or interface_config["dtype"]["finn_type"]
        dtype_code = self._encode_datatype(dtype_name)
        config_values.append(dtype_code)
        
        # Add dimension information
        qDim = interface_config["qDim"]
        tDim = interface_config["tDim"]
        config_values.extend(qDim)
        config_values.extend(tDim)
        
        return config_values
        
    def _quantize_weights(self, weights: np.ndarray, dtype) -> np.ndarray:
        """
        Quantize floating point weights to target datatype.
        
        Args:
            weights: Input floating point weights
            dtype: Target FINN DataType
            
        Returns:
            Quantized weights as numpy array
        """
        if not FINN_AVAILABLE:
            # Simple quantization for development
            return np.round(weights).astype(np.int32)
            
        if dtype.is_integer():
            # Simple linear quantization
            if dtype.min() < 0:  # Signed
                scale = max(abs(weights.min()), abs(weights.max())) / (2 ** (dtype.bitwidth() - 1) - 1)
            else:  # Unsigned
                scale = (weights.max() - weights.min()) / (2 ** dtype.bitwidth() - 1)
                weights = weights - weights.min()
                
            quantized = np.round(weights / scale).astype(np.int32)
            
            # Clip to datatype range
            quantized = np.clip(quantized, dtype.min(), dtype.max())
            
            return quantized
        else:
            # For floating point types, just cast
            return weights.astype(dtype.to_numpy_type())
            
    def _write_param_file(self, weights: np.ndarray, filepath: str, format_type: str, dtype_name: str):
        """
        Write parameter file in specified format.
        
        Args:
            weights: Weight values to write
            filepath: Output file path
            format_type: Format type (hex, binary, decimal)
            dtype_name: DataType name for formatting
        """
        # Get bitwidth for formatting
        if FINN_AVAILABLE:
            bitwidth = DataType[dtype_name].bitwidth()
        else:
            # Extract from name for development
            import re
            match = re.search(r'\d+', dtype_name)
            bitwidth = int(match.group()) if match else 8
            
        with open(filepath, 'w') as f:
            for weight in weights.flatten():
                if format_type == "hex":
                    f.write(f"{int(weight):0{bitwidth//4}x}\n")
                elif format_type == "binary":
                    f.write(f"{int(weight):0{bitwidth}b}\n")
                else:  # decimal
                    f.write(f"{int(weight)}\n")
                    
    def _encode_datatype(self, dtype_name: str) -> int:
        """
        Encode FINN DataType name to integer code for RTL configuration.
        
        Args:
            dtype_name: FINN DataType name
            
        Returns:
            Integer encoding of the datatype
        """
        # Simple encoding scheme - can be customized
        dtype_codes = {
            "UINT1": 1, "UINT2": 2, "UINT4": 4, "UINT8": 8,
            "UINT16": 16, "UINT32": 32,
            "INT1": 101, "INT2": 102, "INT4": 104, "INT8": 108,
            "INT16": 116, "INT32": 132,
            "FLOAT16": 216, "FLOAT32": 232,
        }
        
        return dtype_codes.get(dtype_name, 8)  # Default to UINT8 code
        
    @staticmethod
    def generate_class_name(kernel_name: str) -> str:
        """
        Generate proper CamelCase backend class name from kernel name.
        
        Args:
            kernel_name: Underscore-separated kernel name
            
        Returns:
            CamelCase backend class name
        """
        return generate_backend_class_name(kernel_name)