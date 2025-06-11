############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# AUTO-GENERATED: AutoThresholdingAxiRTLBackendRTLBackend for thresholding_axi
# Generated: 2025-06-11T05:02:22.631262
# Generator: Unified HWKG with Interface-Wise Dataflow Modeling
#
# DATAFLOW-MODEL-POWERED RTL BACKEND
# This RTLBackend uses DataflowModel for interface configuration and
# RTL parameter generation with mathematical foundation.
############################################################################

import os
import numpy as np
from typing import Dict, Any, List, Optional

# Import unified dataflow components
from brainsmith.dataflow.core.auto_rtl_backend import AutoRTLBackend
from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType

# Try to import FINN components (optional for development)
try:
    from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
    from qonnx.core.datatype import DataType
    FINN_AVAILABLE = True
except ImportError:
    FINN_AVAILABLE = False
    # Create stub base class for development
    class RTLBackend:
        def __init__(self):
            pass


class AutoThresholdingAxiRTLBackendRTLBackend(AutoRTLBackend):
    """
    Auto-generated RTLBackend for thresholding_axi kernel.
    
    This class uses the unified HWKG approach with Interface-Wise Dataflow Modeling:
    - AutoRTLBackend provides mathematical foundation for RTL generation
    - DataflowModel-driven interface configuration and parameter generation
    - Automatic signal assignments and wrapper generation
    - Runtime-configurable RTL parameters
    
    Interfaces (4 total):
- ap: config (UINT1, 1 bits)
- s_axis: input (UINT7, 7 bits)
- m_axis: output (UINT7, 7 bits)
- s_axilite: config (UINT31, 31 bits)
    """
    
    def __init__(self):
        """
        Initialize AutoThresholdingAxiRTLBackendRTLBackend with DataflowModel-based interface configuration.
        
        The AutoRTLBackend base class provides all RTL generation capabilities
        using the dataflow interfaces configuration.
        """
        super().__init__()
        
        # Set dataflow interfaces configuration for AutoRTLBackend
        self.dataflow_interfaces = {
            "ap": {
                "interface_type": "config",
                "dtype": {
                    "finn_type": "UINT1",
                    "signed": false,
                    "bitwidth": 1
                },
                "tensor_dims": [128],
                "block_dims": [128],
                "stream_dims": [1],
                "axi_metadata": {
                    "protocol": "axi_stream",
                    "data_width": 1
                }
            },
            "s_axis": {
                "interface_type": "input",
                "dtype": {
                    "finn_type": "UINT7",
                    "signed": false,
                    "bitwidth": 7
                },
                "tensor_dims": [7],
                "block_dims": [7],
                "stream_dims": [1],
                "axi_metadata": {
                    "protocol": "axi_stream",
                    "data_width": 7
                }
            },
            "m_axis": {
                "interface_type": "output",
                "dtype": {
                    "finn_type": "UINT7",
                    "signed": false,
                    "bitwidth": 7
                },
                "tensor_dims": [7],
                "block_dims": [7],
                "stream_dims": [1],
                "axi_metadata": {
                    "protocol": "axi_stream",
                    "data_width": 7
                }
            },
            "s_axilite": {
                "interface_type": "config",
                "dtype": {
                    "finn_type": "UINT31",
                    "signed": false,
                    "bitwidth": 31
                },
                "tensor_dims": [31],
                "block_dims": [31],
                "stream_dims": [1],
                "axi_metadata": {
                    "protocol": "axi_stream",
                    "data_width": 31
                }
            },
        }
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """
        Get node attribute types with RTL backend enhancements.
        
        Uses AutoRTLBackend's enhanced attribute system with dataflow
        interface configuration.
        """
        # Get enhanced attributes from AutoRTLBackend base class
        attrs = super().get_enhanced_nodeattr_types()
        
        # Add kernel-specific RTL configuration
        rtl_specific_attrs = {
            # RTL module configuration
            "rtl_module_name": ("s", False, "thresholding_axi"),
            "wrapper_module_name": ("s", False, "thresholding_axi_wrapper"),
            
            # Generation configuration
            "generation_method": ("s", False, "unified_hwkg_dataflow_modeling"),
            "kernel_type": ("s", False, "thresholding_axi"),
        }
        
        attrs.update(rtl_specific_attrs)
        return attrs
    
    def code_generation_dict(self) -> Dict[str, Any]:
        """
        Generate RTL code generation dictionary using DataflowModel.
        
        This uses AutoRTLBackend's enhanced code generation with
        dataflow interface metadata.
        """
        # Get enhanced code generation dict from AutoRTLBackend
        codegen_dict = super().generate_enhanced_code_dict()
        
        # Add kernel-specific RTL generation parameters
        kernel_specific_params = {
            "kernel_name": "thresholding_axi",
            "module_name": self.get_nodeattr("rtl_module_name") or "thresholding_axi",
            "wrapper_name": self.get_nodeattr("wrapper_module_name") or "thresholding_axi_wrapper",
            
            # Interface-specific parameters for RTL generation
            "interface_count": 4,
            "input_interface_count": 1,
            "output_interface_count": 1,
            "weight_interface_count": 0,
            
            # Add interface-specific RTL parameters
            "ap_width": 1,
            "ap_type": "config",
            "s_axis_width": 7,
            "s_axis_type": "input",
            "m_axis_width": 7,
            "m_axis_type": "output",
            "s_axilite_width": 31,
            "s_axilite_type": "config",
        }
        
        codegen_dict.update(kernel_specific_params)
        return codegen_dict
    
    # All RTL generation methods (generate_params, etc.) are inherited from
    # AutoRTLBackend and use the dataflow interfaces configuration.
    # No placeholder implementations needed!
    
    def generate_rtl_wrapper(self, model, path: str) -> str:
        """
        Generate RTL wrapper using DataflowModel interface information.
        
        This uses AutoRTLBackend's wrapper generation with the dataflow
        interface configuration for accurate signal assignments.
        """
        # Use base class wrapper generation with our interface configuration
        wrapper_file = super().generate_params(model, path)
        
        # Log successful wrapper generation
        if hasattr(self, '_log_wrapper_generation'):
            self._log_wrapper_generation("thresholding_axi", wrapper_file)
        
        return wrapper_file


# Factory function for easy instantiation
def create_thresholding_axi_rtlbackend() -> AutoThresholdingAxiRTLBackendRTLBackend:
    """
    Factory function for creating AutoThresholdingAxiRTLBackendRTLBackend instances.
    
    Returns:
        AutoThresholdingAxiRTLBackendRTLBackend: Configured RTLBackend instance
    """
    return AutoThresholdingAxiRTLBackendRTLBackend()


# Export the main class for FINN integration
__all__ = ["AutoThresholdingAxiRTLBackendRTLBackend", "create_thresholding_axi_rtlbackend"]