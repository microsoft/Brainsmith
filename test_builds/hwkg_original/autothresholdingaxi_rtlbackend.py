############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated RTLBackend for thresholding_axi
# Generated from: examples/thresholding/thresholding_axi.sv
# Generation timestamp: 2025-06-10T03:17:03.727363
#
# RUNTIME-CONFIGURABLE HARDWARE COMPONENT
# This RTLBackend extracts dimensions at runtime from associated HWCustomOp.
# Dimensions are not hardcoded during generation.
############################################################################

import os
import numpy as np
from typing import Dict, Any, List, Optional

# FINN imports for RTLBackend integration
from finn.backends.fpgadataflow.rtlbackend import RTLBackend

# Import dataflow framework components for enhanced functionality
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
from brainsmith.dataflow.core.dataflow_model import DataflowModel


class AutoThresholdingAxiRTLBackend(RTLBackend):
    """
    RTL Backend for thresholding_axi kernel.
    
    Provides RTL generation with dataflow modeling integration.
    
    Generated from: examples/thresholding/thresholding_axi.sv
    
    Interfaces:
    - ap: control
    - s_axis: input
    - m_axis: output
    - s_axilite: config
    """
    
    def __init__(self, model, dataflow_model=None):
        """
        Initialize RTLBackend with optional dataflow model.
        
        Args:
            model: FINN model wrapper
            dataflow_model: Optional dataflow model for enhanced generation
        """
        super().__init__(model)
        
        # Store dataflow model for enhanced RTL generation
        self.dataflow_model = dataflow_model
        self._associated_hwcustomop = None  # Set by FINN compiler at runtime
        
        # Set kernel-specific paths
        self.kernel_name = "thresholding_axi"
        self.rtl_template_path = os.path.join(
            os.path.dirname(__file__), 
            "rtl", 
            "thresholding_axi_wrapper.v"
        )
    
    def set_associated_hwcustomop(self, hwcustomop):
        """
        Set the associated HWCustomOp for runtime dimension extraction.
        
        This method should be called by the FINN compiler when the RTL backend
        is associated with its corresponding HWCustomOp.
        """
        self._associated_hwcustomop = hwcustomop
    
    def get_runtime_interface_config(self, interface_name: str):
        """
        Get runtime configuration for an interface from the associated HWCustomOp.
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            Dict with runtime interface configuration
        """
        if not self._associated_hwcustomop:
            raise RuntimeError(
                f"Cannot get runtime interface config for {interface_name}: "
                f"No associated HWCustomOp available. RTL backend must be properly "
                f"linked to its HWCustomOp by the FINN compiler."
            )
        
        try:
            return self._associated_hwcustomop.get_interface_config(interface_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract runtime interface config for {interface_name}: {e}. "
                f"The HWCustomOp must have a valid ModelWrapper for dimension extraction."
            )
    
    def get_rtl_file_list(self):
        """
        Get list of RTL files for this kernel.
        
        Returns list of RTL files needed for synthesis.
        """
        files = []
        
        # Add main kernel RTL file
        files.append("thresholding_axi.v")
        
        # Add any kernel-specific RTL files
        # Example: Add memory initialization files for thresholding
        if self.get_nodeattr("USE_AXILITE"):
            files.append("thresholding_axi_lut.v")
        
        return files
    
    def generate_hdl(self):
        """
        Generate HDL instantiation.
        
        Creates wrapper and instantiation code for the kernel.
        """
        # Get basic HDL generation from parent
        hdl_code = super().generate_hdl()
        
        # Add any kernel-specific HDL customization here
        return hdl_code
    
    def get_verilog_parameters(self):
        """
        Get Verilog parameters for instantiation.
        
        Returns dictionary of parameter name -> value mappings.
        """
        params = {}
        
        # Add RTL parameters from kernel specification
        params["N"] = self.get_nodeattr("N")
        params["WI"] = self.get_nodeattr("WI")
        params["WT"] = self.get_nodeattr("WT")
        params["C"] = self.get_nodeattr("C")
        params["SIGNED"] = self.get_nodeattr("SIGNED")
        params["FPARG"] = self.get_nodeattr("FPARG")
        params["BIAS"] = self.get_nodeattr("BIAS")
        params["THRESHOLDS_PATH"] = self.get_nodeattr("THRESHOLDS_PATH")
        params["USE_AXILITE"] = self.get_nodeattr("USE_AXILITE")
        params["DEPTH_TRIGGER_URAM"] = self.get_nodeattr("DEPTH_TRIGGER_URAM")
        params["DEPTH_TRIGGER_BRAM"] = self.get_nodeattr("DEPTH_TRIGGER_BRAM")
        params["DEEP_PIPELINE"] = self.get_nodeattr("DEEP_PIPELINE")
        
        # Add dataflow-derived parameters
        
        
        return params
    
    def code_generation_dict(self):
        """
        Generate code generation dictionary for RTL templates.
        
        Returns dictionary with all information needed for RTL generation.
        """
        codegen_dict = {
            "kernel_name": "thresholding_axi",
            "wrapper_name": f"thresholding_axi_wrapper",
            "top_module_name": "thresholding_axi",
            
            # Basic information
            "source_file": "examples/thresholding/thresholding_axi.sv",
            "generation_timestamp": "2025-06-10T03:17:03.727363",
            
            # Parameters
            "verilog_parameters": self.get_verilog_parameters(),
            "rtl_files": self.get_rtl_file_list(),
            
            # Interface information - dimensions extracted at runtime
            "interfaces": {
                "ap": {
                    "type": "control",
                    "direction": "output",
                    "dtype": "UINT32",
                    # Dimensions extracted at runtime from dataflow model
                    "runtime_extraction": True
                },
                "s_axis": {
                    "type": "input",
                    "direction": "input",
                    "dtype": "UINT8",
                    # Dimensions extracted at runtime from dataflow model
                    "runtime_extraction": True
                },
                "m_axis": {
                    "type": "output",
                    "direction": "output",
                    "dtype": "UINT8",
                    # Dimensions extracted at runtime from dataflow model
                    "runtime_extraction": True
                },
                "s_axilite": {
                    "type": "config",
                    "direction": "output",
                    "dtype": "UINT32",
                    # Dimensions extracted at runtime from dataflow model
                    "runtime_extraction": True
                },
            },
            
            # Interface groupings for convenient access
            "input_interfaces": [],
            "output_interfaces": [],
            "weight_interfaces": [],
            "config_interfaces": [],
        }
        
        return codegen_dict
    
    def generate_params(self, model, path):
        """
        Generate parameter files for RTL synthesis.
        
        Args:
            model: FINN model wrapper
            path: Output directory for generated files
        """
        os.makedirs(path, exist_ok=True)
        
        
        # Generate main parameter file
        self._generate_main_params(model, path)
    
    
    def _generate_main_params(self, model, path):
        """
        Generate main parameter file.
        
        Args:
            model: FINN model wrapper  
            path: Output directory
        """
        param_file = os.path.join(path, "thresholding_axi_params.txt")
        
        with open(param_file, 'w') as f:
            f.write("# Auto-generated parameters for thresholding_axi\n")
            f.write(f"# Source: examples/thresholding/thresholding_axi.sv\n") 
            f.write(f"# Generated: 2025-06-10T03:17:03.727363\n\n")
            
            # Write Verilog parameters
            params = self.get_verilog_parameters()
            for name, value in params.items():
                f.write(f"{name}={value}\n")
            
            f.write(f"\n# Interface Information\n")
            f.write(f"NUM_INTERFACES=4\n")
            f.write(f"NUM_INPUT_INTERFACES=0\n")
            f.write(f"NUM_OUTPUT_INTERFACES=0\n")
            f.write(f"NUM_WEIGHT_INTERFACES=0\n")
        
        print(f"Generated parameter file: {param_file}")