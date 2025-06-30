"""
AutoRTLBackend base class for auto-generated RTL backend implementations.

This module provides the base class for all auto-generated RTLBackend classes,
implementing standardized methods for template processing, file management,
resource estimation, and FINN integration that are common across all RTL operations.
"""

import os
import shutil
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_memutil_alternatives


class AutoRTLBackend(RTLBackend):
    """
    Base class for auto-generated RTLBackend implementations.
    
    Provides standardized functionality for:
    - Dual execution mode handling (cppsim + rtlsim)
    - Template variable generation and processing
    - File management and HDL generation
    - TCL command generation for IPI integration
    - Resource estimation using standard formulas
    - finn-rtllib integration patterns
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize AutoRTLBackend with standard RTL backend setup."""
        # For multiple inheritance, pass through all arguments to the next class in MRO
        # This allows proper cooperative inheritance
        super().__init__(*args, **kwargs)
    
    @property
    @abstractmethod
    def finn_rtllib_module(self) -> str:
        """
        Return the finn-rtllib module name for this operation.
        
        This is used to locate template files and supporting RTL files.
        Subclasses must implement this to specify their finn-rtllib directory.
        
        Returns:
            str: Module name (e.g., "mvu", "thresholding", "dwc")
        """
        pass
    
    def prepare_codegen_rtl_values(self, model):
        """
        Prepare template variables for RTL code generation.
        
        Subclasses must override this method to implement FINN's standard pattern.
        
        Returns:
            Dict[str, List[str]]: Template variable mappings in FINN format
        """
        raise NotImplementedError("Subclasses must implement prepare_codegen_rtl_values()")
    
    def get_base_template_variables(self) -> Dict[str, Any]:
        """
        Get base template variables common to all RTL operations.
        
        These variables are derived from standard node attributes and
        provide basic functionality needed by most operations.
        
        Returns:
            Dict[str, Any]: Base template variables
        """
        return {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "FORCE_BEHAVIORAL": 0,  # Default to synthesizable RTL
        }
    
    def get_all_template_variables(self) -> Dict[str, Any]:
        """
        Get complete template variable set combining base and operation-specific variables.
        
        Returns:
            Dict[str, Any]: Complete template variable mappings
        """
        variables = self.get_base_template_variables()
        
        # Get operation-specific variables from prepare_codegen_rtl_values
        try:
            rtl_values = self.prepare_codegen_rtl_values(None)
            # Convert from FINN format (${var}$: [value]) to simple dict (var: value)
            operation_vars = {
                k.replace('$', ''): v[0] if isinstance(v, list) and v else v
                for k, v in rtl_values.items()
            }
            variables.update(operation_vars)
        except NotImplementedError:
            # If subclass doesn't implement prepare_codegen_rtl_values, no operation-specific vars
            pass
            
        return variables
    
    def execute_node(self, context, graph):
        """
        Execute node using standard dual execution mode pattern.
        
        This implements the common pattern used by 6/8 RTLBackend implementations:
        - cppsim mode: Delegate to base kernel class
        - rtlsim mode: Delegate to RTLBackend
        
        Subclasses can override for custom execution handling.
        """
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            # Get the base kernel class (first in MRO after AutoRTLBackend)
            base_classes = [cls for cls in self.__class__.__mro__ 
                          if cls != AutoRTLBackend and cls != RTLBackend 
                          and hasattr(cls, 'execute_node')]
            if base_classes:
                base_classes[0].execute_node(self, context, graph)
            else:
                raise RuntimeError(f"No base kernel class found for {self.__class__.__name__}")
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise Exception(
                f"Invalid exec_mode '{mode}'. Must be 'cppsim' or 'rtlsim'"
            )
    
    def generate_hdl(self, model, fpgapart, clk):
        """
        Generate HDL using template processing and file management.
        
        This implements the standard pattern:
        1. Get template variables
        2. Process template file with variable substitution
        3. Copy supporting RTL files
        4. Set node attributes for downstream tools
        """
        # Get paths
        finn_root = os.environ.get("FINN_ROOT")
        if not finn_root:
            raise RuntimeError("FINN_ROOT environment variable not set")
        
        rtllib_src = os.path.join(finn_root, "finn-rtllib", self.finn_rtllib_module, "hdl")
        template_path = os.path.join(rtllib_src, f"{self.finn_rtllib_module}_template.v")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        
        # Save top module name for reference after node renaming
        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)
        
        # Get template variables using FINN's standard method
        code_gen_dict = self.prepare_codegen_rtl_values(model)
        
        # Process template if it exists
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                template_content = f.read()
            
            # Apply template variable substitution (FINN format)
            for placeholder, values in code_gen_dict.items():
                # values should be a list with one element in FINN format
                value = values[0] if isinstance(values, list) and values else str(values)
                template_content = template_content.replace(placeholder, value)
            
            # Write generated file
            output_path = os.path.join(code_gen_dir, f"{topname}.v")
            with open(output_path, "w") as f:
                f.write(template_content)
        
        # Copy supporting RTL files
        self.copy_rtl_files(rtllib_src, code_gen_dir)
        
        # Set paths for downstream tools
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)
    
    def copy_rtl_files(self, rtllib_src: str, code_gen_dir: str):
        """
        Copy supporting RTL files from finn-rtllib to code generation directory.
        
        Subclasses can override to customize which files are copied.
        Default implementation copies all .sv and .v files except templates.
        """
        if not os.path.exists(rtllib_src):
            return
        
        for file_path in Path(rtllib_src).glob("*.sv"):
            if "template" not in file_path.name:
                shutil.copy(str(file_path), code_gen_dir)
        
        for file_path in Path(rtllib_src).glob("*.v"):
            if "template" not in file_path.name:
                shutil.copy(str(file_path), code_gen_dir)
    
    def get_rtl_file_list(self, abspath=False):
        """
        Get list of RTL files for this operation.
        
        Returns both generated wrapper and supporting RTL files.
        Implements standard pattern used across RTLBackend implementations.
        """
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            finn_root = os.environ.get("FINN_ROOT", "")
            rtllib_dir = os.path.join(finn_root, "finn-rtllib", self.finn_rtllib_module, "hdl") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""
        
        files = []
        
        # Add generated wrapper
        gen_top_module = self.get_nodeattr("gen_top_module")
        if gen_top_module:
            files.append(f"{code_gen_dir}{gen_top_module}.v")
        
        # Add supporting RTL files (subclasses can override get_supporting_rtl_files)
        supporting_files = self.get_supporting_rtl_files()
        for file_name in supporting_files:
            files.append(f"{rtllib_dir}{file_name}")
        
        return files
    
    def get_supporting_rtl_files(self) -> List[str]:
        """
        Get list of supporting RTL files to include.
        
        Subclasses should override this to specify their required files.
        Default implementation returns empty list.
        """
        return []
    
    def code_generation_ipi(self):
        """
        Generate TCL commands for Vivado IPI integration.
        
        Implements standard pattern for adding files and creating BD cells.
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        
        # Get all source files
        source_files = []
        
        # Add generated wrapper
        gen_top_module = self.get_nodeattr("gen_top_module")
        if gen_top_module:
            source_files.append(f"{gen_top_module}.v")
        
        # Add supporting files
        source_files.extend(self.get_supporting_rtl_files())
        
        # Convert to absolute paths
        source_files = [os.path.join(code_gen_dir, f) for f in source_files]
        
        # Generate TCL commands
        cmd = []
        for f in source_files:
            cmd.append(f"add_files -norecurse {f}")
        
        cmd.append(
            f"create_bd_cell -type module -reference {gen_top_module} {self.onnx_node.name}"
        )
        
        return cmd
    
    # Resource Estimation Methods
    
    def lut_estimation(self) -> int:
        """
        Estimate LUT usage using standard formula.
        
        Subclasses can override for operation-specific estimation.
        Default provides conservative estimate based on stream widths.
        """
        try:
            # Simple heuristic: LUTs proportional to input/output width
            input_width = self.get_instream_width() if hasattr(self, 'get_instream_width') else 32
            output_width = self.get_outstream_width() if hasattr(self, 'get_outstream_width') else 32
            
            # Conservative estimate: ~10 LUTs per bit of total width
            total_width = input_width + output_width
            return max(100, total_width * 10)
        except:
            return 100  # Fallback conservative estimate
    
    def bram_estimation(self) -> int:
        """
        Estimate BRAM usage using standard formula.
        
        Subclasses can override for operation-specific estimation.
        Default returns 0 for operations without memory.
        """
        # Most operations don't use BRAM unless they have weights or buffers
        return 0
    
    def dsp_estimation(self, fpgapart) -> int:
        """
        Estimate DSP usage using standard formula.
        
        Subclasses can override for operation-specific estimation.
        Default returns 0 for operations without arithmetic.
        """
        # Most operations don't use DSPs unless they have multiplication
        return 0
    
    def uram_estimation(self) -> int:
        """
        Estimate URAM usage using standard formula.
        
        Subclasses can override for operation-specific estimation.
        Default returns 0 for operations without large memories.
        """
        # Most operations don't use URAM
        return 0
    
    # Utility Methods
    
    def get_verilog_top_module_name(self) -> str:
        """
        Get Verilog top module name for this operation.
        
        Uses FINN's standard naming convention.
        """
        return f"{self.onnx_node.name}_{self.__class__.__name__}"
    
    def get_template_path(self) -> str:
        """Get path to template file for this operation."""
        finn_root = os.environ.get("FINN_ROOT", "")
        return os.path.join(
            finn_root, "finn-rtllib", self.finn_rtllib_module, 
            "hdl", f"{self.finn_rtllib_module}_template.v"
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(node={self.onnx_node.name}, module={self.finn_rtllib_module})"


# Mixin classes for advanced features

class ImplementationStyleMixin:
    """
    Mixin for operations that support multiple implementation styles.
    
    Used by operations like StreamingFIFO_rtl and ConvolutionInputGenerator_rtl.
    """
    
    def get_nodeattr_types(self):
        """Add implementation style selection attribute."""
        attrs = super().get_nodeattr_types()
        attrs["impl_style"] = ("s", False, "rtl", {"rtl", "vivado"})
        return attrs
    
    def get_implementation_style(self) -> str:
        """Get current implementation style."""
        return self.get_nodeattr("impl_style")
    
    def select_impl_style(self):
        """
        Select optimal implementation style based on operation parameters.
        
        Subclasses should override with operation-specific logic.
        """
        return self.get_nodeattr("impl_style")


class AdvancedMemoryMixin:
    """
    Mixin for operations with advanced memory management.
    
    Used by operations like Thresholding_rtl with complex memory requirements.
    """
    
    def get_nodeattr_types(self):
        """Add memory configuration attributes."""
        attrs = super().get_nodeattr_types()
        attrs.update({
            "depth_trigger_uram": ("i", False, 0),
            "depth_trigger_bram": ("i", False, 0),
            "uniform_thres": ("i", False, 0, {0, 1}),
            "deep_pipeline": ("i", False, 1, {0, 1}),
        })
        return attrs
    
    def get_memory_primitive_alternatives(self):
        """Get memory primitive alternatives for this operation."""
        # Subclasses implement operation-specific memory primitive selection
        return get_memutil_alternatives()


class DynamicConfigMixin:
    """
    Mixin for operations with dynamic configuration support.
    
    Used by operations like ConvolutionInputGenerator_rtl and FMPadding_rtl.
    """
    
    def get_nodeattr_types(self):
        """Add dynamic configuration attributes."""
        attrs = super().get_nodeattr_types()
        attrs["runtime_writeable_weights"] = ("i", False, 0, {0, 1})
        return attrs
    
    def get_dynamic_config(self):
        """
        Get dynamic configuration data for runtime weight updates.
        
        Subclasses should override with operation-specific logic.
        """
        return {}