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
    - KernelModel access for interface properties
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize AutoRTLBackend with standard RTL backend setup."""
        # For multiple inheritance, pass through all arguments to the next class in MRO
        # This allows proper cooperative inheritance
        super().__init__(*args, **kwargs)
    
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
    
    @property
    def kernel_model(self):
        """Access KernelModel from our AutoHWCustomOp inheritance.
        
        Since AutoRTLBackend is always used with multiple inheritance
        alongside AutoHWCustomOp, we can directly access _kernel_model.
        
        Raises:
            RuntimeError: If KernelModel is not initialized
        """
        if hasattr(self, '_kernel_model') and self._kernel_model is not None:
            return self._kernel_model
        raise RuntimeError(
            f"{self.__class__.__name__} requires initialized KernelModel. "
            "Ensure node has been added to model and shapes have been inferred."
        )
    
    def _get_interface_bdim(self, interface_name: str, dimension_index: int = 0) -> int:
        """Get block dimension for interface from KernelModel."""
        try:
            # Find interface in inputs or outputs
            for inp in self.kernel_model.input_models:
                if inp.definition.name == interface_name:
                    if dimension_index < len(inp.block_dims):
                        return inp.block_dims[dimension_index]
                    return 1
            for out in self.kernel_model.output_models:
                if out.definition.name == interface_name:
                    if dimension_index < len(out.block_dims):
                        return out.block_dims[dimension_index]
                    return 1
        except RuntimeError:
            # KernelModel not available - use fallback
            pass
        
        # Fallback to node attribute if KernelModel not available
        # This maintains backward compatibility
        param_name = f"{interface_name}_BDIM" if interface_name else "BDIM"
        return self.get_nodeattr(param_name, 1)
    
    def _get_interface_sdim(self, interface_name: str, dimension_index: int = 0) -> int:
        """Get stream dimension for interface from KernelModel."""
        try:
            # Find interface in inputs (only inputs have SDIM)
            for inp in self.kernel_model.input_models:
                if inp.definition.name == interface_name:
                    if dimension_index < len(inp.sdim):
                        return inp.sdim[dimension_index]
                    return 1
        except RuntimeError:
            # KernelModel not available - use fallback
            pass
        
        # Fallback to node attribute if KernelModel not available
        param_name = f"{interface_name}_SDIM" if interface_name else "SDIM"
        return self.get_nodeattr(param_name, 1)
    
    def _get_interface_width(self, interface_name: str) -> int:
        """Get datatype width for interface using parent class methods."""
        # Use parent class's _get_interface_model if available
        if hasattr(self, '_get_interface_model'):
            interface = self._get_interface_model(interface_name)
            if interface and hasattr(interface, 'datatype'):
                return interface.datatype.bitwidth()
        
        # Fallback to node attribute
        from qonnx.core.datatype import DataType
        dtype_attr = f"{interface_name}DataType"
        dtype_str = self.get_nodeattr(dtype_attr, "INT8")
        return DataType[dtype_str].bitwidth()
    
    def _get_interface_signed(self, interface_name: str) -> bool:
        """Get datatype signed property for interface using parent class methods."""
        # Use parent class's _get_interface_model if available
        if hasattr(self, '_get_interface_model'):
            interface = self._get_interface_model(interface_name)
            if interface and hasattr(interface, 'datatype'):
                return interface.datatype.signed()
        
        # Fallback to node attribute
        from qonnx.core.datatype import DataType
        dtype_attr = f"{interface_name}DataType"
        dtype_str = self.get_nodeattr(dtype_attr, "INT8")
        return DataType[dtype_str].signed()
    
    def execute_node(self, context, graph):
        """
        Execute node using standard dual execution mode pattern.
        
        This implements the common pattern used by RTLBackend implementations:
        - cppsim mode: Delegate to AutoHWCustomOp's execute_node
        - rtlsim mode: Delegate to RTLBackend's execute_node
        
        Subclasses can override for custom execution handling.
        """
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            # Import here to avoid circular dependency
            from brainsmith.core.finn.auto_hw_custom_op import AutoHWCustomOp
            # Call AutoHWCustomOp's execute_node directly
            AutoHWCustomOp.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise ValueError(
                f"Invalid exec_mode '{mode}'. Must be 'cppsim' or 'rtlsim'"
            )
    
    def generate_hdl(self, model, fpgapart, clk):
        """
        Generate HDL for the operation.
        
        Brainsmith kernels must override this method to implement their own
        HDL generation logic. Unlike finn-rtllib based nodes, Brainsmith kernels
        are self-contained and don't rely on external RTL templates.
        
        Args:
            model: ONNX model
            fpgapart: Target FPGA part
            clk: Target clock period
        """
        # Ensure we have KernelModel before proceeding
        if hasattr(self, '_ensure_kernel_model'):
            self._ensure_kernel_model()
        
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_hdl() method. "
            "Brainsmith kernels are self-contained and must provide their own "
            "HDL generation logic rather than relying on finn-rtllib templates."
        )
    
    def copy_rtl_files(self, rtllib_src: str, code_gen_dir: str):
        """
        Copy supporting RTL files from finn-rtllib to code generation directory.
        
        Subclasses can override to customize which files are copied.
        Default implementation copies all .sv and .v files except templates.
        """
        if not os.path.exists(rtllib_src):
            return
        
        # Copy all SystemVerilog files (including packages)
        for file_path in Path(rtllib_src).glob("*.sv"):
            if "template" not in file_path.name:
                shutil.copy(str(file_path), code_gen_dir)
        
        # Copy all Verilog files
        for file_path in Path(rtllib_src).glob("*.v"):
            if "template" not in file_path.name:
                shutil.copy(str(file_path), code_gen_dir)
        
        # Copy any Verilog header files
        for file_path in Path(rtllib_src).glob("*.vh"):
            shutil.copy(str(file_path), code_gen_dir)
    
    def copy_included_rtl_files(self, included_files: List[str], code_gen_dir: str) -> None:
        """
        Copy included RTL files to code generation directory.
        
        Handles path resolution with precedence:
        1. Absolute paths
        2. Relative to source file directory
        3. Relative to current directory
        
        Args:
            included_files: List of RTL file paths (source file should be first)
            code_gen_dir: Target directory for copied files
        """
        if not included_files:
            return
        
        # Get source directory from first file (which should be the main source)
        source_dir = Path(included_files[0]).parent if included_files else Path.cwd()
        
        for rtl_file in included_files:
            rtl_path = Path(rtl_file)
            
            # Try absolute path first
            if rtl_path.is_absolute() and rtl_path.exists():
                shutil.copy(str(rtl_path), code_gen_dir)
            # Try relative to source file directory
            elif (source_dir / rtl_file).exists():
                shutil.copy(str(source_dir / rtl_file), code_gen_dir)
            # Try relative to current directory
            elif Path(rtl_file).exists():
                shutil.copy(str(Path(rtl_file)), code_gen_dir)
            else:
                # Silently skip missing files - validation should happen elsewhere
                pass
    
    def get_rtl_file_list(self, abspath=False):
        """
        Get list of RTL files for this operation.
        
        Returns generated wrapper and any additional RTL files.
        Subclasses can override to add more files.
        """
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
        else:
            code_gen_dir = ""
        
        files = []
        
        # Add generated wrapper
        gen_top_module = self.get_nodeattr("gen_top_module")
        if gen_top_module:
            files.append(f"{code_gen_dir}{gen_top_module}.v")
        
        return files
    
    @abstractmethod
    def get_included_rtl_filenames(self) -> List[str]:
        """
        Get list of included RTL file names (basename only).
        
        This retrieves the included RTL files from the kernel metadata
        and returns just the filenames for use in file lists.
        
        Subclasses MUST implement this method to provide the list of RTL files
        that need to be included for IP generation.
        
        Returns:
            List of RTL file basenames
        """
        pass
    
    def code_generation_ipi(self, behavioral=False):
        """
        Generate TCL commands for Vivado IPI integration.
        
        Implements standard pattern for adding RTL files and creating BD cells.
        This standardized implementation:
        1. Collects the generated wrapper file
        2. Gets kernel-specific RTL files via get_included_rtl_filenames()
        3. Removes duplicates while preserving order
        4. Generates TCL commands to add files and create the module instance
        
        Subclasses can override this method if they need custom IPI generation.
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        
        # List all source files
        source_files = []
        
        # Add generated wrapper
        gen_top_module = self.get_nodeattr("gen_top_module")
        if gen_top_module:
            source_files.append(f"{gen_top_module}.v")
        
        # Add included RTL files
        source_files.extend(self.get_included_rtl_filenames())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in source_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        
        # Convert to absolute paths and filter existing files
        cmd = []
        for f in unique_files:
            full_path = os.path.join(code_gen_dir, f)
            if os.path.exists(full_path):
                cmd.append(f"add_files -norecurse {full_path}")
        
        # Create module instance
        if gen_top_module:
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
    
    @abstractmethod
    def get_verilog_top_module_intf_names(self) -> Dict[str, Any]:
        """
        Get interface names for the Verilog top module.
        
        Returns a dictionary mapping interface types to their signal names.
        This is used for IP integration and proper port connections.
        
        Subclasses MUST implement this method to provide actual signal names
        from the RTL module, rather than relying on hardcoded heuristics.
        
        Expected return format:
        {
            "clk": ["clk_name"],
            "rst": ["rst_name"],
            "s_axis": [["input_tdata", "input_tvalid", "input_tready"], ...],
            "m_axis": [["output_tdata", "output_tvalid", "output_tready"], ...],
            "axilite": ["config_interface_name"]  # Optional for AXI-Lite
        }
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_verilog_top_module_intf_names() "
            "to provide actual RTL signal names"
        )
    
    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """
        Generate weight initialization file for this kernel.
        
        Base implementation that can be overridden by specific backends.
        This is a placeholder that subclasses should override.
        
        Args:
            weights: numpy array with weight values
            weight_file_mode: 'decoupled' or 'const' 
            weight_file_name: path where weight file should be written
        """
        raise NotImplementedError(
            f"make_weight_file() not implemented for {self.__class__.__name__}. "
            "Please implement this method if your kernel uses weights."
        )
    
    def generate_init_files(self, model, code_gen_dir):
        """
        Generate initialization files (weights, thresholds, etc.) for the kernel.
        
        This is called during HDL generation and should create any .dat or .mem
        files needed by the RTL for initialization.
        
        Args:
            model: ONNX model containing initializers
            code_gen_dir: Directory where files should be written
        """
        # Default implementation does nothing
        # Subclasses override this for kernels that need init files
        pass
    
    def get_all_meminit_filenames(self, abspath=False):
        """
        Return a list of all memory initializer files used for this node.
        
        This is used by FINN for tracking generated files.
        
        Args:
            abspath: If True, return absolute paths; if False, relative to code_gen_dir
            
        Returns:
            List of file paths
        """
        # Default implementation returns empty list
        # Subclasses override this for kernels with memory init files
        return []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(node={self.onnx_node.name})"


