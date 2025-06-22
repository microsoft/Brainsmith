############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Unified RTL Backend

Clean RTL backend implementation using the new framework's code generation
capabilities.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from abc import abstractmethod

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from jinja2 import Environment, FileSystemLoader, Template

from .kernel_definition import KernelDefinition


class UnifiedRTLBackend(RTLBackend):
    """
    Clean RTL backend implementation using new framework's code generation.
    
    This class should be used as a mixin with UnifiedHWCustomOp to provide
    RTL generation capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize RTL backend."""
        super().__init__(*args, **kwargs)
        self.template_engine = None
        self._template_cache = {}
    
    @property
    @abstractmethod
    def finn_rtllib_module(self) -> str:
        """Return the finn-rtllib module name for this operation."""
        pass
    
    def prepare_codegen_rtl_values(self, model) -> Dict[str, List[str]]:
        """
        Prepare template variables for RTL code generation.
        
        Uses the kernel definition and current configuration to generate
        all necessary template variables.
        """
        if not hasattr(self, 'kernel_def'):
            raise RuntimeError("UnifiedRTLBackend requires kernel_def attribute")
        
        rtl_values = {}
        
        # Get current configuration
        if hasattr(self, '_current_config') and self._current_config:
            kernel_config = self._current_config.kernel_configs[self.kernel.name]
            
            # Add interface parallelism values
            for intf in self.kernel_def.interfaces:
                parallelism = kernel_config.interface_parallelism.get(intf.name, 1)
                stream_shape = kernel_config.stream_shapes.get(intf.name, [1])
                
                # Standard naming for template compatibility
                rtl_values[f"${intf.name.upper()}_PARALLELISM$"] = [str(parallelism)]
                rtl_values[f"${intf.name.upper()}_WIDTH$"] = [str(stream_shape[0])]
                
                # Legacy naming for FINN compatibility
                if intf.type.value == "INPUT":
                    rtl_values["$SIMD$"] = [str(parallelism)]
                elif intf.type.value == "WEIGHT":
                    rtl_values["$PE$"] = [str(parallelism)]
        
        # Add exposed parameters
        for param in self.kernel_def.exposed_parameters:
            value = self.get_nodeattr(param) if hasattr(self, 'get_nodeattr') else \
                    self.kernel_def.parameter_defaults.get(param, 0)
            rtl_values[f"${param.upper()}$"] = [str(value)]
        
        # Add datatype information
        for intf in self.kernel_def.interfaces:
            dt_attr = f"{intf.name}_datatype"
            if hasattr(self, 'get_nodeattr') and hasattr(self, 'has_nodeattr'):
                if self.has_nodeattr(dt_attr):
                    dt_str = self.get_nodeattr(dt_attr)
                    rtl_values[f"${intf.name.upper()}_DATATYPE$"] = [dt_str]
        
        # Add any additional RTL parameters
        for param_name, param_value in self.kernel_def.rtl_parameters.items():
            if param_name not in self.kernel_def.exposed_parameters:
                # Evaluate parameter value if it's a formula
                if isinstance(param_value, str) and any(op in param_value for op in ['+', '-', '*', '/', '//']):
                    # Simple evaluation - in real implementation would be more robust
                    try:
                        value = eval(param_value, {"__builtins__": {}}, rtl_values)
                    except:
                        value = param_value
                else:
                    value = param_value
                rtl_values[f"${param_name.upper()}$"] = [str(value)]
        
        return rtl_values
    
    def generate_hdl(self, model, fpgapart, clk):
        """
        Generate HDL using template processing and file management.
        
        This method generates all necessary RTL files for the operator.
        """
        # Get paths
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        
        # Ensure output directory exists
        os.makedirs(code_gen_dir, exist_ok=True)
        
        # Generate RTL files
        rtl_files = self.generate_rtl_files(model, fpgapart, clk)
        
        # Write generated files
        for filename, content in rtl_files.items():
            filepath = os.path.join(code_gen_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        # Copy support files if any
        self.copy_support_files(code_gen_dir)
        
        # Set node attributes for downstream tools
        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)
    
    def generate_rtl_files(self, model, fpgapart, clk) -> Dict[str, str]:
        """
        Generate RTL files for the operator.
        
        Returns dictionary mapping filename to content.
        """
        files = {}
        
        # Generate main module
        main_module = self.generate_main_module(model)
        topname = self.get_verilog_top_module_name()
        files[f"{topname}.v"] = main_module
        
        # Generate wrapper if needed
        if self.needs_wrapper():
            wrapper = self.generate_wrapper(model)
            files[f"{topname}_wrapper.v"] = wrapper
        
        # Generate configuration package if parameters exposed
        if hasattr(self, 'kernel_def') and self.kernel_def.exposed_parameters:
            config_pkg = self.generate_config_package(model)
            files[f"{topname}_config.sv"] = config_pkg
        
        return files
    
    def generate_main_module(self, model) -> str:
        """Generate main RTL module using templates."""
        # Get template variables
        template_vars = self.prepare_template_variables(model)
        
        # Get template
        template = self.get_template("main_module.v.j2")
        if not template:
            # Fallback to FINN template processing
            return self.generate_from_finn_template(model)
        
        # Render template
        return template.render(**template_vars)
    
    def generate_wrapper(self, model) -> str:
        """Generate wrapper module if needed."""
        template_vars = self.prepare_template_variables(model)
        template = self.get_template("wrapper.v.j2")
        
        if not template:
            return ""  # No wrapper needed
        
        return template.render(**template_vars)
    
    def generate_config_package(self, model) -> str:
        """Generate SystemVerilog package with configuration parameters."""
        template_vars = self.prepare_template_variables(model)
        template = self.get_template("config_package.sv.j2")
        
        if not template:
            # Generate simple package
            return self.generate_simple_config_package(template_vars)
        
        return template.render(**template_vars)
    
    def generate_simple_config_package(self, template_vars: Dict[str, Any]) -> str:
        """Generate simple configuration package without template."""
        lines = [
            f"package {template_vars['module_name']}_config;",
            ""
        ]
        
        # Add parameters
        if 'parameters' in template_vars:
            for param_name, param_value in template_vars['parameters'].items():
                lines.append(f"  parameter {param_name} = {param_value};")
        
        lines.extend(["", "endpackage"])
        return '\n'.join(lines)
    
    def generate_from_finn_template(self, model) -> str:
        """Fallback to FINN template processing."""
        finn_root = os.environ.get("FINN_ROOT", "")
        if not finn_root:
            raise RuntimeError("FINN_ROOT not set")
        
        template_path = os.path.join(
            finn_root, "finn-rtllib", self.finn_rtllib_module,
            "hdl", f"{self.finn_rtllib_module}_template.v"
        )
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        # Read template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Get RTL values in FINN format
        rtl_values = self.prepare_codegen_rtl_values(model)
        
        # Apply substitutions
        for placeholder, values in rtl_values.items():
            value = values[0] if isinstance(values, list) and values else str(values)
            template_content = template_content.replace(placeholder, value)
        
        return template_content
    
    def prepare_template_variables(self, model) -> Dict[str, Any]:
        """Prepare all template variables for rendering."""
        variables = {
            'module_name': self.get_verilog_top_module_name(),
            'kernel_name': self.kernel_def.name if hasattr(self, 'kernel_def') else "unknown",
            'interfaces': [],
            'parameters': {},
            'config': {}
        }
        
        # Add interface information
        if hasattr(self, 'kernel_def'):
            for intf in self.kernel_def.interfaces:
                intf_info = {
                    'name': intf.name,
                    'type': intf.type.value,
                    'protocol': intf.protocol.value,
                    'direction': 'input' if intf.type.value in ['INPUT', 'WEIGHT'] else 'output'
                }
                
                # Add width information if available
                if hasattr(self, '_current_config') and self._current_config:
                    kernel_config = self._current_config.kernel_configs.get(self.kernel.name, None)
                    if kernel_config:
                        stream_shape = kernel_config.stream_shapes.get(intf.name, [1])
                        intf_info['width'] = stream_shape[0]
                
                variables['interfaces'].append(intf_info)
        
        # Add parameters
        rtl_values = self.prepare_codegen_rtl_values(model)
        for key, values in rtl_values.items():
            # Clean up parameter name
            param_name = key.replace('$', '').lower()
            param_value = values[0] if isinstance(values, list) and values else str(values)
            variables['parameters'][param_name] = param_value
        
        # Add configuration
        if hasattr(self, '_current_config') and self._current_config:
            variables['config'] = {
                'optimized': getattr(self, '_optimized', False),
                'configuration': self._current_config.to_dict() if hasattr(self._current_config, 'to_dict') else {}
            }
        
        return variables
    
    def get_template(self, template_name: str) -> Optional[Template]:
        """Get Jinja2 template by name."""
        # Check cache
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # Initialize template engine if needed
        if not self.template_engine:
            self._initialize_template_engine()
        
        try:
            template = self.template_engine.get_template(template_name)
            self._template_cache[template_name] = template
            return template
        except:
            return None
    
    def _initialize_template_engine(self):
        """Initialize Jinja2 template engine."""
        # Look for templates in multiple locations
        template_dirs = []
        
        # Project-specific templates
        unified_templates = Path(__file__).parent.parent / "templates"
        if unified_templates.exists():
            template_dirs.append(str(unified_templates))
        
        # Operator-specific templates
        if hasattr(self, 'kernel_def'):
            op_templates = Path(__file__).parent.parent / "operators" / self.kernel_def.name / "templates"
            if op_templates.exists():
                template_dirs.append(str(op_templates))
        
        # FINN templates
        finn_root = os.environ.get("FINN_ROOT", "")
        if finn_root and hasattr(self, 'finn_rtllib_module'):
            finn_templates = Path(finn_root) / "finn-rtllib" / self.finn_rtllib_module / "templates"
            if finn_templates.exists():
                template_dirs.append(str(finn_templates))
        
        self.template_engine = Environment(
            loader=FileSystemLoader(template_dirs),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def needs_wrapper(self) -> bool:
        """Determine if wrapper generation is needed."""
        # Override in subclasses for specific logic
        return False
    
    def copy_support_files(self, output_dir: str):
        """Copy any required support files."""
        if not hasattr(self, 'kernel_def'):
            return
        
        # Copy files listed in kernel definition
        for support_file in self.kernel_def.support_files:
            # Look for file in various locations
            for search_dir in self._get_support_file_search_dirs():
                source_path = Path(search_dir) / support_file
                if source_path.exists():
                    dest_path = Path(output_dir) / support_file
                    shutil.copy2(source_path, dest_path)
                    break
    
    def _get_support_file_search_dirs(self) -> List[str]:
        """Get directories to search for support files."""
        dirs = []
        
        # Operator-specific directory
        if hasattr(self, 'kernel_def'):
            op_dir = Path(__file__).parent.parent / "operators" / self.kernel_def.name / "rtl"
            if op_dir.exists():
                dirs.append(str(op_dir))
        
        # FINN RTL directory
        finn_root = os.environ.get("FINN_ROOT", "")
        if finn_root and hasattr(self, 'finn_rtllib_module'):
            finn_dir = Path(finn_root) / "finn-rtllib" / self.finn_rtllib_module / "hdl"
            if finn_dir.exists():
                dirs.append(str(finn_dir))
        
        return dirs
    
    def get_rtl_file_list(self, abspath=False) -> List[str]:
        """Get list of all RTL files for this operation."""
        files = []
        
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") if abspath else ""
        if code_gen_dir and not code_gen_dir.endswith("/"):
            code_gen_dir += "/"
        
        # Add generated files
        gen_top = self.get_nodeattr("gen_top_module")
        if gen_top:
            files.append(f"{code_gen_dir}{gen_top}.v")
            
            # Add wrapper if it exists
            wrapper_path = f"{code_gen_dir}{gen_top}_wrapper.v"
            if os.path.exists(wrapper_path):
                files.append(wrapper_path)
            
            # Add config package if it exists
            config_path = f"{code_gen_dir}{gen_top}_config.sv"
            if os.path.exists(config_path):
                files.append(config_path)
        
        # Add support files
        if hasattr(self, 'kernel_def'):
            for support_file in self.kernel_def.support_files:
                files.append(f"{code_gen_dir}{support_file}")
        
        return files