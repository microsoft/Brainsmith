"""
Direct Template Renderer for Enhanced RTL Parsing Results.

This module implements direct template rendering using EnhancedRTLParsingResult,
eliminating the need for DataflowModel conversion for template generation.

Key Features:
- Direct template context from RTL parsing results
- Template rendering without DataflowModel overhead
- Compatible with existing Jinja2 templates
- Cached template context for performance
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import jinja2

from ..rtl_parser.data import EnhancedRTLParsingResult

logger = logging.getLogger(__name__)


class DirectTemplateRenderingError(Exception):
    """Exception raised during direct template rendering."""
    pass


class DirectTemplateRenderer:
    """
    Direct template renderer using Enhanced RTL Parsing Results.
    
    This class renders templates directly from RTL parsing results without
    requiring DataflowModel conversion, significantly improving performance
    and simplifying the template generation pipeline.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize direct template renderer.
        
        Args:
            template_dir: Optional custom template directory
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.jinja_env = self._setup_jinja_environment()
        
    def _get_default_template_dir(self) -> Path:
        """Get default template directory."""
        return Path(__file__).parent
    
    def _setup_jinja_environment(self) -> jinja2.Environment:
        """
        Setup Jinja2 environment for direct template rendering.
        
        Returns:
            jinja2.Environment: Configured Jinja2 environment
        """
        loaders = [jinja2.FileSystemLoader(self.template_dir)]
        
        # Add fallback to unified HWKG templates
        unified_template_dir = Path(__file__).parent.parent.parent / "unified_hwkg" / "templates"
        if unified_template_dir.exists():
            loaders.append(jinja2.FileSystemLoader(unified_template_dir))
        
        loader = jinja2.ChoiceLoader(loaders)
        
        env = jinja2.Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined
        )
        
        # Add custom filters for enhanced RTL result processing
        env.filters['format_interface_list'] = self._format_interface_list
        env.filters['format_datatype_constraint'] = self._format_datatype_constraint
        env.filters['format_rtl_parameters'] = self._format_rtl_parameters
        
        return env
    
    def render_hwcustomop(self, enhanced_result: EnhancedRTLParsingResult, 
                          output_dir: Path, template_name: str = "hw_custom_op_slim.py.j2",
                          compiler_data: Optional[Dict[str, Any]] = None) -> Path:
        """
        Render HWCustomOp file directly from enhanced RTL result.
        
        Args:
            enhanced_result: Enhanced RTL parsing result
            output_dir: Output directory for generated file
            template_name: Template file name to use
            
        Returns:
            Path to generated HWCustomOp file
            
        Raises:
            DirectTemplateRenderingError: If rendering fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get template context directly from enhanced result
            context = enhanced_result.get_template_context()
            
            # Add HWCustomOp-specific context
            context.update({
                'import_paths': {
                    'auto_hw_custom_op': 'brainsmith.dataflow.core.auto_hw_custom_op',
                    'interface_metadata': 'brainsmith.dataflow.rtl_integration',
                    'dataflow_interface': 'brainsmith.dataflow.core.dataflow_interface'
                },
                'template_type': 'hwcustomop'
            })
            
            # Add compiler_data if provided
            if compiler_data:
                context["compiler_data"] = compiler_data
            
            # Render template
            rendered_content = self._render_template(template_name, context)
            
            # Write generated file
            output_file = output_dir / f"{enhanced_result.name}_hwcustomop.py"
            with open(output_file, 'w') as f:
                f.write(rendered_content)
            
            logger.info(f"Generated HWCustomOp file: {output_file}")
            return output_file
            
        except Exception as e:
            error_msg = f"Failed to render HWCustomOp for {enhanced_result.name}: {e}"
            logger.error(error_msg)
            raise DirectTemplateRenderingError(error_msg) from e
    
    def render_rtlbackend(self, enhanced_result: EnhancedRTLParsingResult, 
                          output_dir: Path, template_name: str = "rtl_backend.py.j2",
                          compiler_data: Optional[Dict[str, Any]] = None) -> Path:
        """
        Render RTLBackend file directly from enhanced RTL result.
        
        Args:
            enhanced_result: Enhanced RTL parsing result
            output_dir: Output directory for generated file
            template_name: Template file name to use
            
        Returns:
            Path to generated RTLBackend file
            
        Raises:
            DirectTemplateRenderingError: If rendering fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get template context directly from enhanced result
            context = enhanced_result.get_template_context()
            
            # Add RTLBackend-specific context
            context.update({
                'import_paths': {
                    'auto_rtl_backend': 'brainsmith.dataflow.core.auto_rtl_backend',
                    'dataflow_interface': 'brainsmith.dataflow.core.dataflow_interface'
                },
                'template_type': 'rtlbackend'
            })
            
            # Add compiler_data if provided
            if compiler_data:
                context["compiler_data"] = compiler_data
            
            # Render template
            rendered_content = self._render_template(template_name, context)
            
            # Write generated file
            output_file = output_dir / f"{enhanced_result.name}_rtlbackend.py"
            with open(output_file, 'w') as f:
                f.write(rendered_content)
            
            logger.info(f"Generated RTLBackend file: {output_file}")
            return output_file
            
        except Exception as e:
            error_msg = f"Failed to render RTLBackend for {enhanced_result.name}: {e}"
            logger.error(error_msg)
            raise DirectTemplateRenderingError(error_msg) from e
    
    def render_test_suite(self, enhanced_result: EnhancedRTLParsingResult, 
                          output_dir: Path, template_name: str = "test_suite.py.j2",
                          compiler_data: Optional[Dict[str, Any]] = None) -> Path:
        """
        Render test suite file directly from enhanced RTL result.
        
        Args:
            enhanced_result: Enhanced RTL parsing result
            output_dir: Output directory for generated file
            template_name: Template file name to use
            
        Returns:
            Path to generated test file
            
        Raises:
            DirectTemplateRenderingError: If rendering fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get template context directly from enhanced result
            context = enhanced_result.get_template_context()
            
            # Add test-specific context
            context.update({
                'test_class_name': f"Test{context['class_name']}",
                'hwcustomop_class': f"{context['class_name']}HWCustomOp",
                'rtlbackend_class': f"{context['class_name']}RTLBackend",
                'test_scenarios': self._generate_test_scenarios(enhanced_result),
                'import_paths': {
                    'pytest': 'pytest',
                    'numpy': 'numpy',
                    'dataflow_model': 'brainsmith.dataflow.core.dataflow_model'
                },
                'template_type': 'test'
            })
            
            # Add compiler_data if provided
            if compiler_data:
                context["compiler_data"] = compiler_data
            
            # Render template
            rendered_content = self._render_template(template_name, context)
            
            # Write generated file
            output_file = output_dir / f"test_{enhanced_result.name}.py"
            with open(output_file, 'w') as f:
                f.write(rendered_content)
            
            logger.info(f"Generated test file: {output_file}")
            return output_file
            
        except Exception as e:
            error_msg = f"Failed to render test suite for {enhanced_result.name}: {e}"
            logger.error(error_msg)
            raise DirectTemplateRenderingError(error_msg) from e
    
    def render_rtl_wrapper(self, enhanced_result: EnhancedRTLParsingResult, 
                           output_dir: Path, template_name: str = "rtl_wrapper.v.j2") -> Path:
        """
        Render RTL wrapper file directly from enhanced RTL result.
        
        Args:
            enhanced_result: Enhanced RTL parsing result
            output_dir: Output directory for generated file
            template_name: Template file name to use
            
        Returns:
            Path to generated RTL wrapper file
            
        Raises:
            DirectTemplateRenderingError: If rendering fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get template context directly from enhanced result
            context = enhanced_result.get_template_context()
            
            # Add RTL wrapper-specific context
            context.update({
                'template_type': 'rtl_wrapper',
                'wrapper_module_name': f"{enhanced_result.name}_wrapper"
            })
            
            # Render template
            rendered_content = self._render_template(template_name, context)
            
            # Write generated file
            output_file = output_dir / f"{enhanced_result.name}_wrapper.v"
            with open(output_file, 'w') as f:
                f.write(rendered_content)
            
            logger.info(f"Generated RTL wrapper file: {output_file}")
            return output_file
            
        except Exception as e:
            error_msg = f"Failed to render RTL wrapper for {enhanced_result.name}: {e}"
            logger.error(error_msg)
            raise DirectTemplateRenderingError(error_msg) from e
    
    def render_documentation(self, enhanced_result: EnhancedRTLParsingResult, 
                            output_dir: Path, template_name: str = "documentation.md.j2") -> Path:
        """
        Render documentation file directly from enhanced RTL result.
        
        Args:
            enhanced_result: Enhanced RTL parsing result
            output_dir: Output directory for generated file
            template_name: Template file name to use
            
        Returns:
            Path to generated documentation file
            
        Raises:
            DirectTemplateRenderingError: If rendering fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get template context directly from enhanced result
            context = enhanced_result.get_template_context()
            
            # Add documentation-specific context
            context.update({
                'template_type': 'documentation',
                'documentation_title': f"{context['class_name']} Hardware Kernel Documentation"
            })
            
            # Render template
            rendered_content = self._render_template(template_name, context)
            
            # Write generated file
            output_file = output_dir / f"{enhanced_result.name}_README.md"
            with open(output_file, 'w') as f:
                f.write(rendered_content)
            
            logger.info(f"Generated documentation file: {output_file}")
            return output_file
            
        except Exception as e:
            error_msg = f"Failed to render documentation for {enhanced_result.name}: {e}"
            logger.error(error_msg)
            raise DirectTemplateRenderingError(error_msg) from e
    
    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Load and render template with context.
        
        Args:
            template_name: Name of template file
            context: Template context dictionary
            
        Returns:
            str: Rendered template content
            
        Raises:
            DirectTemplateRenderingError: If rendering fails
        """
        try:
            template = self.jinja_env.get_template(template_name)
            rendered = template.render(**context)
            logger.debug(f"Rendered template {template_name} ({len(rendered)} chars)")
            return rendered
        except jinja2.TemplateNotFound as e:
            raise DirectTemplateRenderingError(f"Template not found: {template_name}") from e
        except jinja2.TemplateSyntaxError as e:
            raise DirectTemplateRenderingError(f"Template syntax error in {template_name}: {e}") from e
        except jinja2.TemplateRuntimeError as e:
            raise DirectTemplateRenderingError(f"Template rendering error in {template_name}: {e}") from e
    
    def _generate_test_scenarios(self, enhanced_result: EnhancedRTLParsingResult) -> List[Dict[str, Any]]:
        """Generate test scenarios from enhanced RTL result."""
        scenarios = []
        
        # Basic functionality test
        scenarios.append({
            'name': 'test_basic_functionality',
            'description': 'Test basic HWCustomOp instantiation and method calls',
            'test_type': 'functionality'
        })
        
        # Interface configuration test
        if enhanced_result.interfaces:
            scenarios.append({
                'name': 'test_interface_configuration',
                'description': 'Test interface configuration and metadata',
                'test_type': 'configuration'
            })
        
        # Template context validation test
        scenarios.append({
            'name': 'test_template_context',
            'description': 'Test template context generation and validation',
            'test_type': 'validation'
        })
        
        # Performance estimation test
        if len(enhanced_result.interfaces) > 1:
            scenarios.append({
                'name': 'test_performance_estimation',
                'description': 'Test performance and resource estimation',
                'test_type': 'performance'
            })
        
        return scenarios
    
    # Custom Jinja2 filters for enhanced RTL result processing
    def _format_interface_list(self, interfaces: List[Dict[str, Any]]) -> str:
        """Format interface list for template rendering."""
        if not interfaces:
            return "[]"
        
        formatted_items = []
        for iface in interfaces:
            formatted_items.append(f"'{iface['name']}'")
        
        return "[" + ", ".join(formatted_items) + "]"
    
    def _format_datatype_constraint(self, constraint: Dict[str, Any]) -> str:
        """Format datatype constraint for template rendering."""
        return f"'{constraint.get('finn_type', 'UINT8')}'"
    
    def _format_rtl_parameters(self, parameters: List[Dict[str, Any]]) -> str:
        """Format RTL parameters for template rendering."""
        if not parameters:
            return "{}"
        
        formatted_items = []
        for param in parameters:
            name = param['name']
            value = param['default_value']
            formatted_items.append(f"'{name}': {value}")
        
        return "{" + ", ".join(formatted_items) + "}"


def create_direct_template_renderer(template_dir: Optional[Path] = None) -> DirectTemplateRenderer:
    """
    Factory function for creating DirectTemplateRenderer instances.
    
    Args:
        template_dir: Optional custom template directory
        
    Returns:
        DirectTemplateRenderer: Configured renderer instance
    """
    return DirectTemplateRenderer(template_dir)