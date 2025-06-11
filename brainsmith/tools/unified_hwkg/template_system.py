"""
Template System for Unified HWKG.

This module provides template loading, context building, and rendering
functionality for the unified HWKG system. It focuses on minimal instantiation
templates that use AutoHWCustomOp/AutoRTLBackend with DataflowModel rather
than complex template-generated implementation code.

Key Differences from Old Template System:
- Templates instantiate AutoHWCustomOp/AutoRTLBackend with DataflowModel
- Minimal template complexity - no implementation code generation
- DataflowModel-driven context building
- Mathematical foundation instead of placeholders
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

import jinja2

from ...dataflow.core.dataflow_model import DataflowModel
from ...dataflow.core.dataflow_interface import DataflowInterface
from ...dataflow.core.class_naming import generate_class_name, generate_backend_class_name

logger = logging.getLogger(__name__)


class TemplateSystemError(Exception):
    """Exception raised by template system operations."""
    pass


class UnifiedTemplateLoader:
    """
    Template loader for unified HWKG system.
    
    Provides Jinja2 environment setup and template loading with fallback
    to existing HWKG templates when needed for compatibility.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize template loader.
        
        Args:
            template_dir: Optional custom template directory
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.jinja_env = self._setup_jinja_environment()
        
    def _get_default_template_dir(self) -> Path:
        """Get default template directory for unified HWKG."""
        return Path(__file__).parent / "templates"
    
    def _setup_jinja_environment(self) -> jinja2.Environment:
        """
        Setup Jinja2 environment with template discovery.
        
        Returns:
            jinja2.Environment: Configured Jinja2 environment
        """
        # Primary loader: unified HWKG templates
        loaders = [jinja2.FileSystemLoader(self.template_dir)]
        
        # Fallback loader: existing HWKG templates for compatibility
        existing_template_dir = Path(__file__).parent.parent / "hw_kernel_gen" / "templates"
        if existing_template_dir.exists():
            loaders.append(jinja2.FileSystemLoader(existing_template_dir))
            
        # Create choice loader for fallback capability
        loader = jinja2.ChoiceLoader(loaders)
        
        env = jinja2.Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined  # Catch template errors early
        )
        
        # Add custom filters for DataflowModel serialization
        env.filters['serialize_chunking_strategy'] = self._serialize_chunking_strategy
        env.filters['serialize_dtype_constraint'] = self._serialize_dtype_constraint
        env.filters['format_interface_metadata'] = self._format_interface_metadata
        
        return env
    
    def load_template(self, template_name: str) -> jinja2.Template:
        """
        Load template by name.
        
        Args:
            template_name: Name of template file
            
        Returns:
            jinja2.Template: Loaded template
            
        Raises:
            TemplateSystemError: If template cannot be loaded
        """
        try:
            template = self.jinja_env.get_template(template_name)
            logger.debug(f"Loaded template: {template_name}")
            return template
        except jinja2.TemplateNotFound as e:
            raise TemplateSystemError(f"Template not found: {template_name}") from e
        except jinja2.TemplateSyntaxError as e:
            raise TemplateSystemError(f"Template syntax error in {template_name}: {e}") from e
    
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Load and render template with context.
        
        Args:
            template_name: Name of template file
            context: Template context dictionary
            
        Returns:
            str: Rendered template content
            
        Raises:
            TemplateSystemError: If rendering fails
        """
        try:
            template = self.load_template(template_name)
            rendered = template.render(**context)
            logger.debug(f"Rendered template {template_name} ({len(rendered)} chars)")
            return rendered
        except jinja2.TemplateRuntimeError as e:
            raise TemplateSystemError(f"Template rendering error in {template_name}: {e}") from e
    
    # Custom Jinja2 filters for DataflowModel serialization
    def _serialize_chunking_strategy(self, interface: DataflowInterface) -> Dict[str, Any]:
        """Serialize chunking strategy for template."""
        # Extract chunking information from interface
        return {
            'tensor_dims': interface.tensor_dims,
            'block_dims': interface.block_dims,
            'stream_dims': interface.stream_dims,
            'type': 'dataflow_interface'
        }
    
    def _serialize_dtype_constraint(self, interface: DataflowInterface) -> Dict[str, Any]:
        """Serialize datatype constraint for template."""
        return {
            'finn_type': interface.dtype.finn_type,
            'base_type': interface.dtype.base_type,
            'bitwidth': interface.dtype.bitwidth,
            'signed': interface.dtype.signed
        }
    
    def _format_interface_metadata(self, interface: DataflowInterface) -> Dict[str, Any]:
        """Format interface metadata for template usage."""
        return {
            'name': interface.name,
            'interface_type': interface.interface_type.value,
            'tensor_dims': interface.tensor_dims,
            'block_dims': interface.block_dims,
            'stream_dims': interface.stream_dims,
            'dtype': self._serialize_dtype_constraint(interface),
            'chunking': self._serialize_chunking_strategy(interface)
        }


class DataflowContextBuilder:
    """
    Context builder for DataflowModel-based templates.
    
    Creates template contexts that enable minimal instantiation templates
    rather than complex implementation code generation.
    """
    
    def __init__(self):
        """Initialize context builder."""
        pass
    
    def build_hwcustomop_context(self, dataflow_model: DataflowModel, 
                                kernel_name: str) -> Dict[str, Any]:
        """
        Build template context for HWCustomOp instantiation.
        
        This creates context for minimal instantiation templates that
        simply instantiate AutoHWCustomOp with the DataflowModel.
        
        Args:
            dataflow_model: DataflowModel instance
            kernel_name: Name of the kernel
            
        Returns:
            Dict containing template context
        """
        class_name = generate_class_name(kernel_name)
        
        # Serialize interfaces for template
        interfaces = []
        for interface in dataflow_model.interfaces.values():
            interfaces.append({
                'name': interface.name,
                'interface_type': interface.interface_type.value,
                'tensor_dims': interface.tensor_dims,
                'block_dims': interface.block_dims,
                'stream_dims': interface.stream_dims,
                'dtype_constraint': {
                    'finn_type': interface.dtype.finn_type,
                    'base_type': interface.dtype.base_type,
                    'bitwidth': interface.dtype.bitwidth,
                    'signed': interface.dtype.signed
                },
                'chunking_strategy': {
                    'type': 'interface_based',
                    'tensor_dims': interface.tensor_dims,
                    'block_dims': interface.block_dims
                }
            })
        
        context = {
            'kernel_name': kernel_name,
            'class_name': class_name,
            'interfaces': interfaces,
            'generation_timestamp': datetime.now().isoformat(),
            'dataflow_model_summary': {
                'num_interfaces': len(dataflow_model.interfaces),
                'input_count': len(dataflow_model.input_interfaces),
                'output_count': len(dataflow_model.output_interfaces),
                'weight_count': len(dataflow_model.weight_interfaces)
            },
            'import_paths': {
                'auto_hw_custom_op': 'brainsmith.dataflow.core.auto_hw_custom_op',
                'interface_metadata': 'brainsmith.dataflow.rtl_integration',
                'dataflow_interface': 'brainsmith.dataflow.core.dataflow_interface'
            }
        }
        
        return context
    
    def build_rtlbackend_context(self, dataflow_model: DataflowModel, 
                                kernel_name: str) -> Dict[str, Any]:
        """
        Build template context for RTLBackend instantiation.
        
        Args:
            dataflow_model: DataflowModel instance
            kernel_name: Name of the kernel
            
        Returns:
            Dict containing template context
        """
        class_name = generate_backend_class_name(kernel_name)
        
        # Serialize interfaces for RTL backend template
        interfaces = []
        for interface in dataflow_model.interfaces.values():
            interfaces.append({
                'name': interface.name,
                'interface_type': interface.interface_type.value,
                'tensor_dims': interface.tensor_dims,
                'block_dims': interface.block_dims,
                'stream_dims': interface.stream_dims,
                'dtype_config': {
                    'finn_type': interface.dtype.finn_type,
                    'signed': interface.dtype.signed,
                    'bitwidth': interface.dtype.bitwidth
                },
                'axi_metadata': {
                    'protocol': 'axi_stream',
                    'data_width': interface.dtype.bitwidth * (interface.stream_dims[0] if interface.stream_dims else 1)
                }
            })
        
        context = {
            'kernel_name': kernel_name,
            'class_name': class_name,
            'interfaces': interfaces,
            'generation_timestamp': datetime.now().isoformat(),
            'dataflow_model_summary': {
                'num_interfaces': len(dataflow_model.interfaces),
                'input_count': len(dataflow_model.input_interfaces),
                'output_count': len(dataflow_model.output_interfaces),
                'weight_count': len(dataflow_model.weight_interfaces)
            },
            'import_paths': {
                'auto_rtl_backend': 'brainsmith.dataflow.core.auto_rtl_backend',
                'dataflow_interface': 'brainsmith.dataflow.core.dataflow_interface'
            }
        }
        
        return context
    
    def build_test_context(self, dataflow_model: DataflowModel, 
                          kernel_name: str) -> Dict[str, Any]:
        """
        Build template context for test suite generation.
        
        Args:
            dataflow_model: DataflowModel instance
            kernel_name: Name of the kernel
            
        Returns:
            Dict containing template context
        """
        test_class_name = f"Test{generate_class_name(kernel_name)}"
        
        # Extract interface information for test generation
        input_interfaces = [iface.name for iface in dataflow_model.input_interfaces]
        output_interfaces = [iface.name for iface in dataflow_model.output_interfaces]
        weight_interfaces = [iface.name for iface in dataflow_model.weight_interfaces]
        
        context = {
            'kernel_name': kernel_name,
            'class_name': generate_class_name(kernel_name),
            'test_class_name': test_class_name,
            'hwcustomop_class': f"{generate_class_name(kernel_name)}HWCustomOp",
            'rtlbackend_class': f"{generate_backend_class_name(kernel_name)}RTLBackend",
            'interfaces': {
                'all': list(dataflow_model.interfaces.keys()),
                'input': input_interfaces,
                'output': output_interfaces,
                'weight': weight_interfaces
            },
            'test_scenarios': self._generate_test_scenarios(dataflow_model),
            'generation_timestamp': datetime.now().isoformat(),
            'import_paths': {
                'pytest': 'pytest',
                'numpy': 'numpy',
                'dataflow_model': 'brainsmith.dataflow.core.dataflow_model'
            }
        }
        
        return context
    
    def _generate_test_scenarios(self, dataflow_model: DataflowModel) -> List[Dict[str, Any]]:
        """Generate test scenarios based on dataflow model."""
        scenarios = []
        
        # Basic functionality test
        scenarios.append({
            'name': 'test_basic_functionality',
            'description': 'Test basic HWCustomOp instantiation and method calls',
            'test_type': 'functionality'
        })
        
        # Interface configuration test
        if dataflow_model.interfaces:
            scenarios.append({
                'name': 'test_interface_configuration',
                'description': 'Test interface configuration and metadata',
                'test_type': 'configuration'
            })
        
        # Dataflow model validation test
        scenarios.append({
            'name': 'test_dataflow_model',
            'description': 'Test DataflowModel mathematical correctness',
            'test_type': 'mathematical'
        })
        
        # Performance calculation test
        if dataflow_model.input_interfaces and dataflow_model.output_interfaces:
            scenarios.append({
                'name': 'test_performance_calculations',
                'description': 'Test performance and resource calculations',
                'test_type': 'performance'
            })
        
        return scenarios


def create_template_system() -> tuple[UnifiedTemplateLoader, DataflowContextBuilder]:
    """
    Factory function for creating template system components.
    
    Returns:
        tuple: (UnifiedTemplateLoader, DataflowContextBuilder)
    """
    template_loader = UnifiedTemplateLoader()
    context_builder = DataflowContextBuilder()
    
    return template_loader, context_builder