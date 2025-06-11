"""
Enhanced base class for unified generators.

Based on hw_kernel_gen_simple GeneratorBase with enhancements for
optional BDIM sophistication while maintaining template compatibility.
Follows HWKG Axiom 9: Generator Factory Pattern.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import jinja2

from ..rtl_parser.data import HWKernel
from ..errors import TemplateError, GenerationError


class GeneratorBase(ABC):
    """
    Enhanced base class for all unified HWKG generators.
    
    Based on hw_kernel_gen_simple GeneratorBase with enhancements for
    optional BDIM pragma processing while maintaining template compatibility
    and error resilience.
    """
    
    def __init__(self, template_name: str, template_dir: Optional[Path] = None):
        self.template_name = template_name
        self.template_env = self._setup_jinja_env(template_dir)
    
    def _setup_jinja_env(self, template_dir: Optional[Path] = None) -> jinja2.Environment:
        """
        Setup Jinja2 environment with template discovery.
        
        Following HWKG Axiom 3: Template-Driven Code Generation.
        Uses existing templates from hw_kernel_gen for compatibility.
        """
        if template_dir and template_dir.exists():
            loader = jinja2.FileSystemLoader(template_dir)
        else:
            # Use unified templates first, fallback to original HWKG templates
            unified_template_path = Path(__file__).parent.parent / "templates"
            original_template_path = Path(__file__).parent.parent.parent / "hw_kernel_gen" / "templates"
            
            if unified_template_path.exists():
                loader = jinja2.FileSystemLoader(unified_template_path)
            elif original_template_path.exists():
                loader = jinja2.FileSystemLoader(original_template_path)
            else:
                raise TemplateError(f"Template directory not found: tried {unified_template_path} and {original_template_path}")
        
        env = jinja2.Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True
        )
        return env
    
    def generate(self, hw_kernel: HWKernel, output_dir: Path) -> Path:
        """Generate output file for the given unified hardware kernel."""
        try:
            template = self.template_env.get_template(self.template_name)
            
            # Build template context with enhanced capabilities
            context = self._get_template_context(hw_kernel)
            
            # Render template with unified context
            content = template.render(
                hw_kernel=hw_kernel, 
                **context
            )
            
            output_file = output_dir / self._get_output_filename(hw_kernel)
            output_file.write_text(content)
            
            return output_file
            
        except jinja2.TemplateError as e:
            raise TemplateError(f"Template rendering failed: {e}") from e
        except Exception as e:
            raise GenerationError(f"File generation failed: {e}") from e
    
    @abstractmethod
    def _get_output_filename(self, hw_kernel: HWKernel) -> str:
        """Get output filename for the kernel."""
        pass
    
    def _get_template_context(self, hw_kernel: HWKernel) -> dict:
        """
        Get enhanced template context for unified HWKG.
        
        Following HWKG Axiom 6: Metadata-Driven Generation.
        Provides both simple and advanced context based on sophistication level.
        """
        # Base context (identical to hw_kernel_gen_simple)
        context = {
            'class_name': hw_kernel.class_name,
            'kernel_name': hw_kernel.kernel_name,
            'source_file': str(hw_kernel.source_file) if hw_kernel.source_file else 'unknown',
            'generation_timestamp': hw_kernel.generation_timestamp,
            'interfaces': list(hw_kernel.interfaces.values()),  # Convert to list of Interface objects
            'rtl_parameters': hw_kernel.rtl_parameters,
            'resource_estimation_required': hw_kernel.resource_estimation_required,
            'verification_required': hw_kernel.verification_required,
            'weight_interfaces_count': hw_kernel.weight_interfaces_count,
            'kernel_complexity': hw_kernel.kernel_complexity,
            'kernel_type': hw_kernel.kernel_type
        }
        
        # Enhanced context for advanced BDIM pragma mode
        if hw_kernel.has_enhanced_bdim:
            context.update({
                'enhanced_bdim_available': True,
                'bdim_metadata': hw_kernel.bdim_metadata,
                'chunking_strategies': hw_kernel.chunking_strategies,
                'tensor_dims_metadata': hw_kernel.tensor_dims_metadata,
                'block_dims_metadata': hw_kernel.block_dims_metadata,
                'stream_dims_metadata': hw_kernel.stream_dims_metadata,
                'dataflow_interfaces': hw_kernel.dataflow_interfaces,
                'pragma_sophistication_level': hw_kernel.pragma_sophistication_level
            })
        else:
            context.update({
                'enhanced_bdim_available': False,
                'pragma_sophistication_level': 'simple'
            })
        
        # Add Interface-Wise Dataflow context following axioms
        interface_categories = self._categorize_interfaces(hw_kernel.interfaces)
        context.update({
            'follows_dataflow_axioms': True,
            'interface_count': len(hw_kernel.interfaces),
            'dataflow_interface_count': len(hw_kernel.dataflow_interfaces),
            'complexity_level': hw_kernel.pragma_sophistication_level,
            # Add categorized interfaces for template access
            'input_interfaces': interface_categories['input_interfaces'],
            'output_interfaces': interface_categories['output_interfaces'],
            'weight_interfaces': interface_categories['weight_interfaces'],
            'config_interfaces': interface_categories['config_interfaces'],
            'control_interfaces': interface_categories['control_interfaces']
        })
        
        return context
    
    def _categorize_interfaces(self, interfaces) -> dict:
        """
        Categorize interfaces following Interface-Wise Dataflow Axiom 3.
        
        Categories: Input, Output, Weight, Config/Control
        """
        categories = {
            'input_interfaces': [],
            'output_interfaces': [],
            'weight_interfaces': [],
            'config_interfaces': [],
            'control_interfaces': []
        }
        
        # Handle both dict and Interface object formats
        interface_list = interfaces.values() if hasattr(interfaces, 'values') else interfaces
        
        for iface in interface_list:
            # Handle Interface objects from unified HWKernel
            if hasattr(iface, 'type') and hasattr(iface, 'name'):
                # Interface object from RTL parser
                interface_type = iface.type.value if hasattr(iface.type, 'value') else str(iface.type)
                is_weight = iface.metadata.get('is_weight', False)
                interface_name = iface.name.lower()
                
                # Determine dataflow type based on interface type and naming patterns
                if interface_type == 'axistream':
                    # Enhanced pattern-based classification for AXI-Stream
                    if is_weight or any(pattern in interface_name for pattern in ['weight', 'w_axis', 'param', 'kernel', 'filter']):
                        categories['weight_interfaces'].append(iface)
                    elif any(pattern in interface_name for pattern in ['s_axis', 'input', 'in_', 'slave']) or interface_name.startswith('s_'):
                        categories['input_interfaces'].append(iface)
                    elif any(pattern in interface_name for pattern in ['m_axis', 'output', 'out_', 'master']) or interface_name.startswith('m_'):
                        categories['output_interfaces'].append(iface)
                    else:
                        # Check port patterns as fallback
                        has_tvalid = any('tvalid' in port.name.lower() for port in iface.ports.values())
                        has_tready = any('tready' in port.name.lower() for port in iface.ports.values())
                        
                        if has_tvalid and not has_tready:
                            categories['input_interfaces'].append(iface)
                        elif has_tready and not has_tvalid:
                            categories['output_interfaces'].append(iface)
                        else:
                            categories['input_interfaces'].append(iface)  # Default assumption
                            
                elif interface_type == 'axilite':
                    categories['config_interfaces'].append(iface)
                elif interface_type == 'global':
                    categories['control_interfaces'].append(iface)
                else:
                    categories['control_interfaces'].append(iface)
            else:
                # Dict format (legacy compatibility)
                dataflow_type = iface.get('dataflow_type', 'UNKNOWN').upper()
                if dataflow_type == 'INPUT':
                    categories['input_interfaces'].append(iface)
                elif dataflow_type == 'OUTPUT':
                    categories['output_interfaces'].append(iface)
                elif dataflow_type == 'WEIGHT':
                    categories['weight_interfaces'].append(iface)
                else:
                    categories['control_interfaces'].append(iface)
        
        return categories