"""
Enhanced base class for HW Kernel Generator.

Based on hw_kernel_gen_simple GeneratorBase with enhancements for
optional BDIM sophistication while maintaining template compatibility.
Follows HWKG Axiom 9: Generator Factory Pattern.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import jinja2

from brainsmith.dataflow.core.kernel_metadata import KernelMetadata
from ..errors import TemplateError, GenerationError
from ..templates.context_generator import TemplateContextGenerator


class GeneratorBase(ABC):
    """
    Enhanced base class for all HWKG generators.
    
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
            # Use current templates first, fallback to legacy templates if needed
            current_template_path = Path(__file__).parent.parent / "templates"
            legacy_template_path = Path(__file__).parent.parent.parent / "hw_kernel_gen_legacy" / "templates"
            
            if current_template_path.exists():
                loader = jinja2.FileSystemLoader(current_template_path)
            elif legacy_template_path.exists():
                loader = jinja2.FileSystemLoader(legacy_template_path)
            else:
                raise TemplateError(f"Template directory not found: tried {current_template_path} and {legacy_template_path}")
        
        env = jinja2.Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True
        )
        return env
    
    def generate(self, kernel_metadata: KernelMetadata, output_dir: Path) -> Path:
        """Generate output file for the given hardware kernel."""
        try:
            template = self.template_env.get_template(self.template_name)
            
            # Build template context with enhanced capabilities
            context = self._get_template_context(kernel_metadata)
            
            # Render template with context
            content = template.render(
                kernel_metadata=kernel_metadata, 
                **context
            )
            
            output_file = output_dir / self._get_output_filename(kernel_metadata)
            output_file.write_text(content)
            
            return output_file
            
        except jinja2.TemplateError as e:
            raise TemplateError(f"Template rendering failed: {e}") from e
        except Exception as e:
            raise GenerationError(f"File generation failed: {e}") from e
    
    @abstractmethod
    def _get_output_filename(self, kernel_metadata: KernelMetadata) -> str:
        """Get output filename for the kernel."""
        pass
    
    def _get_template_context(self, kernel_metadata: KernelMetadata) -> dict:
        """
        Get enhanced template context for HWKG.
        
        Following HWKG Axiom 6: Metadata-Driven Generation.
        Uses TemplateContextGenerator to build context from KernelMetadata.
        """
        # Get complete template context using the context generator
        context = TemplateContextGenerator.generate_context(kernel_metadata)
        
        # Add additional generator-specific context if needed
        context.update({
            'follows_dataflow_axioms': True,
            'enhanced_bdim_available': len(kernel_metadata.pragmas) > 0,
            'pragma_sophistication_level': 'advanced' if len(kernel_metadata.pragmas) > 0 else 'simple',
        })
        
        return context
