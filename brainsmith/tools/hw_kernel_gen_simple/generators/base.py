"""
Simple base class for all generators.

Provides clean generator pattern without enterprise abstractions.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import jinja2

from ..data import HWKernel
from ..errors import TemplateError, GenerationError


class GeneratorBase(ABC):
    """Simple base class for all HWKG generators."""
    
    def __init__(self, template_name: str, template_dir: Optional[Path] = None):
        self.template_name = template_name
        self.template_env = self._setup_jinja_env(template_dir)
    
    def _setup_jinja_env(self, template_dir: Optional[Path] = None) -> jinja2.Environment:
        """Setup Jinja2 environment."""
        if template_dir and template_dir.exists():
            loader = jinja2.FileSystemLoader(template_dir)
        else:
            # Use existing templates from the original HWKG
            template_path = Path(__file__).parent.parent.parent / "hw_kernel_gen" / "templates"
            if template_path.exists():
                loader = jinja2.FileSystemLoader(template_path)
            else:
                raise TemplateError(f"Template directory not found: {template_path}")
        
        env = jinja2.Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True
        )
        return env
    
    def generate(self, hw_kernel: HWKernel, output_dir: Path) -> Path:
        """Generate output file for the given hardware kernel."""
        try:
            template = self.template_env.get_template(self.template_name)
            content = template.render(
                hw_kernel=hw_kernel, 
                **self._get_template_context(hw_kernel)
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
        """Get additional template context. Override in subclasses."""
        return {
            'class_name': hw_kernel.class_name,
            'kernel_name': hw_kernel.kernel_name,
            'source_file': hw_kernel.source_file.name,
            'generation_timestamp': hw_kernel.generation_timestamp,
            'interfaces': hw_kernel.interfaces,
            'rtl_parameters': hw_kernel.rtl_parameters,
            'resource_estimation_required': hw_kernel.resource_estimation_required,
            'verification_required': hw_kernel.verification_required,
            'weight_interfaces_count': hw_kernel.weight_interfaces_count,
            'kernel_complexity': hw_kernel.kernel_complexity,
            'kernel_type': hw_kernel.kernel_type
        }