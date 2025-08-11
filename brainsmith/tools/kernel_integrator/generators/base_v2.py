"""Enhanced base generator with direct KernelMetadata support."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import jinja2

from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata


class GeneratorBase(ABC):
    """Base class for all code generators with direct KernelMetadata support."""
    
    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize generator with template directory."""
        self.template_dir = template_dir or Path(__file__).parent.parent / "templates"
        self._env = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name for identification."""
        pass
    
    @property
    @abstractmethod
    def template_file(self) -> str:
        """Template filename to use."""
        pass
    
    @property
    @abstractmethod
    def output_pattern(self) -> str:
        """Output filename pattern with {kernel_name} placeholder."""
        pass
    
    def generate(self, metadata: KernelMetadata) -> str:
        """Generate code from KernelMetadata."""
        # Create Jinja environment
        env = self._create_jinja_env()
        
        # Load template
        template = env.get_template(self.template_file)
        
        # Prepare template variables
        template_vars = self._prepare_template_vars(metadata)
        
        # Render template
        return template.render(**template_vars)
    
    def _prepare_template_vars(self, metadata: KernelMetadata) -> Dict[str, Any]:
        """Prepare all template variables."""
        # Start with common variables
        vars_dict = self._get_common_vars(metadata)
        
        # Add generator-specific variables
        specific_vars = self._get_specific_vars(metadata)
        vars_dict.update(specific_vars)
        
        return vars_dict
    
    def _get_common_vars(self, metadata: KernelMetadata) -> Dict[str, Any]:
        """Get variables common to all generators."""
        return {
            # Only pass metadata itself, let templates access properties directly
            'kernel_metadata': metadata,
        }
    
    @abstractmethod
    def _get_specific_vars(self, metadata: KernelMetadata) -> Dict[str, Any]:
        """Get generator-specific template variables.
        
        Subclasses implement this to add their custom transformations.
        """
        pass
    
    def _create_jinja_env(self) -> jinja2.Environment:
        """Create Jinja environment with filters and globals."""
        if self._env is None:
            self._env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
                undefined=jinja2.StrictUndefined
            )
            
            # Add custom filters
            self._env.filters['snake_to_camel'] = self._snake_to_camel
            self._env.filters['camel_to_snake'] = self._camel_to_snake
            
            # Add custom globals
            self._env.globals['enumerate'] = enumerate
            self._env.globals['zip'] = zip
            self._env.globals['len'] = len
            
        return self._env
    
    def get_output_filename(self, kernel_name: str) -> str:
        """Get output filename for given kernel."""
        return self.output_pattern.format(kernel_name=kernel_name)
    
    @staticmethod
    def _snake_to_camel(snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)
    
    @staticmethod
    def _camel_to_snake(camel_str: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()