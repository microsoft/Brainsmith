"""
Generator discovery and management for HWKG.

The GeneratorManager handles discovery of generator classes, loading templates,
and rendering output through the Jinja2 template engine.
"""

import glob
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .base import GeneratorBase
from ..templates.template_context import TemplateContext

logger = logging.getLogger(__name__)


class GeneratorManagerError(Exception):
    """Custom exception for GeneratorManager errors."""
    pass


class GeneratorManager:
    """
    Manages generator discovery, loading, and template rendering.
    
    Automatically discovers generator classes from *_generator.py files
    and provides template rendering capabilities through Jinja2.
    """
    
    def __init__(self, generator_dir: Path, template_dir: Path):
        """
        Initialize GeneratorManager.
        
        Args:
            generator_dir: Directory containing generator Python files
            template_dir: Directory containing Jinja2 template files
        """
        self.generator_dir = Path(generator_dir)
        self.template_dir = Path(template_dir)
        self.generators: Dict[str, GeneratorBase] = {}
        
        # Initialize Jinja2 environment
        try:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Add custom filters and tests
            self.jinja_env.filters['repr'] = repr
            self.jinja_env.tests['none'] = lambda x: x is None
            
            logger.info(f"Initialized Jinja2 environment with templates from {self.template_dir}")
        except Exception as e:
            raise GeneratorManagerError(f"Failed to initialize Jinja2 environment: {e}")
        
        # Discover generators
        self._discover_generators()
        logger.info(f"Discovered {len(self.generators)} generators")
    
    def _discover_generators(self) -> None:
        """Auto-discover generator definitions using package introspection."""
        # Import the generators package to ensure all generators are loaded
        try:
            from . import hw_custom_op_generator, rtl_wrapper_generator, rtl_backend_generator
            
            # Use introspection to find all GeneratorBase subclasses in current namespace
            import sys
            current_module = sys.modules[__name__]
            generators_package = sys.modules[current_module.__package__]
            
            # Find all GeneratorBase subclasses in the package
            for name in dir(generators_package):
                obj = getattr(generators_package, name)
                if (inspect.isclass(obj) and 
                    issubclass(obj, GeneratorBase) and
                    obj != GeneratorBase):
                    
                    try:
                        generator_instance = obj()
                        if generator_instance.validate():
                            if generator_instance.name in self.generators:
                                logger.warning(f"Duplicate generator name: {generator_instance.name}")
                            self.generators[generator_instance.name] = generator_instance
                            logger.debug(f"Registered generator: {generator_instance}")
                        else:
                            logger.warning(f"Invalid generator configuration: {obj.__name__}")
                    except Exception as e:
                        logger.error(f"Failed to instantiate generator {obj.__name__}: {e}")
                        
        except ImportError as e:
            logger.error(f"Failed to import generators package: {e}")
            # Fallback to dynamic discovery if package imports fail
            self._discover_generators_dynamically()
    
    def _discover_generators_dynamically(self) -> None:
        """Fallback: Dynamic generator discovery from files."""
        generator_pattern = str(self.generator_dir / "*_generator.py")
        
        for generator_file in glob.glob(generator_pattern):
            try:
                module = self._import_generator_module(generator_file)
                
                # Find GeneratorBase subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, GeneratorBase) and
                        obj != GeneratorBase):
                        
                        # Instantiate and validate generator
                        generator_instance = obj()
                        if generator_instance.validate():
                            if generator_instance.name in self.generators:
                                logger.warning(f"Duplicate generator name: {generator_instance.name}")
                            self.generators[generator_instance.name] = generator_instance
                            logger.debug(f"Registered generator: {generator_instance}")
                        else:
                            logger.warning(f"Invalid generator configuration: {obj.__name__}")
                            
            except Exception as e:
                logger.error(f"Failed to load generator from {generator_file}: {e}")
                continue
    
    def _import_generator_module(self, generator_file: str):
        """Import a generator module from file path."""
        module_name = Path(generator_file).stem
        
        # Add the parent directory to sys.path temporarily to help with imports
        import sys
        parent_dir = str(Path(generator_file).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, generator_file)
            if spec is None or spec.loader is None:
                raise GeneratorManagerError(f"Cannot load module from {generator_file}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.error(f"Failed to import generator from {generator_file}: {e}")
            raise
    
    def list_generators(self) -> List[str]:
        """
        Get list of available generator names.
        
        Returns:
            List of generator names
        """
        return list(self.generators.keys())
    
    def get_generator(self, name: str) -> Optional[GeneratorBase]:
        """
        Get generator by name.
        
        Args:
            name: Generator name
            
        Returns:
            Generator instance or None if not found
        """
        return self.generators.get(name)
    
    def render_generator(
        self, 
        generator_name: str, 
        context: TemplateContext
    ) -> str:
        """
        Render template via generator with processed context.
        
        Args:
            generator_name: Name of generator to use
            context: Full template context
            
        Returns:
            Rendered template content
            
        Raises:
            GeneratorManagerError: If generator not found or rendering fails
        """
        if generator_name not in self.generators:
            available = ", ".join(self.generators.keys())
            raise GeneratorManagerError(
                f"Generator '{generator_name}' not found. Available: {available}"
            )
        
        generator = self.generators[generator_name]
        
        try:
            # Let generator process context
            processed_context = generator.process_context(context)
            
            # Get template file (with potential fallbacks)
            try:
                template_file = generator.get_template_file(self.jinja_env)
                template = self.jinja_env.get_template(template_file)
            except TemplateNotFound:
                raise GeneratorManagerError(
                    f"Template not found: {generator.template_file} for generator {generator_name}"
                )
            
            rendered_content = template.render(**processed_context)
            logger.debug(f"Successfully rendered {generator_name}")
            return rendered_content
            
        except Exception as e:
            logger.error(f"Failed to render {generator_name}: {e}")
            raise GeneratorManagerError(f"Rendering failed for {generator_name}: {e}")
    
    def get_output_filename(self, generator_name: str, kernel_name: str) -> str:
        """
        Get output filename for a generator.
        
        Args:
            generator_name: Name of generator
            kernel_name: Name of kernel
            
        Returns:
            Output filename
            
        Raises:
            GeneratorManagerError: If generator not found
        """
        if generator_name not in self.generators:
            raise GeneratorManagerError(f"Generator '{generator_name}' not found")
        
        return self.generators[generator_name].get_output_filename(kernel_name)
    
    def validate_templates(self) -> Dict[str, bool]:
        """
        Validate that all generator templates exist.
        
        Returns:
            Dictionary mapping generator names to template availability
        """
        status = {}
        
        for name, generator in self.generators.items():
            template_path = self.template_dir / generator.template_file
            status[name] = template_path.exists()
            
            if not status[name]:
                logger.warning(f"Template not found for {name}: {generator.template_file}")
        
        return status