"""
Enhanced Template Manager with Dataflow Integration and Advanced Caching.

This module provides a sophisticated template management system that integrates
with the dataflow modeling framework and provides intelligent template selection,
caching, and rendering capabilities.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import time
import hashlib
from collections import OrderedDict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Jinja2 imports
try:
    from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound, TemplateError as Jinja2TemplateError
    from jinja2 import StrictUndefined, Undefined
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    # Create placeholder classes
    class Environment: pass
    class FileSystemLoader: pass
    class Template: pass
    class TemplateNotFound(Exception): pass
    class Jinja2TemplateError(Exception): pass
    class StrictUndefined: pass
    class Undefined: pass

from .enhanced_config import PipelineConfig, TemplateConfig, DataflowMode, GeneratorType
from .errors import TemplateError, ConfigurationError


class TemplateCache:
    """Advanced LRU cache with TTL support for compiled templates."""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """Initialize template cache."""
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self._cache: OrderedDict[str, Tuple[Template, float]] = OrderedDict()
        self._access_times: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Template]:
        """Get template from cache if valid."""
        current_time = time.time()
        
        if key in self._cache:
            template, cache_time = self._cache[key]
            
            # Check TTL
            if current_time - cache_time <= self.ttl:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._access_times[key] = current_time
                self._hits += 1
                return template
            else:
                # Expired, remove from cache
                del self._cache[key]
                del self._access_times[key]
        
        self._misses += 1
        return None
    
    def put(self, key: str, template: Template) -> None:
        """Add template to cache."""
        current_time = time.time()
        
        # Remove if already exists
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
        
        # Check size limit
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        # Add to cache
        self._cache[key] = (template, current_time)
        self._access_times[key] = current_time
    
    def clear(self) -> None:
        """Clear all cached templates."""
        self._cache.clear()
        self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(
            1 for _, cache_time in self._cache.values()
            if current_time - cache_time > self.ttl
        )
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
            "expired_count": expired_count,
            "ttl": self.ttl
        }


class TemplateSelector:
    """Intelligent template selection based on configuration and context."""
    
    def __init__(self, config: TemplateConfig):
        """Initialize template selector."""
        self.config = config
    
    def select_template(
        self, 
        template_name: str, 
        context: Dict[str, Any],
        generator_type: Optional[GeneratorType] = None
    ) -> Tuple[str, Path]:
        """
        Select the best template based on configuration and context.
        
        Returns:
            Tuple of (selected_template_name, template_directory)
        """
        # Check for explicit custom template override
        if template_name in self.config.custom_templates:
            custom_path = self.config.custom_templates[template_name]
            return template_name, custom_path.parent
        
        # Handle template selection strategy
        if self.config.template_selection_strategy == "custom":
            # Use only custom templates
            if template_name in self.config.custom_templates:
                custom_path = self.config.custom_templates[template_name]
                return template_name, custom_path.parent
            else:
                raise TemplateError(
                    f"Template '{template_name}' not found in custom templates",
                    template_name=template_name,
                    suggestion="Add template to custom_templates or change selection strategy"
                )
        
        elif self.config.template_selection_strategy == "dataflow":
            # Prefer dataflow templates
            return self._select_dataflow_template(template_name, context, generator_type)
        
        elif self.config.template_selection_strategy == "legacy":
            # Use only legacy templates
            return self._select_legacy_template(template_name, context)
        
        else:  # auto strategy
            # Intelligent selection based on context
            return self._auto_select_template(template_name, context, generator_type)
    
    def _select_dataflow_template(
        self, 
        template_name: str, 
        context: Dict[str, Any],
        generator_type: Optional[GeneratorType]
    ) -> Tuple[str, Path]:
        """Select dataflow-optimized template."""
        # Check dataflow template directories first
        for template_dir in self.config.dataflow_template_dirs:
            template_path = template_dir / template_name
            if template_path.exists():
                return template_name, template_dir
        
        # Fallback to regular templates if not found
        for template_dir in self.config.template_dirs:
            template_path = template_dir / template_name
            if template_path.exists():
                return template_name, template_dir
        
        raise TemplateError(
            f"Dataflow template '{template_name}' not found",
            template_name=template_name,
            suggestion="Check dataflow template directories or switch selection strategy"
        )
    
    def _select_legacy_template(self, template_name: str, context: Dict[str, Any]) -> Tuple[str, Path]:
        """Select legacy template."""
        for template_dir in self.config.template_dirs:
            template_path = template_dir / template_name
            if template_path.exists():
                return template_name, template_dir
        
        raise TemplateError(
            f"Legacy template '{template_name}' not found",
            template_name=template_name,
            suggestion="Check template directories or add missing template"
        )
    
    def _auto_select_template(
        self, 
        template_name: str, 
        context: Dict[str, Any],
        generator_type: Optional[GeneratorType]
    ) -> Tuple[str, Path]:
        """Automatically select best template based on context."""
        # Determine if dataflow is being used
        dataflow_enabled = context.get("config_metadata", {}).get("dataflow_enabled", False)
        use_autogenerated = context.get("use_autohwcustomop_base", False) or context.get("use_autortlbackend_base", False)
        
        # Select template variant based on capabilities
        if dataflow_enabled and use_autogenerated:
            # Try dataflow-optimized slim templates first
            slim_template_name = self._get_slim_template_name(template_name, generator_type)
            
            # Check dataflow directories for slim templates
            for template_dir in self.config.dataflow_template_dirs:
                template_path = template_dir / slim_template_name
                if template_path.exists():
                    return slim_template_name, template_dir
            
            # Try regular dataflow templates
            for template_dir in self.config.dataflow_template_dirs:
                template_path = template_dir / template_name
                if template_path.exists():
                    return template_name, template_dir
        
        # Fallback to standard template selection
        all_dirs = list(self.config.dataflow_template_dirs) + list(self.config.template_dirs)
        for template_dir in all_dirs:
            template_path = template_dir / template_name
            if template_path.exists():
                return template_name, template_dir
        
        raise TemplateError(
            f"Template '{template_name}' not found in any directory",
            template_name=template_name,
            context={"searched_dirs": [str(d) for d in all_dirs]},
            suggestion="Check template name and ensure template exists in one of the configured directories"
        )
    
    def _get_slim_template_name(self, template_name: str, generator_type: Optional[GeneratorType]) -> str:
        """Get the slim template variant name."""
        # Map template names to their slim variants
        slim_mapping = {
            "hw_custom_op.py.j2": "hw_custom_op_slim.py.j2",
            "rtl_backend.py.j2": "rtl_backend_slim.py.j2",
            "test_suite.py.j2": "test_suite_slim.py.j2"
        }
        
        return slim_mapping.get(template_name, template_name)


class EnhancedTemplateManager:
    """
    Enhanced template manager with dataflow integration and intelligent caching.
    
    This manager provides sophisticated template handling with support for:
    - Multiple template directories with priority ordering
    - Intelligent template selection based on context
    - Advanced caching with TTL support
    - Dataflow-aware template variants
    - Custom Jinja2 filters and functions
    """
    
    def __init__(self, config: TemplateConfig):
        """Initialize enhanced template manager."""
        if not JINJA2_AVAILABLE:
            raise ConfigurationError(
                "Jinja2 is required for template management",
                config_section="template",
                suggestion="Install Jinja2: pip install jinja2"
            )
        
        self.config = config
        self.selector = TemplateSelector(config)
        
        # Initialize cache
        self.cache: Optional[TemplateCache] = None
        if self.config.enable_caching:
            self.cache = TemplateCache(
                max_size=getattr(self.config, 'cache_size', 100),
                ttl=getattr(self.config, 'cache_ttl', 3600.0)
            )
        
        # Create Jinja2 environments for each directory
        self.environments: Dict[str, Environment] = {}
        self._create_environments()
        
        # Template rendering statistics
        self._render_count = 0
        self._total_render_time = 0.0
        self._last_render_time = None
    
    def _create_environments(self) -> None:
        """Create Jinja2 environments for template directories."""
        all_dirs = list(self.config.dataflow_template_dirs) + list(self.config.template_dirs)
        
        for template_dir in all_dirs:
            if template_dir.exists():
                env = Environment(
                    loader=FileSystemLoader(str(template_dir)),
                    auto_reload=self.config.auto_reload,
                    undefined=StrictUndefined if self.config.strict_undefined else Undefined,
                    trim_blocks=self.config.trim_blocks,
                    lstrip_blocks=self.config.lstrip_blocks,
                    keep_trailing_newline=self.config.keep_trailing_newline
                )
                
                # Add custom filters and functions
                self._add_custom_filters(env)
                self._add_custom_functions(env)
                
                self.environments[str(template_dir)] = env
    
    def _add_custom_filters(self, env: Environment) -> None:
        """Add custom Jinja2 filters for hardware generation."""
        
        def camel_case(text: str) -> str:
            """Convert snake_case to CamelCase."""
            words = text.replace('-', '_').split('_')
            return ''.join(word.capitalize() for word in words)
        
        def snake_case(text: str) -> str:
            """Convert CamelCase to snake_case."""
            import re
            # Insert underscores before uppercase letters that follow lowercase letters
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
            # Insert underscores before uppercase letters that follow lowercase letters or other uppercase letters
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
        def hex_format(value: int, width: int = 8) -> str:
            """Format integer as hexadecimal with specified width."""
            return f"0x{value:0{width}x}"
        
        def bit_width(value: int) -> int:
            """Calculate minimum bit width needed for value."""
            if value == 0:
                return 1
            return value.bit_length()
        
        def signal_list(signals: List[Dict[str, Any]], direction: str = None) -> List[Dict[str, Any]]:
            """Filter signals by direction."""
            if direction is None:
                return signals
            return [sig for sig in signals if sig.get("direction") == direction]
        
        def join_with_and(items: List[str]) -> str:
            """Join list with commas and 'and' for last item."""
            if len(items) == 0:
                return ""
            elif len(items) == 1:
                return items[0]
            elif len(items) == 2:
                return f"{items[0]} and {items[1]}"
            else:
                return ", ".join(items[:-1]) + f", and {items[-1]}"
        
        # Register filters
        env.filters['camel_case'] = camel_case
        env.filters['snake_case'] = snake_case
        env.filters['hex_format'] = hex_format
        env.filters['bit_width'] = bit_width
        env.filters['signal_list'] = signal_list
        env.filters['join_with_and'] = join_with_and
    
    def _add_custom_functions(self, env: Environment) -> None:
        """Add custom Jinja2 global functions."""
        
        def range_inclusive(start: int, end: int) -> range:
            """Inclusive range function."""
            return range(start, end + 1)
        
        def timestamp() -> str:
            """Current timestamp."""
            return datetime.now().isoformat()
        
        def format_list(items: List[Any], template: str) -> List[str]:
            """Format each item in list using template string."""
            return [template.format(item) for item in items]
        
        def max_value(*args) -> Any:
            """Return maximum value from arguments."""
            return max(args)
        
        def min_value(*args) -> Any:
            """Return minimum value from arguments."""
            return min(args)
        
        # Register functions
        env.globals['range_inclusive'] = range_inclusive
        env.globals['timestamp'] = timestamp
        env.globals['format_list'] = format_list
        env.globals['max_value'] = max_value
        env.globals['min_value'] = min_value
    
    def get_template(self, template_name: str, context: Optional[Dict[str, Any]] = None) -> Template:
        """
        Get compiled template with intelligent selection and caching.
        
        Args:
            template_name: Name of the template file
            context: Template context for selection optimization
            
        Returns:
            Compiled Jinja2 Template object
        """
        context = context or {}
        
        # Select appropriate template
        selected_name, template_dir = self.selector.select_template(
            template_name, 
            context,
            context.get("generator_type")
        )
        
        # Create cache key
        cache_key = f"{template_dir}:{selected_name}"
        
        # Check cache
        if self.cache:
            cached_template = self.cache.get(cache_key)
            if cached_template:
                return cached_template
        
        # Load template
        try:
            env = self.environments.get(str(template_dir))
            if not env:
                raise TemplateError(
                    f"No environment found for template directory: {template_dir}",
                    template_name=template_name,
                    suggestion="Check template directory configuration"
                )
            
            template = env.get_template(selected_name)
            
            # Cache if enabled
            if self.cache:
                self.cache.put(cache_key, template)
            
            return template
            
        except TemplateNotFound as e:
            raise TemplateError(
                f"Template '{selected_name}' not found in directory '{template_dir}'",
                template_name=template_name,
                context={"template_dir": str(template_dir), "selected_name": selected_name},
                suggestion="Check template name and ensure file exists"
            ) from e
        except Jinja2TemplateError as e:
            raise TemplateError(
                f"Template syntax error in '{selected_name}': {e}",
                template_name=template_name,
                suggestion="Check template syntax and fix any Jinja2 errors"
            ) from e
    
    def render_template(
        self, 
        template_name: str, 
        context: Dict[str, Any],
        generator_type: Optional[GeneratorType] = None
    ) -> str:
        """
        Render template with context.
        
        Args:
            template_name: Name of the template file
            context: Template context variables
            generator_type: Type of generator for template selection
            
        Returns:
            Rendered template content
        """
        start_time = time.time()
        
        try:
            # Add generator type to context for template selection
            if generator_type:
                context = dict(context)
                context["generator_type"] = generator_type
            
            # Get template
            template = self.get_template(template_name, context)
            
            # Render template
            rendered = template.render(**context)
            
            # Update statistics
            render_time = time.time() - start_time
            self._render_count += 1
            self._total_render_time += render_time
            self._last_render_time = render_time
            
            return rendered
            
        except Jinja2TemplateError as e:
            raise TemplateError(
                f"Template rendering error in '{template_name}': {e}",
                template_name=template_name,
                context={"error_line": getattr(e, "lineno", None)},
                suggestion="Check template context variables and syntax"
            ) from e
        except Exception as e:
            raise TemplateError(
                f"Unexpected error rendering template '{template_name}': {e}",
                template_name=template_name,
                suggestion="Check template and context for issues"
            ) from e
    
    def render_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """
        Render template from string.
        
        Args:
            template_string: Template content as string
            context: Template context variables
            
        Returns:
            Rendered content
        """
        start_time = time.time()
        
        try:
            # Use first available environment
            env = next(iter(self.environments.values()))
            template = env.from_string(template_string)
            
            rendered = template.render(**context)
            
            # Update statistics
            render_time = time.time() - start_time
            self._render_count += 1
            self._total_render_time += render_time
            self._last_render_time = render_time
            
            return rendered
            
        except Jinja2TemplateError as e:
            raise TemplateError(
                f"String template rendering error: {e}",
                template_name="<string>",
                suggestion="Check template string syntax and context variables"
            ) from e
    
    def list_templates(self, pattern: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            pattern: Optional pattern to filter templates
            
        Returns:
            List of template names
        """
        templates = set()
        
        # Collect from all environments
        for env in self.environments.values():
            try:
                env_templates = env.list_templates()
                if pattern:
                    import fnmatch
                    env_templates = [t for t in env_templates if fnmatch.fnmatch(t, pattern)]
                templates.update(env_templates)
            except Exception:
                # Skip environments that don't support listing
                continue
        
        return sorted(list(templates))
    
    def template_exists(self, template_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if template exists.
        
        Args:
            template_name: Name of the template
            context: Context for template selection
            
        Returns:
            True if template exists
        """
        try:
            self.get_template(template_name, context)
            return True
        except TemplateError:
            return False
    
    def clear_cache(self) -> None:
        """Clear template cache."""
        if self.cache:
            self.cache.clear()
    
    def reload_templates(self) -> None:
        """Reload templates by recreating environments."""
        self.clear_cache()
        self.environments.clear()
        self._create_environments()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get template manager statistics."""
        stats = {
            "render_count": self._render_count,
            "total_render_time": self._total_render_time,
            "average_render_time": self._total_render_time / max(self._render_count, 1),
            "last_render_time": self._last_render_time,
            "environments_count": len(self.environments),
            "cache_enabled": self.cache is not None
        }
        
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        return stats


# Factory functions for convenience
def create_template_manager(config: TemplateConfig) -> EnhancedTemplateManager:
    """Create an enhanced template manager with the given configuration."""
    return EnhancedTemplateManager(config)


def create_default_template_manager(
    template_dirs: Optional[List[Path]] = None,
    enable_dataflow: bool = True
) -> EnhancedTemplateManager:
    """Create a template manager with sensible defaults."""
    if template_dirs is None:
        base_dir = Path(__file__).parent / "templates"
        template_dirs = [base_dir]
    
    config = TemplateConfig(
        template_dirs=template_dirs,
        dataflow_template_dirs=[Path(__file__).parent / "templates" / "dataflow"] if enable_dataflow else [],
        enable_caching=True,
        cache_size=100,
        cache_ttl=3600,
        prefer_dataflow_templates=enable_dataflow,
        template_selection_strategy="auto"
    )
    
    return EnhancedTemplateManager(config)