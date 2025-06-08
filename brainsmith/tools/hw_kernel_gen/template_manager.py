"""
Template management system for the Hardware Kernel Generator.

This module provides centralized template management with caching, Jinja2
environment optimization, and template loading capabilities.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from functools import lru_cache
import hashlib
import time

try:
    from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
    from jinja2 import select_autoescape, StrictUndefined
except ImportError:
    raise ImportError("Jinja2 is required for template management. Install with: pip install jinja2")

from .config import TemplateConfig
from .errors import CodeGenerationError


class TemplateCache:
    """Cache for compiled templates with size and time-based eviction."""
    
    def __init__(self, max_size: int = 100, max_age: int = 3600):
        """Initialize template cache.
        
        Args:
            max_size: Maximum number of templates to cache
            max_age: Maximum age of cached templates in seconds
        """
        self.max_size = max_size
        self.max_age = max_age
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Template]:
        """Get template from cache if valid."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check if entry is expired
        if time.time() - entry['timestamp'] > self.max_age:
            del self._cache[key]
            return None
        
        # Update access time for LRU
        entry['access_time'] = time.time()
        return entry['template']
    
    def put(self, key: str, template: Template) -> None:
        """Store template in cache with eviction if needed."""
        # Evict oldest entries if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[key] = {
            'template': template,
            'timestamp': time.time(),
            'access_time': time.time()
        }
    
    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) template."""
        if not self._cache:
            return
        
        # Find the entry with the oldest access time
        oldest_key = min(self._cache.keys(), 
                        key=lambda k: self._cache[k]['access_time'])
        del self._cache[oldest_key]
    
    def clear(self) -> None:
        """Clear all cached templates."""
        self._cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'max_age': self.max_age,
            'cached_templates': list(self._cache.keys())
        }


class TemplateManager:
    """Centralized template management with caching and optimization."""
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """Initialize template manager.
        
        Args:
            config: Template configuration, creates default if None
        """
        self.config = config or TemplateConfig()
        self._environment: Optional[Environment] = None
        self._cache: Optional[TemplateCache] = None
        self._template_paths: List[Path] = []
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize template manager components."""
        # Setup template cache
        if self.config.enable_caching:
            self._cache = TemplateCache(max_size=self.config.cache_size)
        
        # Setup template paths
        self._setup_template_paths()
        
        # Initialize Jinja2 environment
        self._setup_environment()
    
    def _setup_template_paths(self) -> None:
        """Setup template search paths."""
        self._template_paths = []
        
        # Add configured template directories
        for template_dir in self.config.template_dirs:
            if template_dir.exists():
                self._template_paths.append(template_dir)
        
        # Add default template directory (relative to this module)
        default_template_dir = Path(__file__).parent / "templates"
        if default_template_dir.exists():
            self._template_paths.append(default_template_dir)
        
        # Ensure we have at least one template path
        if not self._template_paths:
            raise CodeGenerationError(
                "No valid template directories found",
                template_name="system",
                suggestion="Ensure template directories exist or create default templates"
            )
    
    def _setup_environment(self) -> None:
        """Setup Jinja2 environment with optimizations."""
        try:
            # Create file system loader for template paths
            loader = FileSystemLoader([str(path) for path in self._template_paths])
            
            # Create environment with optimized settings
            self._environment = Environment(
                loader=loader,
                autoescape=select_autoescape(['html', 'xml']),
                undefined=StrictUndefined,
                trim_blocks=self.config.trim_blocks,
                lstrip_blocks=self.config.lstrip_blocks,
                keep_trailing_newline=self.config.keep_trailing_newline,
                # Performance optimizations
                cache_size=self.config.cache_size if self.config.enable_caching else 0,
                auto_reload=False,  # Disable auto-reload for performance
                optimized=True
            )
            
            # Add custom filters and functions
            self._add_custom_filters()
            
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to initialize Jinja2 environment: {e}",
                template_name="system",
                suggestion="Check template directory permissions and Jinja2 installation"
            )
    
    def _add_custom_filters(self) -> None:
        """Add custom Jinja2 filters for hardware generation."""
        if not self._environment:
            return
        
        # Filter to convert snake_case to CamelCase
        def to_camel_case(text: str) -> str:
            return ''.join(word.capitalize() for word in text.split('_'))
        
        # Filter to convert CamelCase to snake_case
        def to_snake_case(text: str) -> str:
            import re
            # Insert underscore before uppercase letters
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
        # Filter to format bit width
        def format_bit_width(width: Optional[int]) -> str:
            if width is None or width <= 1:
                return ""
            return f"[{width-1}:0]"
        
        # Filter to format parameter value
        def format_parameter(value: Any) -> str:
            if isinstance(value, str):
                return f'"{value}"'
            elif isinstance(value, bool):
                return "true" if value else "false"
            else:
                return str(value)
        
        # Register filters
        self._environment.filters.update({
            'camelcase': to_camel_case,
            'snakecase': to_snake_case,
            'bitwidth': format_bit_width,
            'parameter': format_parameter
        })
        
        # Add global functions
        def debug_info(msg: str) -> str:
            """Add debug information to generated code."""
            return f"# DEBUG: {msg}"
        
        def timestamp() -> str:
            """Get current timestamp for generated code."""
            import datetime
            return datetime.datetime.now().isoformat()
        
        self._environment.globals.update({
            'debug_info': debug_info,
            'timestamp': timestamp
        })
    
    def get_template(self, template_name: str) -> Template:
        """Get template by name with caching.
        
        Args:
            template_name: Name of template file
            
        Returns:
            Compiled Jinja2 template
            
        Raises:
            CodeGenerationError: If template not found or compilation fails
        """
        # Check for custom template override
        if template_name in self.config.template_overrides:
            template_content = self.config.template_overrides[template_name]
            return self._compile_template_string(template_content, template_name)
        
        # Check for custom template file
        if template_name in self.config.custom_templates:
            custom_path = self.config.custom_templates[template_name]
            return self._load_template_file(custom_path)
        
        # Create cache key
        cache_key = self._get_cache_key(template_name)
        
        # Try cache first
        if self._cache:
            cached_template = self._cache.get(cache_key)
            if cached_template:
                return cached_template
        
        # Load from file system
        try:
            if not self._environment:
                raise CodeGenerationError("Template environment not initialized")
            
            template = self._environment.get_template(template_name)
            
            # Cache the template
            if self._cache:
                self._cache.put(cache_key, template)
            
            return template
            
        except TemplateNotFound:
            raise CodeGenerationError(
                f"Template not found: {template_name}",
                template_name=template_name,
                suggestion=f"Check template exists in paths: {self._template_paths}"
            )
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to load template '{template_name}': {e}",
                template_name=template_name,
                suggestion="Check template syntax and file permissions"
            )
    
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render template with context data.
        
        Args:
            template_name: Name of template to render
            context: Template context data
            
        Returns:
            Rendered template content
            
        Raises:
            CodeGenerationError: If rendering fails
        """
        try:
            template = self.get_template(template_name)
            return template.render(**context)
            
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to render template '{template_name}': {e}",
                template_name=template_name,
                suggestion="Check template syntax and context data completeness"
            )
    
    def render_string(self, template_content: str, context: Dict[str, Any], name: str = "string_template") -> str:
        """Render template from string content.
        
        Args:
            template_content: Template content as string
            context: Template context data
            name: Name for error reporting
            
        Returns:
            Rendered template content
        """
        try:
            template = self._compile_template_string(template_content, name)
            return template.render(**context)
            
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to render string template '{name}': {e}",
                template_name=name,
                suggestion="Check template syntax and context data"
            )
    
    def _compile_template_string(self, content: str, name: str) -> Template:
        """Compile template from string content."""
        if not self._environment:
            raise CodeGenerationError("Template environment not initialized")
        
        try:
            return self._environment.from_string(content, template_class=Template)
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to compile template '{name}': {e}",
                template_name=name,
                suggestion="Check template syntax"
            )
    
    def _load_template_file(self, template_path: Path) -> Template:
        """Load template from file path."""
        if not template_path.exists():
            raise CodeGenerationError(
                f"Template file not found: {template_path}",
                template_name=str(template_path),
                suggestion="Check file path and permissions"
            )
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._compile_template_string(content, str(template_path))
            
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to load template file '{template_path}': {e}",
                template_name=str(template_path),
                suggestion="Check file permissions and encoding"
            )
    
    def _get_cache_key(self, template_name: str) -> str:
        """Generate cache key for template."""
        # Include template paths and modification times in key
        key_data = [template_name]
        
        for path in self._template_paths:
            template_file = path / template_name
            if template_file.exists():
                key_data.append(str(template_file.stat().st_mtime))
        
        key_string = "|".join(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        templates = set()
        
        # Add templates from file system
        for path in self._template_paths:
            if path.exists():
                for template_file in path.glob("*.j2"):
                    templates.add(template_file.name)
                for template_file in path.glob("*.jinja2"):
                    templates.add(template_file.name)
        
        # Add custom templates
        templates.update(self.config.custom_templates.keys())
        templates.update(self.config.template_overrides.keys())
        
        return sorted(list(templates))
    
    def template_exists(self, template_name: str) -> bool:
        """Check if template exists."""
        try:
            self.get_template(template_name)
            return True
        except CodeGenerationError:
            return False
    
    def clear_cache(self) -> None:
        """Clear template cache."""
        if self._cache:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get template manager statistics."""
        stats = {
            'template_paths': [str(p) for p in self._template_paths],
            'available_templates': len(self.list_templates()),
            'caching_enabled': self.config.enable_caching,
            'custom_templates': len(self.config.custom_templates),
            'template_overrides': len(self.config.template_overrides)
        }
        
        if self._cache:
            stats['cache_stats'] = self._cache.stats()
        
        return stats
    
    def reload_templates(self) -> None:
        """Reload template environment and clear cache."""
        self.clear_cache()
        self._setup_environment()


def create_template_manager(config: Optional[TemplateConfig] = None) -> TemplateManager:
    """Factory function to create template manager."""
    return TemplateManager(config)


# Global template manager instance (lazy initialization)
_global_template_manager: Optional[TemplateManager] = None


def get_global_template_manager() -> TemplateManager:
    """Get global template manager instance."""
    global _global_template_manager
    if _global_template_manager is None:
        _global_template_manager = TemplateManager()
    return _global_template_manager


def set_global_template_manager(manager: TemplateManager) -> None:
    """Set global template manager instance."""
    global _global_template_manager
    _global_template_manager = manager