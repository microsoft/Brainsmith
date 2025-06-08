"""
Tests for template management system.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from brainsmith.tools.hw_kernel_gen.template_manager import (
    TemplateCache,
    TemplateManager,
    create_template_manager,
    get_global_template_manager,
    set_global_template_manager
)
from brainsmith.tools.hw_kernel_gen.config import TemplateConfig
from brainsmith.tools.hw_kernel_gen.errors import CodeGenerationError


class TestTemplateCache:
    """Test TemplateCache class."""
    
    def test_template_cache_initialization(self):
        """Test template cache initialization."""
        cache = TemplateCache(max_size=50, max_age=1800)
        
        assert cache.max_size == 50
        assert cache.max_age == 1800
        assert cache._cache == {}
    
    def test_cache_put_and_get(self):
        """Test putting and getting templates from cache."""
        cache = TemplateCache()
        
        # Mock template object
        mock_template = MagicMock()
        mock_template.name = "test_template"
        
        # Put template in cache
        cache.put("test_key", mock_template)
        
        # Get template from cache
        retrieved = cache.get("test_key")
        
        assert retrieved is mock_template
    
    def test_cache_get_nonexistent(self):
        """Test getting non-existent template from cache."""
        cache = TemplateCache()
        
        result = cache.get("nonexistent_key")
        
        assert result is None
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = TemplateCache(max_age=1)  # 1 second expiration
        
        mock_template = MagicMock()
        cache.put("test_key", mock_template)
        
        # Template should be available immediately
        assert cache.get("test_key") is mock_template
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Template should be expired
        assert cache.get("test_key") is None
    
    def test_cache_size_limit(self):
        """Test cache size limit and eviction."""
        cache = TemplateCache(max_size=2)
        
        # Add templates up to limit
        template1 = MagicMock()
        template2 = MagicMock()
        template3 = MagicMock()
        
        cache.put("key1", template1)
        cache.put("key2", template2)
        
        # Both should be available
        assert cache.get("key1") is template1
        assert cache.get("key2") is template2
        
        # Add third template (should evict oldest)
        cache.put("key3", template3)
        
        # First template should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") is template2
        assert cache.get("key3") is template3
    
    def test_cache_lru_eviction(self):
        """Test LRU (Least Recently Used) eviction."""
        cache = TemplateCache(max_size=2)
        
        template1 = MagicMock()
        template2 = MagicMock()
        template3 = MagicMock()
        
        cache.put("key1", template1)
        cache.put("key2", template2)
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add third template (should evict key2, not key1)
        cache.put("key3", template3)
        
        assert cache.get("key1") is template1  # Still available
        assert cache.get("key2") is None       # Evicted
        assert cache.get("key3") is template3  # Available
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = TemplateCache()
        
        cache.put("key1", MagicMock())
        cache.put("key2", MagicMock())
        
        assert len(cache._cache) == 2
        
        cache.clear()
        
        assert len(cache._cache) == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = TemplateCache(max_size=10, max_age=3600)
        
        cache.put("key1", MagicMock())
        cache.put("key2", MagicMock())
        
        stats = cache.stats()
        
        assert stats['size'] == 2
        assert stats['max_size'] == 10
        assert stats['max_age'] == 3600
        assert 'cached_templates' in stats
        assert len(stats['cached_templates']) == 2


class TestTemplateManager:
    """Test TemplateManager class."""
    
    def test_template_manager_initialization(self):
        """Test template manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemplateConfig(template_dirs=[Path(tmpdir)])
            manager = TemplateManager(config)
            
            assert manager.config is config
            assert manager._environment is not None
            assert manager._cache is not None
    
    def test_template_manager_default_config(self):
        """Test template manager with default configuration."""
        # Create a temporary template directory to avoid errors
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            with patch('brainsmith.tools.hw_kernel_gen.template_manager.Path') as mock_path:
                # Mock the default template directory
                mock_default_dir = MagicMock()
                mock_default_dir.exists.return_value = True
                mock_path.return_value.__truediv__.return_value = mock_default_dir
                
                manager = TemplateManager()
                
                assert isinstance(manager.config, TemplateConfig)
    
    def test_template_paths_setup(self):
        """Test template paths setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create a test template file
            test_template = template_dir / "test.j2"
            test_template.write_text("Hello {{ name }}!")
            
            config = TemplateConfig(template_dirs=[template_dir])
            manager = TemplateManager(config)
            
            assert template_dir in manager._template_paths
    
    def test_template_manager_no_valid_paths(self):
        """Test template manager with no valid template paths."""
        # Create a temporary config without validation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config but then remove the directory to simulate invalid paths
            temp_dir = Path(tmpdir) / "nonexistent"
            config = TemplateConfig()
            config.template_dirs = [temp_dir]  # Set directly to bypass validation
            
            # Mock the default template directory to not exist
            with patch.object(Path, 'exists', return_value=False):
                with pytest.raises(CodeGenerationError) as exc_info:
                    TemplateManager(config)
                
                assert "No valid template directories found" in str(exc_info.value)
    
    @patch('brainsmith.tools.hw_kernel_gen.template_manager.FileSystemLoader')
    @patch('brainsmith.tools.hw_kernel_gen.template_manager.Environment')
    def test_jinja_environment_setup(self, mock_env, mock_loader):
        """Test Jinja2 environment setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemplateConfig(
                template_dirs=[Path(tmpdir)],
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=False
            )
            
            TemplateManager(config)
            
            # Verify loader was created with correct paths
            mock_loader.assert_called_once()
            
            # Verify environment was created with correct settings
            mock_env.assert_called_once()
            call_kwargs = mock_env.call_args[1]
            assert call_kwargs['trim_blocks'] is True
            assert call_kwargs['lstrip_blocks'] is True
            assert call_kwargs['keep_trailing_newline'] is False
    
    def test_get_template_from_filesystem(self):
        """Test getting template from file system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create test template
            test_template = template_dir / "test.j2"
            test_template.write_text("Hello {{ name }}!")
            
            config = TemplateConfig(template_dirs=[template_dir])
            manager = TemplateManager(config)
            
            template = manager.get_template("test.j2")
            
            assert template is not None
            result = template.render(name="World")
            assert result == "Hello World!"
    
    def test_get_template_not_found(self):
        """Test getting non-existent template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemplateConfig(template_dirs=[Path(tmpdir)])
            manager = TemplateManager(config)
            
            with pytest.raises(CodeGenerationError) as exc_info:
                manager.get_template("nonexistent.j2")
            
            assert "Template not found" in str(exc_info.value)
    
    def test_template_caching(self):
        """Test template caching functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create test template
            test_template = template_dir / "test.j2"
            test_template.write_text("Hello {{ name }}!")
            
            config = TemplateConfig(template_dirs=[template_dir], enable_caching=True)
            manager = TemplateManager(config)
            
            # Get template twice
            template1 = manager.get_template("test.j2")
            template2 = manager.get_template("test.j2")
            
            # Should be the same object due to caching
            assert template1 is template2
    
    def test_template_caching_disabled(self):
        """Test template manager with caching disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create test template
            test_template = template_dir / "test.j2"
            test_template.write_text("Hello {{ name }}!")
            
            config = TemplateConfig(template_dirs=[template_dir], enable_caching=False)
            manager = TemplateManager(config)
            
            assert manager._cache is None
    
    def test_custom_template_override(self):
        """Test custom template override functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemplateConfig(
                template_dirs=[Path(tmpdir)],
                template_overrides={"test.j2": "Custom {{ content }}!"}
            )
            manager = TemplateManager(config)
            
            template = manager.get_template("test.j2")
            result = template.render(content="Override")
            
            assert result == "Custom Override!"
    
    def test_custom_template_file(self):
        """Test custom template file functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create custom template file
            custom_template = template_dir / "custom.j2"
            custom_template.write_text("Custom {{ value }}!")
            
            config = TemplateConfig(
                template_dirs=[template_dir],
                custom_templates={"special": custom_template}
            )
            manager = TemplateManager(config)
            
            template = manager.get_template("special")
            result = template.render(value="Template")
            
            assert result == "Custom Template!"
    
    def test_render_template(self):
        """Test template rendering with context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create test template
            test_template = template_dir / "test.j2"
            test_template.write_text("Hello {{ name }}, you are {{ age }} years old!")
            
            config = TemplateConfig(template_dirs=[template_dir])
            manager = TemplateManager(config)
            
            result = manager.render_template("test.j2", {"name": "Alice", "age": 30})
            
            assert result == "Hello Alice, you are 30 years old!"
    
    def test_render_template_error(self):
        """Test template rendering error handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create template with undefined variable
            test_template = template_dir / "test.j2"
            test_template.write_text("Hello {{ undefined_var }}!")
            
            config = TemplateConfig(template_dirs=[template_dir])
            manager = TemplateManager(config)
            
            with pytest.raises(CodeGenerationError) as exc_info:
                manager.render_template("test.j2", {})
            
            assert "Failed to render template" in str(exc_info.value)
    
    def test_render_string(self):
        """Test rendering template from string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemplateConfig(template_dirs=[Path(tmpdir)])
            manager = TemplateManager(config)
            
            template_content = "Hello {{ name }}!"
            result = manager.render_string(template_content, {"name": "World"})
            
            assert result == "Hello World!"
    
    def test_render_string_error(self):
        """Test string template rendering error handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemplateConfig(template_dirs=[Path(tmpdir)])
            manager = TemplateManager(config)
            
            template_content = "Hello {{ undefined }}!"
            
            with pytest.raises(CodeGenerationError) as exc_info:
                manager.render_string(template_content, {})
            
            assert "Failed to render string template" in str(exc_info.value)
    
    def test_list_templates(self):
        """Test listing available templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create test templates
            (template_dir / "template1.j2").write_text("Template 1")
            (template_dir / "template2.jinja2").write_text("Template 2")
            (template_dir / "custom.j2").write_text("Custom Template")  # Create custom template file
            
            config = TemplateConfig(
                template_dirs=[template_dir],
                custom_templates={"custom": template_dir / "custom.j2"},
                template_overrides={"override": "Override content"}
            )
            manager = TemplateManager(config)
            
            templates = manager.list_templates()
            
            assert "template1.j2" in templates
            assert "template2.jinja2" in templates
            assert "custom" in templates
            assert "override" in templates
    
    def test_template_exists(self):
        """Test template existence checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create test template
            (template_dir / "existing.j2").write_text("Exists")
            
            config = TemplateConfig(template_dirs=[template_dir])
            manager = TemplateManager(config)
            
            assert manager.template_exists("existing.j2") is True
            assert manager.template_exists("nonexistent.j2") is False
    
    def test_clear_cache(self):
        """Test clearing template cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create test template
            (template_dir / "test.j2").write_text("Test")
            
            config = TemplateConfig(template_dirs=[template_dir])
            manager = TemplateManager(config)
            
            # Load template to populate cache
            manager.get_template("test.j2")
            
            # Clear cache
            manager.clear_cache()
            
            # Verify cache is empty
            if manager._cache:
                assert len(manager._cache._cache) == 0
    
    def test_get_stats(self):
        """Test getting template manager statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create custom template file
            (template_dir / "custom.j2").write_text("Custom Template")
            
            config = TemplateConfig(
                template_dirs=[template_dir],
                custom_templates={"custom": template_dir / "custom.j2"},
                template_overrides={"override": "content"}
            )
            manager = TemplateManager(config)
            
            stats = manager.get_stats()
            
            assert 'template_paths' in stats
            assert 'available_templates' in stats
            assert 'caching_enabled' in stats
            assert 'custom_templates' in stats
            assert 'template_overrides' in stats
            assert stats['caching_enabled'] is True
            assert stats['custom_templates'] == 1
            assert stats['template_overrides'] == 1
    
    def test_reload_templates(self):
        """Test reloading templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            config = TemplateConfig(template_dirs=[template_dir])
            manager = TemplateManager(config)
            
            # Should not raise an exception
            manager.reload_templates()
    
    def test_custom_filters(self):
        """Test custom Jinja2 filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create template using custom filters
            test_template = template_dir / "test.j2"
            test_template.write_text("""
            CamelCase: {{ 'snake_case_text' | camelcase }}
            SnakeCase: {{ 'CamelCaseText' | snakecase }}
            BitWidth: {{ 8 | bitwidth }}
            Parameter: {{ 'test_value' | parameter }}
            """)
            
            config = TemplateConfig(template_dirs=[template_dir])
            manager = TemplateManager(config)
            
            result = manager.render_template("test.j2", {})
            
            assert "SnakeCaseText" in result
            assert "camel_case_text" in result
            assert "[7:0]" in result
            assert '"test_value"' in result


class TestTemplateManagerHelpers:
    """Test template manager helper functions."""
    
    def test_create_template_manager(self):
        """Test create_template_manager factory function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemplateConfig(template_dirs=[Path(tmpdir)])
            manager = create_template_manager(config)
            
            assert isinstance(manager, TemplateManager)
            assert manager.config is config
    
    def test_create_template_manager_default(self):
        """Test create_template_manager with default config."""
        # Mock the default template directory
        with patch.object(Path, 'exists', return_value=True):
            manager = create_template_manager()
            
            assert isinstance(manager, TemplateManager)
            assert isinstance(manager.config, TemplateConfig)
    
    def test_global_template_manager(self):
        """Test global template manager functionality."""
        # Reset global manager
        set_global_template_manager(None)
        
        # First call should create new manager
        manager1 = get_global_template_manager()
        assert isinstance(manager1, TemplateManager)
        
        # Second call should return same manager
        manager2 = get_global_template_manager()
        assert manager1 is manager2
    
    def test_set_global_template_manager(self):
        """Test setting global template manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemplateConfig(template_dirs=[Path(tmpdir)])
            custom_manager = TemplateManager(config)
            
            set_global_template_manager(custom_manager)
            
            retrieved_manager = get_global_template_manager()
            assert retrieved_manager is custom_manager


class TestTemplateManagerIntegration:
    """Integration tests for template manager."""
    
    def test_end_to_end_template_rendering(self):
        """Test complete template rendering workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create a realistic hardware template
            hw_template = template_dir / "hw_custom_op.py.j2"
            hw_template.write_text("""
class {{ class_name }}(CustomOp):
    \"\"\"{{ description }}\"\"\"
    
    def __init__(self):
        super().__init__()
        self.onnx_op_type = "{{ onnx_op_type }}"
        
    {% for interface in interfaces -%}
    # Interface: {{ interface.name }} ({{ interface.direction }})
    {% endfor %}
    
    {% if include_debug_info -%}
    # DEBUG: Generated at {{ timestamp() }}
    {% endif %}
            """.strip())
            
            config = TemplateConfig(template_dirs=[template_dir])
            manager = TemplateManager(config)
            
            context = {
                'class_name': 'MyCustomOp',
                'description': 'Custom hardware operation',
                'onnx_op_type': 'MyOp',
                'interfaces': [
                    {'name': 'data_in', 'direction': 'input'},
                    {'name': 'data_out', 'direction': 'output'}
                ],
                'include_debug_info': True
            }
            
            result = manager.render_template("hw_custom_op.py.j2", context)
            
            assert "class MyCustomOp(CustomOp):" in result
            assert "Custom hardware operation" in result
            assert "MyOp" in result
            assert "Interface: data_in (input)" in result
            assert "Interface: data_out (output)" in result
            assert "DEBUG: Generated at" in result