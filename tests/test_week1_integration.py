"""
Week 1 Integration Tests for HWKG Phase 2 Refactoring.

This module tests the integration of all Week 1 components:
- Configuration framework
- Template context and template manager
- Generator base interface and data structures
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import all Week 1 components
from brainsmith.tools.hw_kernel_gen.config import (
    PipelineConfig, TemplateConfig, GenerationConfig, 
    GeneratorType, create_default_config
)
from brainsmith.tools.hw_kernel_gen.template_context import (
    HWCustomOpContext, TemplateContextBuilder, InterfaceInfo, ParameterInfo
)
from brainsmith.tools.hw_kernel_gen.template_manager import TemplateManager, create_template_manager
from brainsmith.tools.hw_kernel_gen.generator_base import (
    GeneratorBase, GenerationStatus, ArtifactType, GeneratedArtifact, GenerationResult
)
from brainsmith.tools.hw_kernel_gen.data_structures import (
    PipelineInputs, PipelineResults, PipelineStage, create_pipeline_inputs, create_pipeline_results
)
from brainsmith.tools.hw_kernel_gen.errors import CodeGenerationError, ValidationError


class TestWeek1Integration:
    """Integration tests for Week 1 components."""
    
    def test_complete_pipeline_configuration(self):
        """Test complete pipeline configuration with all components."""
        # Create configuration
        config = create_default_config(GeneratorType.HW_CUSTOM_OP)
        
        # Verify configuration is properly structured
        assert isinstance(config.template, TemplateConfig)
        assert isinstance(config.generation, GenerationConfig)
        assert config.generator_type == GeneratorType.HW_CUSTOM_OP
        
        # Test configuration serialization
        config_dict = config.to_dict()
        assert config_dict['generator_type'] == 'hw_custom_op'
        
        # Test configuration reconstruction
        config2 = PipelineConfig.from_dict(config_dict)
        assert config2.generator_type == config.generator_type
    
    def test_template_system_integration(self):
        """Test template manager and context builder integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create a test template
            test_template = template_dir / "hw_custom_op.py.j2"
            test_template.write_text("""
class {{ class_name }}(CustomOp):
    \"\"\"Generated hardware custom operation for {{ module_name }}\"\"\"
    
    def __init__(self):
        super().__init__()
        self.onnx_op_type = "{{ module_name | camelcase }}"
        
        # Interfaces: {{ interfaces|length }}
        {% for interface in interfaces -%}
        # - {{ interface.name }} ({{ interface.direction }})
        {% endfor %}
        
        # Parameters: {{ parameters|length }}
        {% for param in parameters -%}
        # - {{ param.name }}: {{ param.value }}
        {% endfor %}
            """.strip())
            
            # Create template configuration and manager
            template_config = TemplateConfig(template_dirs=[template_dir])
            template_manager = TemplateManager(template_config)
            
            # Create context builder
            context_builder = TemplateContextBuilder()
            
            # Build context from analysis data
            analysis_data = {
                'module_name': 'test_op',
                'top_module': 'test_top',
                'rtl_file': 'test.sv',
                'interfaces': [
                    {'name': 'ap_clk', 'direction': 'input'},
                    {'name': 'ap_rst_n', 'direction': 'input'},
                    {'name': 's_axis_input_tdata', 'direction': 'input', 'width': 32},
                    {'name': 'm_axis_output_tdata', 'direction': 'output', 'width': 32}
                ],
                'parameters': {
                    'DATA_WIDTH': 32,
                    'BUFFER_DEPTH': {'value': 1024, 'type': 'int'}
                }
            }
            
            context = context_builder.build_hw_custom_op_context(analysis_data)
            
            # Verify context was built correctly
            assert context.module_name == 'test_op'
            assert context.top_module_name == 'test_top'
            assert len(context.interfaces) == 4
            assert len(context.parameters) == 2
            
            # Render template with context
            rendered = template_manager.render_template("hw_custom_op.py.j2", context.to_dict())
            
            # Verify template was rendered correctly
            assert "class TestTop(CustomOp):" in rendered
            assert "Interfaces: 4" in rendered
            assert "Parameters: 2" in rendered
            assert "ap_clk (input)" in rendered
            assert "s_axis_input_tdata (input)" in rendered
            assert "Generated hardware custom operation for test_op" in rendered
    
    def test_generator_implementation_with_all_components(self):
        """Test a complete generator implementation using all Week 1 components."""
        
        class IntegratedTestGenerator(GeneratorBase):
            """Test generator that uses all Week 1 components."""
            
            def __init__(self, config, template_manager, context_builder):
                super().__init__(config, template_manager, context_builder)
                self.generator_type = "IntegratedTestGenerator"
            
            def generate(self, analysis_data, **kwargs):
                self.start_generation_timer()
                
                # Validate input using base validation
                self.validate_input(analysis_data)
                
                # Build context using context builder
                context = self.context_builder.build_hw_custom_op_context(analysis_data)
                
                # Create result
                result = self.create_result(GenerationStatus.SUCCESS, "Integration test successful")
                
                # Create Python artifact using template
                if self.template_manager.template_exists("test_template.j2"):
                    python_content = self.render_template("test_template.j2", context)
                else:
                    # Fallback content
                    python_content = f"""
# Generated by {self.generator_type}
class {context.class_name}:
    def __init__(self):
        self.module_name = "{context.module_name}"
        self.interfaces = {len(context.interfaces)}
        self.parameters = {len(context.parameters)}
"""
                
                python_artifact = self.create_artifact(
                    context.module_name,
                    ArtifactType.PYTHON_FILE,
                    python_content.strip(),
                    "test_template.j2",
                    context.to_dict()
                )
                result.add_artifact(python_artifact)
                
                # Create configuration artifact
                config_content = f'{{"module": "{context.module_name}", "generated_at": "{context.generated_timestamp}"}}'
                config_artifact = self.create_artifact(
                    f"{context.module_name}_config",
                    ArtifactType.CONFIG_FILE,
                    config_content
                )
                result.add_artifact(config_artifact)
                
                return result
            
            def get_supported_templates(self):
                return ["test_template.j2", "config_template.j2"]
            
            def validate_input(self, analysis_data):
                if not analysis_data.get('module_name'):
                    raise ValidationError("Module name is required")
                if not analysis_data.get('top_module'):
                    raise ValidationError("Top module name is required")
        
        # Setup components
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create template file
            template_file = template_dir / "test_template.j2"
            template_file.write_text("""
# Generated {{ class_name }}
class {{ class_name }}:
    \"\"\"Generated from {{ top_module_name }}\"\"\"
    
    def __init__(self):
        self.module_name = "{{ module_name }}"
        self.interface_count = {{ interfaces|length }}
        self.parameter_count = {{ parameters|length }}
        
        # Control signals
        self.control_interfaces = [
            {% for iface in interfaces if iface.is_control or iface.is_clock or iface.is_reset -%}
            "{{ iface.name }}",
            {% endfor %}
        ]
        
        # Data interfaces
        self.data_interfaces = [
            {% for iface in interfaces if not (iface.is_control or iface.is_clock or iface.is_reset) -%}
            "{{ iface.name }}",
            {% endfor %}
        ]
            """.strip())
            
            # Create configuration
            config = PipelineConfig()
            config.template = TemplateConfig(template_dirs=[template_dir])
            config.generation.include_debug_info = True
            
            # Create template manager and context builder
            template_manager = TemplateManager(config.template)
            context_builder = TemplateContextBuilder(config)
            
            # Create generator
            generator = IntegratedTestGenerator(config, template_manager, context_builder)
            
            # Test analysis data
            analysis_data = {
                'module_name': 'advanced_filter',
                'top_module': 'advanced_filter_top',
                'rtl_file': 'advanced_filter.sv',
                'interfaces': [
                    {'name': 'ap_clk', 'direction': 'input'},
                    {'name': 'ap_rst_n', 'direction': 'input'},
                    {'name': 'ap_start', 'direction': 'input'},
                    {'name': 'ap_done', 'direction': 'output'},
                    {'name': 's_axis_input_tdata', 'direction': 'input', 'width': 64},
                    {'name': 's_axis_input_tvalid', 'direction': 'input'},
                    {'name': 's_axis_input_tready', 'direction': 'output'},
                    {'name': 'm_axis_output_tdata', 'direction': 'output', 'width': 64},
                    {'name': 'm_axis_output_tvalid', 'direction': 'output'},
                    {'name': 'm_axis_output_tready', 'direction': 'input'}
                ],
                'parameters': {
                    'DATA_WIDTH': 64,
                    'FILTER_TAPS': 16,
                    'PRECISION': {'value': 'Q8.8', 'type': 'string', 'description': 'Fixed point format'}
                }
            }
            
            # Generate code
            result = generator.generate(analysis_data)
            
            # Verify generation was successful
            assert result.status == GenerationStatus.SUCCESS
            assert len(result.artifacts) == 2
            assert result.generation_time > 0
            
            # Check Python artifact
            python_artifacts = result.get_artifacts_by_type(ArtifactType.PYTHON_FILE)
            assert len(python_artifacts) == 1
            
            python_artifact = python_artifacts[0]
            assert python_artifact.name == 'advanced_filter'
            assert 'class AdvancedFilterTop:' in python_artifact.content
            assert 'interface_count = 10' in python_artifact.content
            assert 'parameter_count = 3' in python_artifact.content
            assert python_artifact.is_valid
            
            # Check config artifact
            config_artifacts = result.get_artifacts_by_type(ArtifactType.CONFIG_FILE)
            assert len(config_artifacts) == 1
            
            config_artifact = config_artifacts[0]
            assert config_artifact.name == 'advanced_filter_config'
            assert '"module": "advanced_filter"' in config_artifact.content
            assert config_artifact.is_valid
            
            # Test file writing
            with tempfile.TemporaryDirectory() as output_tmpdir:
                output_dir = Path(output_tmpdir)
                written_files = result.write_all_artifacts(output_dir, overwrite=True)
                
                assert len(written_files) == 2
                assert (output_dir / "advanced_filter.py").exists()
                assert (output_dir / "advanced_filter_config.json").exists()
    
    def test_pipeline_data_flow(self):
        """Test complete pipeline data flow using all data structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test RTL file
            rtl_file = Path(tmpdir) / "test_module.sv"
            rtl_file.write_text("""
module test_module #(
    parameter DATA_WIDTH = 32
) (
    input  wire ap_clk,
    input  wire ap_rst_n,
    input  wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input  wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input  wire m_axis_output_tready
);
endmodule
            """)
            
            # Create pipeline inputs
            inputs = create_pipeline_inputs(
                rtl_files=[rtl_file],
                module_name="test_module",
                generator_type="hw_custom_op",
                output_directory=Path(tmpdir) / "output",
                generate_testbench=True,
                analyze_interfaces=True
            )
            
            # Validate inputs
            inputs.validate()  # Should not raise
            
            # Create pipeline results
            results = create_pipeline_results("test-pipeline", inputs)
            
            # Simulate pipeline execution stages
            start_time = time.time()
            
            # Stage 1: RTL Parsing (simulated)
            results.mark_stage_complete(PipelineStage.RTL_PARSING, 0.5)
            
            # Stage 2: Interface Analysis (simulated)
            results.mark_stage_complete(PipelineStage.INTERFACE_ANALYSIS, 0.3)
            
            # Stage 3: Code Generation (simulated)
            results.mark_stage_complete(PipelineStage.CODE_GENERATION, 1.2)
            
            # Add some generated files
            output_dir = inputs.output_directory
            output_dir.mkdir(exist_ok=True)
            
            test_file1 = output_dir / "test_module.py"
            test_file1.write_text("# Generated Python file")
            results.generated_files.append(test_file1)
            
            test_file2 = output_dir / "test_module_config.json"
            test_file2.write_text('{"module": "test_module"}')
            results.generated_files.append(test_file2)
            
            # Update metrics
            results.lines_generated = 150
            results.templates_used = 3
            
            # Add a warning
            results.add_warning("No timing constraints found")
            
            # Finalize results
            results.finalize()
            
            # Verify results
            assert results.status == "completed_with_warnings"
            assert results.pipeline_id == "test-pipeline"
            assert len(results.stage_times) == 3
            assert results.total_time > 0
            assert results.files_generated == 2
            assert results.lines_generated == 150
            assert results.has_warnings()
            assert not results.has_errors()
            
            # Test results summary
            summary = results.get_summary()
            assert summary['status'] == "completed_with_warnings"
            assert summary['files_generated'] == 2
            assert summary['error_count'] == 0
            assert summary['warning_count'] == 1
    
    def test_error_handling_integration(self):
        """Test error handling across all Week 1 components."""
        # Test configuration validation error
        with pytest.raises(ValidationError):
            PipelineInputs(
                rtl_files=[Path("/nonexistent.sv")],
                module_name=""
            ).validate()
        
        # Test template context validation error
        context = HWCustomOpContext(
            module_name="test",
            file_name="test.py"
            # Missing top_module_name and interfaces
        )
        
        with pytest.raises(ValidationError):
            context.validate()
        
        # Test generator validation error
        class TestGenerator(GeneratorBase):
            def generate(self, analysis_data, **kwargs):
                return self.create_result(GenerationStatus.SUCCESS)
            
            def get_supported_templates(self):
                return []
            
            def validate_input(self, analysis_data):
                if not analysis_data.get('required_field'):
                    raise ValidationError("Required field missing")
        
        generator = TestGenerator()
        
        with pytest.raises(ValidationError):
            generator.validate_input({})
        
        # Test template manager error
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemplateConfig(template_dirs=[Path(tmpdir)])
            manager = TemplateManager(config)
            
            with pytest.raises(CodeGenerationError):
                manager.get_template("nonexistent.j2")
    
    def test_performance_baseline(self):
        """Test performance baseline for Week 1 components."""
        # Test configuration creation performance
        start_time = time.time()
        for _ in range(100):
            config = create_default_config(GeneratorType.HW_CUSTOM_OP)
        config_time = time.time() - start_time
        
        assert config_time < 1.0  # Should be very fast
        
        # Test context building performance
        analysis_data = {
            'module_name': 'perf_test',
            'top_module': 'perf_test_top',
            'rtl_file': 'test.sv',
            'interfaces': [
                {'name': 'ap_clk', 'direction': 'input'},
                {'name': 'ap_rst_n', 'direction': 'input'}
            ] + [
                {'name': f'signal_{i}', 'direction': 'input', 'width': 32}
                for i in range(48)  # 48 data interfaces + 2 control = 50 total
            ],
            'parameters': {f'PARAM_{i}': i for i in range(20)}  # 20 parameters
        }
        
        builder = TemplateContextBuilder()
        
        start_time = time.time()
        for _ in range(10):
            context = builder.build_hw_custom_op_context(analysis_data)
        context_time = time.time() - start_time
        
        assert context_time < 2.0  # Should handle moderate complexity quickly
        
        # Test caching effectiveness
        start_time = time.time()
        for _ in range(100):
            context = builder.build_hw_custom_op_context(analysis_data)
        cached_time = time.time() - start_time
        
        assert cached_time < 0.5  # Cached access should be very fast


class TestWeek1ComponentInteraction:
    """Test specific interactions between Week 1 components."""
    
    def test_config_template_manager_integration(self):
        """Test configuration and template manager integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            # Create configuration with template settings
            config = PipelineConfig()
            config.template.template_dirs = [template_dir]
            config.template.enable_caching = True
            config.template.cache_size = 50
            config.template.trim_blocks = True
            
            # Create template manager from configuration
            manager = TemplateManager(config.template)
            
            # Verify configuration was applied
            assert template_dir in manager._template_paths
            assert manager._cache is not None
            assert manager._cache.max_size == 50
    
    def test_context_generator_integration(self):
        """Test context builder and generator integration."""
        class TestGenerator(GeneratorBase):
            def generate(self, analysis_data, **kwargs):
                # Use inherited context building
                context = self.build_hw_context(analysis_data)
                return self.create_result(GenerationStatus.SUCCESS)
            
            def build_hw_context(self, analysis_data):
                if not self.context_builder:
                    raise CodeGenerationError("Context builder required")
                return self.context_builder.build_hw_custom_op_context(analysis_data)
            
            def get_supported_templates(self):
                return ["test.j2"]
            
            def validate_input(self, analysis_data):
                pass
        
        # Test without context builder
        generator = TestGenerator()
        
        with pytest.raises(CodeGenerationError):
            generator.build_hw_context({})
        
        # Test with context builder
        builder = TemplateContextBuilder()
        generator = TestGenerator(context_builder=builder)
        
        analysis_data = {
            'module_name': 'test',
            'top_module': 'test_top',
            'rtl_file': 'test.sv',
            'interfaces': [
                {'name': 'ap_clk', 'direction': 'input'},
                {'name': 'ap_rst_n', 'direction': 'input'}
            ],
            'parameters': {}
        }
        
        context = generator.build_hw_context(analysis_data)
        assert isinstance(context, HWCustomOpContext)
        assert context.module_name == 'test'
    
    def test_end_to_end_validation(self):
        """Test end-to-end validation across all components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup complete environment
            template_dir = Path(tmpdir)
            output_dir = Path(tmpdir) / "output"
            rtl_file = Path(tmpdir) / "module.sv"
            
            # Create test files
            rtl_file.write_text("module test(); endmodule")
            
            simple_template = template_dir / "simple.j2"
            simple_template.write_text("# {{ module_name }} generated")
            
            # Create pipeline inputs
            inputs = create_pipeline_inputs(
                rtl_files=[rtl_file],
                module_name="test_module",
                output_directory=output_dir
            )
            
            # Create configuration
            config = PipelineConfig()
            config.template.template_dirs = [template_dir]
            config.generation.output_dir = output_dir
            config.generation.create_directories = True
            
            # Create components
            template_manager = TemplateManager(config.template)
            context_builder = TemplateContextBuilder(config)
            
            # Test complete validation chain
            inputs.validate()  # Input validation
            
            context = context_builder.build_hw_custom_op_context({
                'module_name': 'test_module',
                'top_module': 'test_top',
                'rtl_file': str(rtl_file),
                'interfaces': [
                    {'name': 'ap_clk', 'direction': 'input'},
                    {'name': 'ap_rst_n', 'direction': 'input'}
                ],
                'parameters': {}
            })
            context.validate()  # Context validation
            
            rendered = template_manager.render_template("simple.j2", context.to_dict())
            assert "test_module generated" in rendered
            
            # Create and validate artifact
            artifact = GeneratedArtifact(
                "test_module",
                ArtifactType.PYTHON_FILE,
                rendered
            )
            assert artifact.validate_content()
            
            # Create and validate result
            result = GenerationResult(GenerationStatus.SUCCESS)
            result.add_artifact(artifact)
            assert result.validate_all_artifacts()
            
            # Test file writing
            written_files = result.write_all_artifacts(output_dir, overwrite=True)
            assert len(written_files) == 1
            assert written_files[0].exists()


def test_week1_success_criteria():
    """Test that all Week 1 success criteria are met."""
    # Test 1: Configuration framework exists and works
    config = create_default_config()
    assert config is not None
    config.validate()
    
    # Test 2: Template system exists and works
    with tempfile.TemporaryDirectory() as tmpdir:
        template_dir = Path(tmpdir)
        test_template = template_dir / "test.j2"
        test_template.write_text("Hello {{ name }}!")
        
        template_config = TemplateConfig(template_dirs=[template_dir])
        manager = TemplateManager(template_config)
        
        result = manager.render_template("test.j2", {"name": "World"})
        assert result == "Hello World!"
    
    # Test 3: Context building exists and works
    builder = TemplateContextBuilder()
    context = builder.build_hw_custom_op_context({
        'module_name': 'test',
        'top_module': 'test_top',
        'rtl_file': 'test.sv',
        'interfaces': [
            {'name': 'ap_clk', 'direction': 'input'},
            {'name': 'ap_rst_n', 'direction': 'input'}
        ],
        'parameters': {}
    })
    assert context.module_name == 'test'
    
    # Test 4: Generator base interface exists
    class TestGen(GeneratorBase):
        def generate(self, analysis_data, **kwargs):
            return self.create_result(GenerationStatus.SUCCESS)
        def get_supported_templates(self):
            return []
        def validate_input(self, analysis_data):
            pass
    
    gen = TestGen()
    assert gen.generator_type == "TestGen"
    
    # Test 5: Data structures exist and work
    inputs = create_pipeline_inputs([Path("test.sv")], "test")
    results = create_pipeline_results("test", inputs)
    assert results.pipeline_id == "test"
    
    print("✅ All Week 1 success criteria met!")


if __name__ == "__main__":
    test_week1_success_criteria()