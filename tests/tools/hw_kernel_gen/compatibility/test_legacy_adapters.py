"""
Test suite for Week 4 Compatibility Layer.

Tests the legacy adapters, migration utilities, and backward compatibility
features that allow existing code to work with the new architecture.
"""

import pytest
import warnings
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig, GeneratorType
from brainsmith.tools.hw_kernel_gen.enhanced_data_structures import RTLModule, RTLInterface, RTLSignal
from brainsmith.tools.hw_kernel_gen.enhanced_generator_base import GenerationResult, GeneratedArtifact
from brainsmith.tools.hw_kernel_gen.compatibility.legacy_adapter import (
    LegacyGeneratorAdapter, HWCustomOpLegacyAdapter, RTLTemplateLegacyAdapter,
    LegacyGeneratorFactory, create_legacy_adapter
)
from brainsmith.tools.hw_kernel_gen.compatibility.backward_compatibility import (
    LegacyAPILayer, LegacyHardwareKernelGenerator, LegacyConfigurationWrapper,
    generate_hwcustomop, generate_rtl_backend, generate_rtl_template,
    enable_legacy_warnings, get_migration_status
)
from brainsmith.tools.hw_kernel_gen.migration.migration_utilities import (
    ConfigurationMigrator, WorkflowMigrator, DataStructureMigrator,
    MigrationReport, MigrationStatus, migrate_configuration, analyze_workflow
)


class TestLegacyAdapters:
    """Test legacy generator adapters."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.rtl_module = self._create_test_rtl_module()
    
    def _create_test_rtl_module(self) -> RTLModule:
        """Create test RTL module."""
        signals = [
            RTLSignal("data_in", "input", 32),
            RTLSignal("data_out", "output", 32),
            RTLSignal("valid_in", "input", 1),
            RTLSignal("valid_out", "output", 1)
        ]
        interface = RTLInterface("data_interface", "custom", signals)
        
        return RTLModule(
            name="test_module",
            interfaces=[interface],
            parameters={"WIDTH": 32}
        )
    
    def test_legacy_generator_factory(self):
        """Test legacy generator factory."""
        # Test HW Custom Op adapter creation
        hw_adapter = LegacyGeneratorFactory.create_adapter(
            GeneratorType.HW_CUSTOM_OP, self.config
        )
        assert isinstance(hw_adapter, HWCustomOpLegacyAdapter)
        
        # Test RTL Backend adapter creation
        rtl_adapter = LegacyGeneratorFactory.create_adapter(
            GeneratorType.RTL_BACKEND, self.config
        )
        assert isinstance(rtl_adapter, RTLTemplateLegacyAdapter)
        
        # Test unavailable adapter
        doc_adapter = LegacyGeneratorFactory.create_adapter(
            GeneratorType.DOCUMENTATION, self.config
        )
        assert doc_adapter is None
    
    def test_create_legacy_adapter_function(self):
        """Test convenience function for creating adapters."""
        # Test with GeneratorType enum
        adapter = create_legacy_adapter(GeneratorType.HW_CUSTOM_OP, self.config)
        assert isinstance(adapter, HWCustomOpLegacyAdapter)
        
        # Test with string
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter = create_legacy_adapter("hw_custom_op", self.config)
            assert isinstance(adapter, HWCustomOpLegacyAdapter)
            assert len(w) == 1
            assert "legacy adapter" in str(w[0].message)
        
        # Test with invalid string
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter = create_legacy_adapter("invalid_type", self.config)
            assert adapter is None
            assert len(w) == 1
            assert "Unknown generator type" in str(w[0].message)
    
    def test_get_available_legacy_adapters(self):
        """Test getting available legacy adapters."""
        available = LegacyGeneratorFactory.get_available_legacy_adapters()
        
        assert GeneratorType.HW_CUSTOM_OP in available
        assert GeneratorType.AUTO_HW_CUSTOM_OP in available
        assert GeneratorType.RTL_BACKEND in available
        assert GeneratorType.AUTO_RTL_BACKEND in available


class TestHWCustomOpLegacyAdapter:
    """Test HW Custom Op legacy adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.adapter = HWCustomOpLegacyAdapter(self.config)
        self.rtl_module = self._create_test_rtl_module()
    
    def _create_test_rtl_module(self) -> RTLModule:
        """Create test RTL module."""
        signals = [RTLSignal("data", "input", 32)]
        interface = RTLInterface("test_if", "custom", signals)
        return RTLModule("test", [interface], {"WIDTH": 32})
    
    def test_adapter_properties(self):
        """Test adapter properties."""
        assert self.adapter.get_template_name() == "hw_custom_op_slim.py.j2"
        assert self.adapter.get_artifact_type() == "hwcustomop"
    
    def test_convert_inputs(self):
        """Test input conversion."""
        inputs = {
            "rtl_module": self.rtl_module,
            "class_name": "TestClass",
            "source_file": "test.sv"
        }
        
        legacy_inputs = self.adapter._convert_inputs(inputs)
        
        assert "hw_kernel" in legacy_inputs
        assert legacy_inputs["class_name"] == "TestClass"
        assert legacy_inputs["source_file"] == "test.sv"
        assert "output_path" in legacy_inputs
    
    def test_convert_outputs(self):
        """Test output conversion."""
        legacy_result = "class TestClass:\n    pass"
        
        result = self.adapter._convert_outputs(legacy_result)
        
        assert isinstance(result, GenerationResult)
        assert result.success
        assert len(result.artifacts) == 1
        assert result.artifacts[0].content == legacy_result
        assert result.artifacts[0].artifact_type == "hwcustomop"
        assert result.artifacts[0].metadata["adapter_used"]
    
    def test_rtl_module_to_hw_kernel_conversion(self):
        """Test RTL module to HW kernel conversion."""
        hw_kernel = self.adapter._rtl_module_to_hw_kernel(self.rtl_module)
        
        # Should handle the conversion gracefully
        assert hw_kernel is not None
    
    @patch('brainsmith.tools.hw_kernel_gen.compatibility.legacy_adapter.HWCustomOpLegacyAdapter._create_legacy_generator')
    @patch('brainsmith.tools.hw_kernel_gen.compatibility.legacy_adapter.HWCustomOpLegacyAdapter._call_legacy_generator')
    def test_generate_success(self, mock_call, mock_create):
        """Test successful generation through adapter."""
        # Mock legacy generator
        mock_generator = Mock()
        mock_create.return_value = mock_generator
        mock_call.return_value = "class TestClass:\n    pass"
        
        inputs = {
            "rtl_module": self.rtl_module,
            "class_name": "TestClass"
        }
        
        result = self.adapter.generate(inputs)
        
        assert result.success
        assert len(result.artifacts) == 1
        assert result.artifacts[0].content == "class TestClass:\n    pass"
        mock_create.assert_called_once()
        mock_call.assert_called_once()
    
    @patch('brainsmith.tools.hw_kernel_gen.compatibility.legacy_adapter.HWCustomOpLegacyAdapter._create_legacy_generator')
    def test_generate_failure(self, mock_create):
        """Test generation failure handling."""
        # Mock generator creation failure
        mock_create.side_effect = Exception("Legacy generator not available")
        
        inputs = {"rtl_module": self.rtl_module}
        
        result = self.adapter.generate(inputs)
        
        assert not result.success
        assert len(result.errors) > 0
        assert "Legacy generator failed" in result.errors[0]


class TestMigrationUtilities:
    """Test migration utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.migrator = ConfigurationMigrator()
        self.workflow_migrator = WorkflowMigrator()
        self.data_migrator = DataStructureMigrator()
    
    def test_configuration_migrator_basic(self):
        """Test basic configuration migration."""
        legacy_config = {
            "template": {
                "template_dirs": ["/path/to/templates"],
                "enable_caching": True,
                "cache_size": 50
            },
            "generation": {
                "output_dir": "/path/to/output",
                "include_debug_info": True,
                "enabled_generators": ["hwcustomop", "rtlbackend"]
            },
            "generator_type": "hw_custom_op",
            "verbose": True
        }
        
        new_config, report = self.migrator.migrate_legacy_config(legacy_config)
        
        assert isinstance(new_config, PipelineConfig)
        assert report.status in [MigrationStatus.COMPLETED, MigrationStatus.PARTIAL]
        assert len(new_config.template.template_dirs) == 1
        assert new_config.template.enable_caching == True
        assert new_config.template.cache_size == 50
        assert new_config.generation.output_dir == Path("/path/to/output")
        assert new_config.generation.include_debug_info == True
        assert "hwcustomop" in new_config.generation.enabled_generators
        assert new_config.generator_type == GeneratorType.HW_CUSTOM_OP
        assert new_config.verbose == True
    
    def test_configuration_migrator_with_errors(self):
        """Test configuration migration with invalid data."""
        legacy_config = {
            "invalid_section": {"bad_data": "value"},
            "generator_type": "unknown_type"
        }
        
        new_config, report = self.migrator.migrate_legacy_config(legacy_config)
        
        assert isinstance(new_config, PipelineConfig)
        assert len(report.warnings) > 0  # Should have warnings about unknown type
    
    def test_configuration_migrator_save(self):
        """Test saving migrated configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "migrated_config.json"
            
            legacy_config = {
                "template": {"template_dirs": ["/tmp"]},
                "generator_type": "hw_custom_op"
            }
            
            new_config, report = self.migrator.migrate_legacy_config(
                legacy_config, output_path
            )
            
            assert output_path.exists()
            assert report.status == MigrationStatus.COMPLETED
            
            # Verify saved content
            with open(output_path, 'r') as f:
                saved_config = json.load(f)
            assert "template" in saved_config
            assert "generator_type" in saved_config
    
    def test_workflow_analyzer(self):
        """Test workflow analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_path = Path(temp_dir) / "legacy_workflow.py"
            
            # Create sample legacy workflow
            legacy_code = '''
import os
from brainsmith.generators.hw_custom_op_generator import HWCustomOpGenerator
from brainsmith.generators import generate_rtl_template

def legacy_workflow():
    generator = HWCustomOpGenerator(template_dir="templates")
    result = generator.generate_hwcustomop(hw_kernel, output_path)
    rtl_path = generate_rtl_template(hw_data, output_dir)
    return result, rtl_path
'''
            workflow_path.write_text(legacy_code)
            
            analysis = self.workflow_migrator.analyze_legacy_workflow(workflow_path)
            
            assert analysis["migration_required"]
            assert analysis["complexity"] in ["low", "medium", "high"]
            assert len(analysis["legacy_patterns"]) > 0
            assert len(analysis["recommendations"]) > 0
            
            # Should detect legacy patterns
            pattern_names = [p["pattern"] for p in analysis["legacy_patterns"]]
            assert "HWCustomOpGenerator" in pattern_names
            assert "generate_rtl_template" in pattern_names
    
    def test_migration_template_creation(self):
        """Test migration template creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = Path(temp_dir) / "legacy.py"
            template_path = Path(temp_dir) / "migration_template.py"
            
            # Create legacy file
            legacy_path.write_text("from old_module import HWCustomOpGenerator")
            
            report = self.workflow_migrator.create_migration_template(
                legacy_path, template_path
            )
            
            assert report.status == MigrationStatus.COMPLETED
            assert template_path.exists()
            
            template_content = template_path.read_text()
            assert "Migration Template" in template_content
            assert "enhanced_config" in template_content
            assert "TODO" in template_content
    
    def test_data_structure_migrator(self):
        """Test data structure migration."""
        # Create mock HWKernel
        mock_hw_kernel = Mock()
        mock_hw_kernel.module_name = "test_module"
        mock_hw_kernel.interfaces = []
        mock_hw_kernel.parameters = {"WIDTH": 32}
        mock_hw_kernel.source_file = "test.sv"
        mock_hw_kernel.description = "Test module"
        
        rtl_module = self.data_migrator.migrate_hw_kernel_to_rtl_module(mock_hw_kernel)
        
        assert isinstance(rtl_module, RTLModule)
        assert rtl_module.name == "test_module"
        assert rtl_module.interfaces == []
        assert rtl_module.parameters == {"WIDTH": 32}
        assert rtl_module.source_file == "test.sv"
        assert rtl_module.description == "Test module"
    
    def test_migrate_configuration_convenience(self):
        """Test convenience function for configuration migration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = Path(temp_dir) / "legacy_config.json"
            output_path = Path(temp_dir) / "new_config.json"
            
            # Create legacy config file
            legacy_config = {"generator_type": "hw_custom_op"}
            with open(legacy_path, 'w') as f:
                json.dump(legacy_config, f)
            
            new_config, report = migrate_configuration(legacy_path, output_path)
            
            assert isinstance(new_config, PipelineConfig)
            assert output_path.exists()
            assert report.status == MigrationStatus.COMPLETED
    
    def test_analyze_workflow_convenience(self):
        """Test convenience function for workflow analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_path = Path(temp_dir) / "workflow.py"
            workflow_path.write_text("# No legacy patterns here")
            
            analysis = analyze_workflow(workflow_path)
            
            assert "migration_required" in analysis
            assert not analysis["migration_required"]
            assert analysis["complexity"] == "none"


class TestBackwardCompatibility:
    """Test backward compatibility layer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_layer = LegacyAPILayer()
        self.rtl_module = self._create_test_rtl_module()
    
    def _create_test_rtl_module(self) -> RTLModule:
        """Create test RTL module."""
        signals = [RTLSignal("data", "input", 32)]
        interface = RTLInterface("test", "custom", signals)
        return RTLModule("test", [interface], {})
    
    def test_legacy_api_layer_initialization(self):
        """Test legacy API layer initialization."""
        self.api_layer._ensure_initialized()
        
        assert self.api_layer._config is not None
        assert self.api_layer._factory is not None
        assert self.api_layer._orchestrator is not None
    
    def test_legacy_hardware_kernel_generator(self):
        """Test legacy HardwareKernelGenerator class."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            hkg = LegacyHardwareKernelGenerator()
            
            assert len(w) == 1  # Should get deprecation warning
            assert "deprecated" in str(w[0].message).lower()
        
        assert isinstance(hkg.config, PipelineConfig)
        assert hkg.orchestrator is not None
    
    @patch('brainsmith.tools.hw_kernel_gen.compatibility.backward_compatibility.create_legacy_adapter')
    def test_generate_hwcustomop_function(self, mock_create_adapter):
        """Test legacy generate_hwcustomop function."""
        # Mock adapter
        mock_adapter = Mock()
        mock_result = GenerationResult(success=True)
        mock_result.add_artifact(GeneratedArtifact("test.py", "content", "hwcustomop"))
        mock_adapter.generate.return_value = mock_result
        mock_create_adapter.return_value = mock_adapter
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = generate_hwcustomop(
                hw_kernel=self.rtl_module,
                class_name="TestClass"
            )
            
            assert len(w) == 1  # Deprecation warning
            assert result == "content"
            mock_adapter.generate.assert_called_once()
    
    @patch('brainsmith.tools.hw_kernel_gen.compatibility.backward_compatibility.create_legacy_adapter')
    def test_generate_rtl_backend_function(self, mock_create_adapter):
        """Test legacy generate_rtl_backend function."""
        # Mock adapter
        mock_adapter = Mock()
        mock_result = GenerationResult(success=True)
        mock_result.add_artifact(GeneratedArtifact("backend.py", "content", "rtlbackend"))
        mock_adapter.generate.return_value = mock_result
        mock_create_adapter.return_value = mock_adapter
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = generate_rtl_backend(
                rtl_module=self.rtl_module,
                output_dir=Path("/tmp")
            )
            
            assert len(w) == 1  # Deprecation warning
            assert isinstance(result, Path)
            mock_adapter.generate.assert_called_once()
    
    def test_legacy_configuration_wrapper(self):
        """Test legacy configuration wrapper."""
        legacy_config = {
            "template": {"template_dirs": ["/tmp"]},
            "generator_type": "hw_custom_op"
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            wrapper = LegacyConfigurationWrapper(legacy_config)
            
            # Should have migration warnings
            assert len(w) > 0
        
        assert isinstance(wrapper.to_enhanced_config(), PipelineConfig)
        assert wrapper.get_migration_report() is not None
    
    def test_enable_legacy_warnings(self):
        """Test enabling/disabling legacy warnings."""
        # Test enabling warnings
        enable_legacy_warnings(True)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("test", DeprecationWarning)
            assert len(w) == 1
        
        # Test disabling warnings
        enable_legacy_warnings(False)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("test", DeprecationWarning)
            assert len(w) == 0
    
    def test_get_migration_status(self):
        """Test migration status function."""
        status = get_migration_status()
        
        assert "legacy_api_active" in status
        assert "enhanced_architecture_available" in status
        assert "recommendations" in status
        assert isinstance(status["recommendations"], list)
        assert len(status["recommendations"]) > 0


class TestMigrationReport:
    """Test migration report functionality."""
    
    def test_migration_report_creation(self):
        """Test migration report creation and updates."""
        report = MigrationReport()
        
        assert report.status == MigrationStatus.NOT_STARTED
        assert report.items_processed == 0
        assert report.items_migrated == 0
        assert report.items_failed == 0
        assert len(report.warnings) == 0
        assert len(report.errors) == 0
    
    def test_migration_report_operations(self):
        """Test migration report operations."""
        report = MigrationReport()
        
        # Add warning
        report.add_warning("Test warning", "item1")
        assert len(report.warnings) == 1
        assert len(report.migration_log) == 1
        assert report.migration_log[0]["type"] == "warning"
        
        # Add error
        report.add_error("Test error", "item2")
        assert len(report.errors) == 1
        assert report.items_failed == 1
        assert len(report.migration_log) == 2
        
        # Add success
        report.add_success("Test success", "item3")
        assert report.items_migrated == 1
        assert len(report.migration_log) == 3
    
    def test_migration_report_finalization(self):
        """Test migration report finalization."""
        report = MigrationReport()
        
        # Test completed status
        report.items_processed = 3
        report.add_success("Success 1", "item1")
        report.add_success("Success 2", "item2")
        report.add_success("Success 3", "item3")
        report.finalize()
        assert report.status == MigrationStatus.COMPLETED
        
        # Test partial status
        report = MigrationReport()
        report.items_processed = 3
        report.add_success("Success", "item1")
        report.add_error("Error", "item2")
        report.finalize()
        assert report.status == MigrationStatus.PARTIAL
        
        # Test failed status
        report = MigrationReport()
        report.items_processed = 2
        report.add_error("Error 1", "item1")
        report.add_error("Error 2", "item2")
        report.finalize()
        assert report.status == MigrationStatus.FAILED
    
    def test_migration_report_to_dict(self):
        """Test migration report dictionary conversion."""
        report = MigrationReport()
        report.items_processed = 2
        report.add_success("Success", "item1")
        report.add_warning("Warning", "item2")
        report.finalize()
        
        report_dict = report.to_dict()
        
        assert "status" in report_dict
        assert "items_processed" in report_dict
        assert "items_migrated" in report_dict
        assert "success_rate" in report_dict
        assert "warnings" in report_dict
        assert "errors" in report_dict
        assert "migration_log" in report_dict
        
        assert report_dict["success_rate"] == 0.5  # 1 success out of 2 processed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])