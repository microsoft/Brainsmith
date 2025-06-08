"""
Migration Utilities for Transitioning to Enhanced Architecture.

This module provides utilities to help migrate existing code, configurations,
and workflows from the legacy architecture to the new Week 3 enhanced system.
"""

import json
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import asdict
from enum import Enum

from ..enhanced_config import PipelineConfig, GeneratorType, DataflowMode, ValidationLevel
from ..enhanced_data_structures import RTLModule
from ..errors import BrainsmithError, ConfigurationError, ValidationError


class MigrationStatus(Enum):
    """Status of migration operations."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class MigrationReport:
    """Report of migration results."""
    
    def __init__(self):
        self.status = MigrationStatus.NOT_STARTED
        self.items_processed = 0
        self.items_migrated = 0
        self.items_failed = 0
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.suggestions: List[str] = []
        self.backup_paths: List[Path] = []
        self.migration_log: List[Dict[str, Any]] = []
    
    def add_warning(self, message: str, item: str = None):
        """Add migration warning."""
        warning = f"{item}: {message}" if item else message
        self.warnings.append(warning)
        self.migration_log.append({
            "type": "warning",
            "message": warning,
            "item": item
        })
    
    def add_error(self, message: str, item: str = None):
        """Add migration error."""
        error = f"{item}: {message}" if item else message
        self.errors.append(error)
        self.items_failed += 1
        self.migration_log.append({
            "type": "error", 
            "message": error,
            "item": item
        })
    
    def add_success(self, message: str, item: str = None):
        """Add successful migration."""
        self.items_migrated += 1
        self.migration_log.append({
            "type": "success",
            "message": message,
            "item": item
        })
    
    def finalize(self):
        """Finalize migration report."""
        if self.items_failed == 0:
            if self.items_migrated == self.items_processed:
                self.status = MigrationStatus.COMPLETED
            else:
                self.status = MigrationStatus.PARTIAL
        else:
            if self.items_migrated > 0:
                self.status = MigrationStatus.PARTIAL
            else:
                self.status = MigrationStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "status": self.status.value,
            "items_processed": self.items_processed,
            "items_migrated": self.items_migrated,
            "items_failed": self.items_failed,
            "success_rate": self.items_migrated / max(self.items_processed, 1),
            "warnings": self.warnings,
            "errors": self.errors,
            "suggestions": self.suggestions,
            "backup_paths": [str(p) for p in self.backup_paths],
            "migration_log": self.migration_log
        }


class ConfigurationMigrator:
    """Migrator for configuration formats."""
    
    def __init__(self, backup_enabled: bool = True):
        self.backup_enabled = backup_enabled
        self.logger = logging.getLogger(__name__)
    
    def migrate_legacy_config(
        self, 
        legacy_config: Dict[str, Any], 
        output_path: Optional[Path] = None
    ) -> Tuple[PipelineConfig, MigrationReport]:
        """
        Migrate legacy configuration format to new PipelineConfig.
        
        Args:
            legacy_config: Legacy configuration dictionary
            output_path: Optional path to save migrated config
            
        Returns:
            Tuple of (new_config, migration_report)
        """
        report = MigrationReport()
        report.status = MigrationStatus.IN_PROGRESS
        
        try:
            # Create default new configuration
            new_config = PipelineConfig()
            
            # Migrate template configuration
            self._migrate_template_config(legacy_config, new_config, report)
            
            # Migrate generation configuration
            self._migrate_generation_config(legacy_config, new_config, report)
            
            # Migrate analysis configuration
            self._migrate_analysis_config(legacy_config, new_config, report)
            
            # Migrate validation configuration
            self._migrate_validation_config(legacy_config, new_config, report)
            
            # Migrate dataflow configuration
            self._migrate_dataflow_config(legacy_config, new_config, report)
            
            # Migrate global settings
            self._migrate_global_settings(legacy_config, new_config, report)
            
            # Save migrated configuration if path provided
            if output_path:
                self._save_migrated_config(new_config, output_path, report)
            
            report.finalize()
            return new_config, report
            
        except Exception as e:
            report.add_error(f"Configuration migration failed: {e}")
            report.status = MigrationStatus.FAILED
            return PipelineConfig(), report
    
    def _migrate_template_config(
        self, 
        legacy: Dict[str, Any], 
        new_config: PipelineConfig, 
        report: MigrationReport
    ):
        """Migrate template-related configuration."""
        template_section = legacy.get("template", {})
        report.items_processed += 1
        
        try:
            # Migrate template directories
            if "template_dirs" in template_section:
                dirs = template_section["template_dirs"]
                if isinstance(dirs, (list, tuple)):
                    new_config.template.template_dirs = [Path(d) for d in dirs]
                else:
                    new_config.template.template_dirs = [Path(dirs)]
                report.add_success("Migrated template directories")
            
            # Migrate template settings
            template_mappings = {
                "enable_caching": "enable_caching",
                "cache_size": "cache_size", 
                "auto_reload": "auto_reload",
                "strict_undefined": "strict_undefined"
            }
            
            for legacy_key, new_key in template_mappings.items():
                if legacy_key in template_section:
                    setattr(new_config.template, new_key, template_section[legacy_key])
                    report.add_success(f"Migrated template setting: {legacy_key}")
            
        except Exception as e:
            report.add_error(f"Template configuration migration failed: {e}", "template")
    
    def _migrate_generation_config(
        self, 
        legacy: Dict[str, Any], 
        new_config: PipelineConfig, 
        report: MigrationReport
    ):
        """Migrate generation-related configuration."""
        generation_section = legacy.get("generation", {})
        report.items_processed += 1
        
        try:
            # Migrate output settings
            if "output_dir" in generation_section:
                new_config.generation.output_dir = Path(generation_section["output_dir"])
                report.add_success("Migrated output directory")
            
            # Migrate generation flags
            generation_mappings = {
                "overwrite_existing": "overwrite_existing",
                "include_debug_info": "include_debug_info", 
                "include_documentation": "include_documentation",
                "include_type_hints": "include_type_hints"
            }
            
            for legacy_key, new_key in generation_mappings.items():
                if legacy_key in generation_section:
                    setattr(new_config.generation, new_key, generation_section[legacy_key])
                    report.add_success(f"Migrated generation setting: {legacy_key}")
            
            # Migrate enabled generators
            if "enabled_generators" in generation_section:
                enabled = generation_section["enabled_generators"]
                if isinstance(enabled, (list, tuple)):
                    new_config.generation.enabled_generators = set(enabled)
                else:
                    new_config.generation.enabled_generators = {enabled}
                report.add_success("Migrated enabled generators")
            
        except Exception as e:
            report.add_error(f"Generation configuration migration failed: {e}", "generation")
    
    def _migrate_analysis_config(
        self, 
        legacy: Dict[str, Any], 
        new_config: PipelineConfig, 
        report: MigrationReport
    ):
        """Migrate analysis-related configuration."""
        analysis_section = legacy.get("analysis", {})
        report.items_processed += 1
        
        try:
            analysis_mappings = {
                "analyze_interfaces": "analyze_interfaces",
                "analyze_dependencies": "analyze_dependencies",
                "analyze_timing": "analyze_timing", 
                "follow_includes": "follow_includes",
                "max_depth": "max_depth"
            }
            
            for legacy_key, new_key in analysis_mappings.items():
                if legacy_key in analysis_section:
                    setattr(new_config.analysis, new_key, analysis_section[legacy_key])
                    report.add_success(f"Migrated analysis setting: {legacy_key}")
            
        except Exception as e:
            report.add_error(f"Analysis configuration migration failed: {e}", "analysis")
    
    def _migrate_validation_config(
        self, 
        legacy: Dict[str, Any], 
        new_config: PipelineConfig, 
        report: MigrationReport
    ):
        """Migrate validation-related configuration."""
        validation_section = legacy.get("validation", {})
        report.items_processed += 1
        
        try:
            # Migrate validation level
            if "level" in validation_section:
                level_str = validation_section["level"]
                try:
                    new_config.validation.level = ValidationLevel(level_str)
                    report.add_success("Migrated validation level")
                except ValueError:
                    report.add_warning(f"Unknown validation level '{level_str}', using default")
            
            # Migrate validation flags
            validation_mappings = {
                "validate_syntax": "validate_syntax",
                "validate_semantics": "validate_semantics",
                "fail_on_warnings": "fail_on_warnings",
                "max_errors": "max_errors"
            }
            
            for legacy_key, new_key in validation_mappings.items():
                if legacy_key in validation_section:
                    setattr(new_config.validation, new_key, validation_section[legacy_key])
                    report.add_success(f"Migrated validation setting: {legacy_key}")
            
        except Exception as e:
            report.add_error(f"Validation configuration migration failed: {e}", "validation")
    
    def _migrate_dataflow_config(
        self, 
        legacy: Dict[str, Any], 
        new_config: PipelineConfig, 
        report: MigrationReport
    ):
        """Migrate dataflow-related configuration."""
        dataflow_section = legacy.get("dataflow", {})
        report.items_processed += 1
        
        try:
            # Migrate dataflow mode
            if "mode" in dataflow_section:
                mode_str = dataflow_section["mode"]
                try:
                    new_config.dataflow.mode = DataflowMode(mode_str)
                    report.add_success("Migrated dataflow mode")
                except ValueError:
                    report.add_warning(f"Unknown dataflow mode '{mode_str}', using default")
            
            # Migrate dataflow settings
            dataflow_mappings = {
                "enable_parallelism_optimization": "enable_parallelism_optimization",
                "enable_tensor_chunking": "enable_tensor_chunking",
                "resource_estimation_enabled": "resource_estimation_enabled"
            }
            
            for legacy_key, new_key in dataflow_mappings.items():
                if legacy_key in dataflow_section:
                    setattr(new_config.dataflow, new_key, dataflow_section[legacy_key])
                    report.add_success(f"Migrated dataflow setting: {legacy_key}")
            
        except Exception as e:
            report.add_error(f"Dataflow configuration migration failed: {e}", "dataflow")
    
    def _migrate_global_settings(
        self, 
        legacy: Dict[str, Any], 
        new_config: PipelineConfig, 
        report: MigrationReport
    ):
        """Migrate global settings."""
        report.items_processed += 1
        
        try:
            # Migrate generator type
            if "generator_type" in legacy:
                type_str = legacy["generator_type"]
                try:
                    new_config.generator_type = GeneratorType(type_str)
                    report.add_success("Migrated generator type")
                except ValueError:
                    report.add_warning(f"Unknown generator type '{type_str}', using default")
            
            # Migrate global flags
            global_mappings = {
                "enable_caching": "enable_caching",
                "verbose": "verbose",
                "debug": "debug",
                "optimization_enabled": "optimization_enabled"
            }
            
            for legacy_key, new_key in global_mappings.items():
                if legacy_key in legacy:
                    setattr(new_config, new_key, legacy[legacy_key])
                    report.add_success(f"Migrated global setting: {legacy_key}")
            
        except Exception as e:
            report.add_error(f"Global settings migration failed: {e}", "global")
    
    def _save_migrated_config(
        self, 
        config: PipelineConfig, 
        output_path: Path, 
        report: MigrationReport
    ):
        """Save migrated configuration to file."""
        try:
            # Convert config to dictionary
            config_dict = asdict(config)
            
            # Create backup of existing file if it exists
            if output_path.exists() and self.backup_enabled:
                backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
                output_path.rename(backup_path)
                report.backup_paths.append(backup_path)
            
            # Save new configuration
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            report.add_success(f"Saved migrated configuration to {output_path}")
            
        except Exception as e:
            report.add_error(f"Failed to save migrated configuration: {e}")


class WorkflowMigrator:
    """Migrator for existing workflows and scripts."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_legacy_workflow(self, workflow_path: Path) -> Dict[str, Any]:
        """
        Analyze legacy workflow to identify migration requirements.
        
        Args:
            workflow_path: Path to legacy workflow file
            
        Returns:
            Analysis results with migration recommendations
        """
        analysis = {
            "file_path": str(workflow_path),
            "legacy_patterns": [],
            "migration_required": False,
            "complexity": "low",
            "recommendations": [],
            "estimated_effort": "minimal"
        }
        
        try:
            if not workflow_path.exists():
                analysis["error"] = "File does not exist"
                return analysis
            
            content = workflow_path.read_text()
            
            # Check for legacy patterns
            legacy_patterns = [
                ("HWCustomOpGenerator", "Direct HWCustomOpGenerator usage"),
                ("generate_rtl_template", "Legacy RTL template function"),
                ("HardwareKernelGenerator", "Legacy HKG usage"),
                ("template_dir=", "Direct template directory specification"),
                ("Jinja2", "Direct Jinja2 usage")
            ]
            
            for pattern, description in legacy_patterns:
                if pattern in content:
                    analysis["legacy_patterns"].append({
                        "pattern": pattern,
                        "description": description,
                        "count": content.count(pattern)
                    })
                    analysis["migration_required"] = True
            
            # Assess complexity
            pattern_count = len(analysis["legacy_patterns"])
            if pattern_count == 0:
                analysis["complexity"] = "none"
                analysis["estimated_effort"] = "none"
            elif pattern_count <= 2:
                analysis["complexity"] = "low"
                analysis["estimated_effort"] = "minimal"
            elif pattern_count <= 5:
                analysis["complexity"] = "medium" 
                analysis["estimated_effort"] = "moderate"
            else:
                analysis["complexity"] = "high"
                analysis["estimated_effort"] = "significant"
            
            # Generate recommendations
            if analysis["migration_required"]:
                analysis["recommendations"].extend([
                    "Replace direct generator instantiation with factory pattern",
                    "Use enhanced configuration system",
                    "Migrate to new orchestration APIs",
                    "Update error handling to use BrainsmithError framework",
                    "Consider using workflow engine for complex pipelines"
                ])
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def create_migration_template(
        self, 
        legacy_workflow_path: Path, 
        output_path: Path
    ) -> MigrationReport:
        """
        Create migration template for legacy workflow.
        
        Args:
            legacy_workflow_path: Path to legacy workflow
            output_path: Path for migration template
            
        Returns:
            Migration report
        """
        report = MigrationReport()
        
        try:
            # Analyze legacy workflow
            analysis = self.analyze_legacy_workflow(legacy_workflow_path)
            
            if not analysis["migration_required"]:
                report.add_success("No migration required")
                report.status = MigrationStatus.COMPLETED
                return report
            
            # Generate migration template
            template_content = self._generate_migration_template_content(analysis)
            
            # Save template
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(template_content)
            
            report.add_success(f"Migration template created: {output_path}")
            report.items_processed = 1
            report.items_migrated = 1
            report.status = MigrationStatus.COMPLETED
            
        except Exception as e:
            report.add_error(f"Failed to create migration template: {e}")
            report.status = MigrationStatus.FAILED
        
        return report
    
    def _generate_migration_template_content(self, analysis: Dict[str, Any]) -> str:
        """Generate migration template content."""
        template = f'''"""
Migration Template for {Path(analysis["file_path"]).name}

This file provides a template for migrating the legacy workflow to the
new enhanced architecture.

Legacy patterns detected:
"""

# Import enhanced components
from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig, GeneratorType
from brainsmith.tools.hw_kernel_gen.orchestration.generator_factory import GeneratorFactory
from brainsmith.tools.hw_kernel_gen.orchestration.integration_orchestrator import IntegrationOrchestrator
from brainsmith.tools.hw_kernel_gen.compatibility import create_legacy_adapter

def migrate_legacy_workflow():
    """
    Migrated version of legacy workflow.
    
    Migration notes:
'''
        
        for pattern in analysis["legacy_patterns"]:
            template += f"    - {pattern['description']}: Found {pattern['count']} occurrences\n"
        
        template += '''
    """
    
    # Create enhanced configuration
    config = PipelineConfig()
    
    # Configure for your specific needs
    config.generator_type = GeneratorType.AUTO_HW_CUSTOM_OP
    config.generation.output_dir = Path("output")
    config.dataflow.mode = DataflowMode.HYBRID
    
    # Option 1: Use enhanced generators directly
    factory = GeneratorFactory(config)
    # Register enhanced generators here
    
    # Option 2: Use legacy adapters for gradual migration
    legacy_adapter = create_legacy_adapter(GeneratorType.HW_CUSTOM_OP, config)
    
    # Option 3: Use integration orchestrator for complex workflows
    orchestrator = IntegrationOrchestrator(config)
    
    # TODO: Implement your specific workflow logic here
    # Replace legacy generator calls with new architecture
    
    pass

if __name__ == "__main__":
    migrate_legacy_workflow()
'''
        
        return template


class DataStructureMigrator:
    """Migrator for data structure formats."""
    
    def migrate_hw_kernel_to_rtl_module(self, hw_kernel: Any) -> RTLModule:
        """
        Migrate HWKernel to RTLModule format.
        
        Args:
            hw_kernel: Legacy HWKernel object
            
        Returns:
            Migrated RTLModule
        """
        try:
            return RTLModule(
                name=getattr(hw_kernel, 'module_name', 'unknown_module'),
                interfaces=getattr(hw_kernel, 'interfaces', []),
                parameters=getattr(hw_kernel, 'parameters', {}),
                source_file=getattr(hw_kernel, 'source_file', None),
                description=getattr(hw_kernel, 'description', None)
            )
        except Exception as e:
            raise ValidationError(f"Failed to migrate HWKernel to RTLModule: {e}")


# Convenience functions
def migrate_configuration(
    legacy_config_path: Path, 
    output_path: Optional[Path] = None
) -> Tuple[PipelineConfig, MigrationReport]:
    """
    Convenience function to migrate configuration files.
    
    Args:
        legacy_config_path: Path to legacy configuration
        output_path: Optional output path for migrated config
        
    Returns:
        Tuple of (migrated_config, migration_report)
    """
    try:
        with open(legacy_config_path, 'r') as f:
            legacy_config = json.load(f)
    except Exception as e:
        report = MigrationReport()
        report.add_error(f"Failed to load legacy configuration: {e}")
        return PipelineConfig(), report
    
    migrator = ConfigurationMigrator()
    return migrator.migrate_legacy_config(legacy_config, output_path)


def analyze_workflow(workflow_path: Path) -> Dict[str, Any]:
    """
    Convenience function to analyze legacy workflow.
    
    Args:
        workflow_path: Path to legacy workflow file
        
    Returns:
        Analysis results
    """
    migrator = WorkflowMigrator()
    return migrator.analyze_legacy_workflow(workflow_path)


def create_migration_guide(
    legacy_project_path: Path,
    output_path: Path
) -> MigrationReport:
    """
    Create comprehensive migration guide for legacy project.
    
    Args:
        legacy_project_path: Path to legacy project directory
        output_path: Path for migration guide
        
    Returns:
        Migration report
    """
    report = MigrationReport()
    
    try:
        # Find Python files in project
        python_files = list(legacy_project_path.rglob("*.py"))
        config_files = list(legacy_project_path.rglob("*.json"))
        
        # Analyze each file
        workflow_migrator = WorkflowMigrator()
        analyses = []
        
        for py_file in python_files:
            if "test" not in str(py_file):  # Skip test files
                analysis = workflow_migrator.analyze_legacy_workflow(py_file)
                if analysis["migration_required"]:
                    analyses.append(analysis)
        
        # Generate comprehensive guide
        guide_content = _generate_migration_guide_content(analyses, config_files)
        
        # Save guide
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(guide_content)
        
        report.add_success(f"Migration guide created: {output_path}")
        report.items_processed = len(python_files) + len(config_files)
        report.items_migrated = 1
        report.status = MigrationStatus.COMPLETED
        
    except Exception as e:
        report.add_error(f"Failed to create migration guide: {e}")
        report.status = MigrationStatus.FAILED
    
    return report


def _generate_migration_guide_content(analyses: List[Dict[str, Any]], config_files: List[Path]) -> str:
    """Generate migration guide content."""
    guide = '''# Migration Guide: Legacy to Enhanced Architecture

This guide provides instructions for migrating your project to the new enhanced architecture.

## Overview

The enhanced architecture provides:
- Improved performance and scalability
- Better error handling and validation
- Dataflow integration capabilities
- Orchestration and workflow management
- Enhanced template system

## Files Requiring Migration

'''
    
    for analysis in analyses:
        guide += f"### {Path(analysis['file_path']).name}\n"
        guide += f"- **Complexity**: {analysis['complexity']}\n"
        guide += f"- **Estimated Effort**: {analysis['estimated_effort']}\n"
        guide += "- **Legacy Patterns**:\n"
        
        for pattern in analysis['legacy_patterns']:
            guide += f"  - {pattern['description']} ({pattern['count']} occurrences)\n"
        
        guide += "- **Recommendations**:\n"
        for rec in analysis['recommendations']:
            guide += f"  - {rec}\n"
        guide += "\n"
    
    guide += "## Configuration Files\n\n"
    for config_file in config_files:
        guide += f"- {config_file.name}: Use ConfigurationMigrator to update\n"
    
    guide += '''
## Migration Steps

1. **Backup your project**
2. **Install enhanced dependencies**
3. **Migrate configuration files** using ConfigurationMigrator
4. **Update import statements** to use enhanced modules
5. **Replace legacy generators** with enhanced versions or adapters
6. **Update workflow logic** to use new orchestration APIs
7. **Run tests** to validate migration
8. **Gradually remove legacy adapters** as you fully migrate

## Resources

- Enhanced Architecture Documentation
- API Reference
- Migration Examples
- Support Channels
'''
    
    return guide