"""
Enhanced Generator Base with AutoHWCustomOp Integration.

This module provides a comprehensive generator framework that integrates with
the dataflow modeling system and AutoHWCustomOp/AutoRTLBackend base classes
to minimize generated code and maximize functionality.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Set
from pathlib import Path
import time
from abc import ABC, abstractmethod
from datetime import datetime

# Import dataflow components
try:
    from ...dataflow.core.auto_hw_custom_op import AutoHWCustomOp
    from ...dataflow.core.auto_rtl_backend import AutoRTLBackend
    from ...dataflow.core.dataflow_model import DataflowModel
    from ...dataflow.core.validation import ValidationResult, create_validation_result
    DATAFLOW_AVAILABLE = True
except ImportError:
    DATAFLOW_AVAILABLE = False
    # Create placeholder types
    class AutoHWCustomOp: pass
    class AutoRTLBackend: pass
    class DataflowModel: pass
    ValidationResult = Dict[str, Any]

from .enhanced_config import PipelineConfig, GeneratorType, DataflowMode
from .enhanced_template_context import (
    BaseContext, DataflowContext, HWCustomOpContext, RTLBackendContext
)
from .enhanced_template_manager import EnhancedTemplateManager
from .errors import CodeGenerationError, TemplateError


@dataclass
class GeneratedArtifact:
    """Represents a generated code artifact."""
    
    # Basic artifact information
    file_name: str
    content: str
    artifact_type: str  # hwcustomop, rtlbackend, test, documentation, wrapper
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = field(default_factory=time.time)
    content_hash: Optional[str] = None
    
    # Validation
    validation_result: Optional[ValidationResult] = None
    is_validated: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.content_hash is None:
            import hashlib
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()
    
    def validate(self) -> bool:
        """Validate the generated artifact."""
        if DATAFLOW_AVAILABLE:
            self.validation_result = create_validation_result()
        else:
            self.validation_result = {"errors": [], "warnings": [], "success": True}
        
        # Basic validation checks
        if not self.content.strip():
            if DATAFLOW_AVAILABLE:
                from ...dataflow.core.validation import ValidationError, ValidationSeverity
                error = ValidationError(
                    component="GeneratedArtifact",
                    error_type="empty_content",
                    message=f"Generated content is empty for {self.file_name}",
                    severity=ValidationSeverity.ERROR,
                    context={"artifact_type": self.artifact_type}
                )
                self.validation_result.add_error(error)
            else:
                error = {
                    "component": "GeneratedArtifact",
                    "error_type": "empty_content", 
                    "message": f"Generated content is empty for {self.file_name}",
                    "context": {"artifact_type": self.artifact_type}
                }
                self.validation_result["errors"].append(error)
                self.validation_result["success"] = False
        
        if not self.file_name:
            if DATAFLOW_AVAILABLE:
                from ...dataflow.core.validation import ValidationError, ValidationSeverity
                error = ValidationError(
                    component="GeneratedArtifact",
                    error_type="missing_filename",
                    message="File name is required for artifact",
                    severity=ValidationSeverity.ERROR,
                    context={"artifact_type": self.artifact_type}
                )
                self.validation_result.add_error(error)
            else:
                error = {
                    "component": "GeneratedArtifact", 
                    "error_type": "missing_filename",
                    "message": "File name is required for artifact",
                    "context": {"artifact_type": self.artifact_type}
                }
                self.validation_result["errors"].append(error)
                self.validation_result["success"] = False
        
        # Type-specific validation
        if self.artifact_type == "hwcustomop":
            self._validate_python_syntax()
        elif self.artifact_type == "rtlbackend":
            self._validate_python_syntax()
        elif self.artifact_type == "test":
            self._validate_test_content()
        
        self.is_validated = True
        return (self.validation_result.success if DATAFLOW_AVAILABLE 
                else self.validation_result.get("success", True))
    
    def _validate_python_syntax(self) -> None:
        """Validate Python syntax for Python artifacts."""
        try:
            import ast
            ast.parse(self.content)
        except SyntaxError as e:
            if DATAFLOW_AVAILABLE:
                from ...dataflow.core.validation import ValidationError, ValidationSeverity
                error = ValidationError(
                    component="GeneratedArtifact",
                    error_type="syntax_error",
                    message=f"Python syntax error in {self.file_name}: {e}",
                    severity=ValidationSeverity.ERROR,
                    context={"line": e.lineno, "offset": e.offset}
                )
                self.validation_result.add_error(error)
            else:
                error = {
                    "component": "GeneratedArtifact",
                    "error_type": "syntax_error",
                    "message": f"Python syntax error in {self.file_name}: {e}",
                    "context": {"line": e.lineno, "offset": e.offset}
                }
                self.validation_result["errors"].append(error)
                self.validation_result["success"] = False
    
    def _validate_test_content(self) -> None:
        """Validate test file content."""
        # Check for basic test structure
        required_patterns = ["class Test", "def test_", "import", "assert"]
        missing_patterns = [pattern for pattern in required_patterns 
                          if pattern not in self.content]
        
        if missing_patterns:
            if DATAFLOW_AVAILABLE:
                from ...dataflow.core.validation import ValidationError, ValidationSeverity
                error = ValidationError(
                    component="GeneratedArtifact",
                    error_type="incomplete_test",
                    message=f"Test file missing required patterns: {missing_patterns}",
                    severity=ValidationSeverity.ERROR,
                    context={"missing_patterns": missing_patterns}
                )
                self.validation_result.add_error(error)
            else:
                error = {
                    "component": "GeneratedArtifact",
                    "error_type": "incomplete_test",
                    "message": f"Test file missing required patterns: {missing_patterns}",
                    "context": {"missing_patterns": missing_patterns}
                }
                self.validation_result["errors"].append(error)
                self.validation_result["success"] = False
    
    def write_to_file(self, base_dir: Path) -> Path:
        """Write artifact content to file."""
        file_path = base_dir / self.file_name
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.content)
            return file_path
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to write artifact to {file_path}: {e}",
                artifact_type=self.artifact_type,
                context={"file_path": str(file_path)},
                suggestion="Check file permissions and disk space"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary representation."""
        return {
            "file_name": self.file_name,
            "artifact_type": self.artifact_type,
            "content_length": len(self.content),
            "content_hash": self.content_hash,
            "generation_time": self.generation_time,
            "metadata": self.metadata,
            "is_validated": self.is_validated,
            "validation_success": (
                self.validation_result.success if DATAFLOW_AVAILABLE and self.validation_result 
                else self.validation_result.get("success", True) if self.validation_result 
                else None
            )
        }


@dataclass
class GenerationResult:
    """Container for generation results with validation and metrics."""
    
    # Status
    success: bool = True
    
    # Results
    artifacts: List[GeneratedArtifact] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metrics and metadata
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0
    total_content_size: int = 0
    
    def add_artifact(self, artifact: GeneratedArtifact) -> None:
        """Add artifact to results."""
        self.artifacts.append(artifact)
        self.total_content_size += len(artifact.content)
    
    def add_error(self, error: str) -> None:
        """Add error message."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str) -> None:
        """Add warning message."""
        self.warnings.append(warning)
    
    def validate_all_artifacts(self) -> bool:
        """Validate all artifacts."""
        all_valid = True
        for artifact in self.artifacts:
            if not artifact.validate():
                all_valid = False
                self.add_error(f"Validation failed for {artifact.file_name}")
        
        return all_valid
    
    def write_all_artifacts(self, base_dir: Path) -> List[Path]:
        """Write all artifacts to files."""
        written_files = []
        for artifact in self.artifacts:
            try:
                file_path = artifact.write_to_file(base_dir)
                written_files.append(file_path)
            except Exception as e:
                self.add_error(f"Failed to write {artifact.file_name}: {e}")
        
        return written_files
    
    def get_artifacts_by_type(self, artifact_type: str) -> List[GeneratedArtifact]:
        """Get artifacts of specific type."""
        return [artifact for artifact in self.artifacts if artifact.artifact_type == artifact_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "success": self.success,
            "artifact_count": len(self.artifacts),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "error_count": len(self.errors),
            "errors": self.errors,
            "warning_count": len(self.warnings),
            "warnings": self.warnings,
            "generation_time": self.generation_time,
            "total_content_size": self.total_content_size,
            "metrics": self.metrics
        }


class GeneratorBase(ABC):
    """
    Abstract base class for all code generators with dataflow integration.
    
    This base class provides:
    - Integration with dataflow modeling system
    - Template management and rendering
    - Artifact validation and output handling
    - Performance metrics and caching
    - Error handling and recovery
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        template_manager: Optional[EnhancedTemplateManager] = None,
        context_builder: Optional[Any] = None
    ):
        """Initialize generator with dependencies."""
        self.config = config
        self.template_manager = template_manager
        self.context_builder = context_builder
        
        # Generation state
        self._generation_count = 0
        self._total_generation_time = 0.0
        self._last_generation_time = None
        
        # Cache for contexts if enabled
        self._context_cache: Dict[str, BaseContext] = {}
    
    @abstractmethod
    def get_template_name(self) -> str:
        """Get the primary template name for this generator."""
        pass
    
    @abstractmethod
    def get_artifact_type(self) -> str:
        """Get the artifact type produced by this generator."""
        pass
    
    def generate(self, inputs: Dict[str, Any]) -> GenerationResult:
        """
        Generate code artifacts from inputs.
        
        Args:
            inputs: Generation inputs including hw_kernel, config, etc.
            
        Returns:
            GenerationResult with artifacts and status
        """
        start_time = time.time()
        result = GenerationResult()
        
        try:
            # Build context
            context = self._build_context(inputs)
            
            # Validate context if enabled
            if self.config.validation.validate_dataflow_model:
                context_validation = context.validate()
                if not (context_validation.success if DATAFLOW_AVAILABLE 
                       else context_validation.get("success", True)):
                    errors = (context_validation.errors if DATAFLOW_AVAILABLE 
                             else context_validation.get("errors", []))
                    for error in errors:
                        result.add_error(f"Context validation error: {error}")
                    return result
            
            # Generate primary artifact
            primary_artifact = self._generate_primary_artifact(context)
            result.add_artifact(primary_artifact)
            
            # Generate additional artifacts if configured
            additional_artifacts = self._generate_additional_artifacts(context)
            for artifact in additional_artifacts:
                result.add_artifact(artifact)
            
            # Update metrics
            generation_time = time.time() - start_time
            result.generation_time = generation_time
            result.metrics.update(self._calculate_metrics(result))
            
            # Update generator statistics
            self._generation_count += 1
            self._total_generation_time += generation_time
            self._last_generation_time = generation_time
            
        except Exception as e:
            result.success = False
            result.add_error(f"Generation failed: {str(e)}")
            
            # Add detailed error information
            result.metrics["error_type"] = type(e).__name__
            result.metrics["error_context"] = getattr(e, 'context', {})
        
        return result
    
    def _build_context(self, inputs: Dict[str, Any]) -> BaseContext:
        """Build template context from inputs."""
        if not self.context_builder:
            raise CodeGenerationError(
                "Context builder not configured",
                generator_type=self.get_artifact_type(),
                suggestion="Configure context builder for generator"
            )
        
        # Extract inputs
        hw_kernel = inputs.get("hw_kernel")
        config = inputs.get("config", self.config)
        
        if not hw_kernel:
            raise CodeGenerationError(
                "Hardware kernel not provided in inputs",
                generator_type=self.get_artifact_type(),
                suggestion="Provide hw_kernel in generator inputs"
            )
        
        # Determine context building method based on generator type and config
        if self.config.is_dataflow_enabled():
            return self._build_dataflow_context(inputs)
        else:
            return self._build_legacy_context(inputs)
    
    def _build_dataflow_context(self, inputs: Dict[str, Any]) -> BaseContext:
        """Build dataflow-aware context."""
        hw_kernel = inputs["hw_kernel"]
        config = inputs.get("config", self.config)
        onnx_metadata = inputs.get("onnx_metadata")
        
        if self.get_artifact_type() == "hwcustomop":
            finn_config = inputs.get("finn_config", {})
            return self.context_builder.build_hwcustomop_context(
                hw_kernel, config, finn_config, onnx_metadata
            )
        elif self.get_artifact_type() == "rtlbackend":
            backend_config = inputs.get("backend_config", {})
            return self.context_builder.build_rtlbackend_context(
                hw_kernel, config, backend_config, onnx_metadata
            )
        else:
            # For other generators, use base dataflow context
            return self.context_builder.build_dataflow_context(
                hw_kernel, config, onnx_metadata
            )
    
    def _build_legacy_context(self, inputs: Dict[str, Any]) -> BaseContext:
        """Build legacy context for backward compatibility."""
        hw_kernel = inputs["hw_kernel"]
        config = inputs.get("config", self.config)
        
        return self.context_builder.build_base_context(hw_kernel, config)
    
    def _generate_primary_artifact(self, context: BaseContext) -> GeneratedArtifact:
        """Generate the primary artifact for this generator."""
        if not self.template_manager:
            raise CodeGenerationError(
                "Template manager not configured",
                generator_type=self.get_artifact_type(),
                suggestion="Configure template manager for generator"
            )
        
        # Render template
        template_name = self.get_template_name()
        try:
            content = self.template_manager.render_template(
                template_name,
                context.to_dict(),
                generator_type=self.config.generator_type
            )
        except TemplateError as e:
            raise CodeGenerationError(
                f"Template rendering failed for {template_name}: {e}",
                template_name=template_name,
                generator_type=self.get_artifact_type(),
                suggestion="Check template syntax and context variables"
            ) from e
        
        # Create artifact
        artifact = GeneratedArtifact(
            file_name=context.file_name,
            content=content,
            artifact_type=self.get_artifact_type(),
            metadata={
                "template_name": template_name,
                "generator_type": self.get_artifact_type(),
                "kernel_name": context.kernel_name,
                "class_name": context.class_name,
                "generation_timestamp": context.generation_timestamp,
                "dataflow_enabled": self.config.is_dataflow_enabled(),
                "config_metadata": context.config_metadata
            }
        )
        
        return artifact
    
    def _generate_additional_artifacts(self, context: BaseContext) -> List[GeneratedArtifact]:
        """Generate additional artifacts (e.g., tests, documentation)."""
        artifacts = []
        
        # Generate test if enabled
        if ("test" in self.config.generation.enabled_generators and 
            hasattr(context, 'generate_test') and context.generate_test):
            test_artifact = self._generate_test_artifact(context)
            if test_artifact:
                artifacts.append(test_artifact)
        
        # Generate documentation if enabled
        if ("documentation" in self.config.generation.enabled_generators and
            self.config.generation.include_documentation):
            doc_artifact = self._generate_documentation_artifact(context)
            if doc_artifact:
                artifacts.append(doc_artifact)
        
        return artifacts
    
    def _generate_test_artifact(self, context: BaseContext) -> Optional[GeneratedArtifact]:
        """Generate test artifact if configured."""
        try:
            test_template = f"test_{self.get_artifact_type()}.py.j2"
            test_content = self.template_manager.render_template(
                test_template,
                context.to_dict()
            )
            
            return GeneratedArtifact(
                file_name=f"test_{context.kernel_name.lower()}.py",
                content=test_content,
                artifact_type="test",
                metadata={"generator_type": self.get_artifact_type()}
            )
        except TemplateError:
            # Test template not available, skip
            return None
    
    def _generate_documentation_artifact(self, context: BaseContext) -> Optional[GeneratedArtifact]:
        """Generate documentation artifact if configured."""
        try:
            doc_template = f"{self.get_artifact_type()}_doc.md.j2"
            doc_content = self.template_manager.render_template(
                doc_template,
                context.to_dict()
            )
            
            return GeneratedArtifact(
                file_name=f"{context.kernel_name.lower()}_README.md",
                content=doc_content,
                artifact_type="documentation",
                metadata={"generator_type": self.get_artifact_type()}
            )
        except TemplateError:
            # Documentation template not available, skip
            return None
    
    def _calculate_metrics(self, result: GenerationResult) -> Dict[str, Any]:
        """Calculate generation metrics."""
        return {
            "artifact_count": len(result.artifacts),
            "total_content_size": result.total_content_size,
            "average_artifact_size": (
                result.total_content_size / len(result.artifacts) 
                if result.artifacts else 0
            ),
            "generation_time": result.generation_time,
            "template_name": self.get_template_name(),
            "generator_statistics": {
                "total_generations": self._generation_count,
                "total_time": self._total_generation_time,
                "average_time": (
                    self._total_generation_time / self._generation_count 
                    if self._generation_count > 0 else 0
                )
            }
        }
    
    def get_generator_info(self) -> Dict[str, Any]:
        """Get information about this generator."""
        return {
            "artifact_type": self.get_artifact_type(),
            "template_name": self.get_template_name(),
            "dataflow_enabled": self.config.is_dataflow_enabled(),
            "generator_type": self.config.generator_type.value,
            "statistics": {
                "generation_count": self._generation_count,
                "total_generation_time": self._total_generation_time,
                "last_generation_time": self._last_generation_time
            }
        }


class DataflowAwareGenerator(GeneratorBase):
    """
    Base class for generators that use dataflow modeling.
    
    This class provides additional functionality for generators that
    integrate with the dataflow modeling system.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize dataflow-aware generator."""
        super().__init__(*args, **kwargs)
        
        if not DATAFLOW_AVAILABLE:
            raise CodeGenerationError(
                "Dataflow modeling not available",
                generator_type=self.get_artifact_type(),
                suggestion="Install dataflow dependencies or disable dataflow mode"
            )
    
    def _build_context(self, inputs: Dict[str, Any]) -> DataflowContext:
        """Build dataflow context."""
        context = super()._build_context(inputs)
        
        if not isinstance(context, DataflowContext):
            raise CodeGenerationError(
                "Dataflow context required for dataflow-aware generator",
                generator_type=self.get_artifact_type(),
                suggestion="Enable dataflow mode or use legacy generator"
            )
        
        return context
    
    def _validate_dataflow_requirements(self, context: DataflowContext) -> ValidationResult:
        """Validate dataflow-specific requirements."""
        if DATAFLOW_AVAILABLE:
            result = create_validation_result()
        else:
            result = {"errors": [], "warnings": [], "success": True}
        
        # Check for required interfaces
        if not context.has_axi_interfaces():
            error = {
                "component": "DataflowAwareGenerator",
                "error_type": "missing_interfaces",
                "message": "No AXI interfaces found in dataflow context",
                "context": {"generator_type": self.get_artifact_type()}
            }
            if DATAFLOW_AVAILABLE:
                result.add_error(error)
            else:
                result["errors"].append(error)
                result["success"] = False
        
        # Validate dataflow model if available
        if context.dataflow_model:
            try:
                model_result = context.dataflow_model.validate_mathematical_constraints()
                if DATAFLOW_AVAILABLE:
                    result.merge(model_result)
                else:
                    result["errors"].extend(model_result.get("errors", []))
                    result["warnings"].extend(model_result.get("warnings", []))
                    if not model_result.get("success", True):
                        result["success"] = False
            except Exception as e:
                error = {
                    "component": "DataflowAwareGenerator",
                    "error_type": "model_validation_failed",
                    "message": f"Dataflow model validation failed: {e}",
                    "context": {"generator_type": self.get_artifact_type()}
                }
                if DATAFLOW_AVAILABLE:
                    result.add_error(error)
                else:
                    result["errors"].append(error)
                    result["success"] = False
        
        return result


# Factory functions
def create_generation_result() -> GenerationResult:
    """Create a new generation result."""
    return GenerationResult()


def create_artifact(
    file_name: str,
    content: str,
    artifact_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> GeneratedArtifact:
    """Create a generated artifact."""
    return GeneratedArtifact(
        file_name=file_name,
        content=content,
        artifact_type=artifact_type,
        metadata=metadata or {}
    )