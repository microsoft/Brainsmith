"""
Base generator interface and data structures for the Hardware Kernel Generator.

This module provides the common interface that all generators must implement,
along with shared data structures for generation results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from enum import Enum
import datetime

from .errors import CodeGenerationError, ValidationError
from .template_context import BaseContext


class GenerationStatus(Enum):
    """Status of generation operation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactType(Enum):
    """Types of generated artifacts."""
    PYTHON_FILE = "python_file"
    HEADER_FILE = "header_file"
    CONFIG_FILE = "config_file"
    DOCUMENTATION = "documentation"
    TEST_FILE = "test_file"
    BUILD_SCRIPT = "build_script"
    WRAPPER_FILE = "wrapper_file"


@dataclass
class GeneratedArtifact:
    """Represents a single generated file or artifact."""
    
    # Basic information
    name: str
    type: ArtifactType
    content: str
    
    # File information
    file_path: Optional[Path] = None
    encoding: str = "utf-8"
    
    # Generation metadata
    template_name: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    # Validation and quality
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Size and complexity metrics
    line_count: int = 0
    size_bytes: int = 0
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        if self.content:
            self.line_count = len(self.content.splitlines())
            self.size_bytes = len(self.content.encode(self.encoding))
        
        # Set default file path if not provided
        if not self.file_path and self.name:
            if self.type == ArtifactType.PYTHON_FILE:
                self.file_path = Path(f"{self.name}.py")
            elif self.type == ArtifactType.HEADER_FILE:
                self.file_path = Path(f"{self.name}.h")
            elif self.type == ArtifactType.CONFIG_FILE:
                self.file_path = Path(f"{self.name}.json")
            elif self.type == ArtifactType.DOCUMENTATION:
                self.file_path = Path(f"{self.name}.md")
            elif self.type == ArtifactType.TEST_FILE:
                self.file_path = Path(f"test_{self.name}.py")
            elif self.type == ArtifactType.BUILD_SCRIPT:
                self.file_path = Path(f"{self.name}.sh")
            elif self.type == ArtifactType.WRAPPER_FILE:
                self.file_path = Path(f"{self.name}_wrapper.py")
            else:
                self.file_path = Path(self.name)
    
    def write_to_file(self, output_dir: Optional[Path] = None, overwrite: bool = False) -> None:
        """Write artifact content to file.
        
        Args:
            output_dir: Directory to write file to, uses file_path if None
            overwrite: Whether to overwrite existing files
            
        Raises:
            CodeGenerationError: If file operations fail
        """
        if not self.file_path:
            raise CodeGenerationError(
                f"No file path specified for artifact '{self.name}'",
                suggestion="Set file_path or provide output_dir"
            )
        
        # Determine final file path
        if output_dir:
            final_path = output_dir / self.file_path.name
        else:
            final_path = self.file_path
        
        # Check for existing file
        if final_path.exists() and not overwrite:
            raise CodeGenerationError(
                f"File already exists: {final_path}",
                suggestion="Set overwrite=True or choose different path"
            )
        
        try:
            # Create directory if needed
            final_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            with open(final_path, 'w', encoding=self.encoding) as f:
                f.write(self.content)
                
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to write artifact '{self.name}' to {final_path}: {e}",
                suggestion="Check file permissions and disk space"
            )
    
    def validate_content(self) -> bool:
        """Validate artifact content.
        
        Returns:
            True if content is valid, False otherwise
        """
        self.validation_errors.clear()
        self.warnings.clear()
        
        # Basic validation
        if not self.content.strip():
            self.validation_errors.append("Artifact content is empty")
            self.is_valid = False
            return False
        
        # Type-specific validation
        if self.type == ArtifactType.PYTHON_FILE:
            self._validate_python_content()
        elif self.type == ArtifactType.CONFIG_FILE:
            self._validate_config_content()
        
        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid
    
    def _validate_python_content(self) -> None:
        """Validate Python file content."""
        try:
            compile(self.content, self.name, 'exec')
        except SyntaxError as e:
            self.validation_errors.append(f"Python syntax error: {e}")
        except Exception as e:
            self.validation_errors.append(f"Python compilation error: {e}")
    
    def _validate_config_content(self) -> None:
        """Validate configuration file content."""
        try:
            import json
            json.loads(self.content)
        except json.JSONDecodeError as e:
            self.validation_errors.append(f"JSON syntax error: {e}")
        except Exception as e:
            self.validation_errors.append(f"JSON parsing error: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary."""
        data = asdict(self)
        
        # Convert Path objects to strings
        if self.file_path:
            data['file_path'] = str(self.file_path)
        
        # Convert enum to string
        data['type'] = self.type.value
        
        return data


@dataclass
class GenerationResult:
    """Result of a generation operation."""
    
    # Operation status
    status: GenerationStatus
    message: str = ""
    
    # Generated artifacts
    artifacts: List[GeneratedArtifact] = field(default_factory=list)
    
    # Performance metrics
    generation_time: float = 0.0  # seconds
    template_count: int = 0
    total_lines: int = 0
    total_size: int = 0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Generation metadata
    generator_type: str = ""
    context_summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def __post_init__(self):
        """Initialize derived fields."""
        self._update_metrics()
    
    def add_artifact(self, artifact: GeneratedArtifact) -> None:
        """Add an artifact to the result."""
        self.artifacts.append(artifact)
        self._update_metrics()
    
    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        if self.status == GenerationStatus.SUCCESS:
            self.status = GenerationStatus.PARTIAL
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def _update_metrics(self) -> None:
        """Update performance metrics from artifacts."""
        self.template_count = len(set(a.template_name for a in self.artifacts if a.template_name))
        self.total_lines = sum(a.line_count for a in self.artifacts)
        self.total_size = sum(a.size_bytes for a in self.artifacts)
    
    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> List[GeneratedArtifact]:
        """Get artifacts of a specific type."""
        return [a for a in self.artifacts if a.type == artifact_type]
    
    def has_errors(self) -> bool:
        """Check if result has any errors."""
        return len(self.errors) > 0 or any(a.validation_errors for a in self.artifacts)
    
    def has_warnings(self) -> bool:
        """Check if result has any warnings."""
        return len(self.warnings) > 0 or any(a.warnings for a in self.artifacts)
    
    def validate_all_artifacts(self) -> bool:
        """Validate all artifacts in the result."""
        all_valid = True
        
        for artifact in self.artifacts:
            if not artifact.validate_content():
                all_valid = False
                self.errors.extend(artifact.validation_errors)
                self.warnings.extend(artifact.warnings)
        
        if not all_valid and self.status == GenerationStatus.SUCCESS:
            self.status = GenerationStatus.PARTIAL
        
        return all_valid
    
    def write_all_artifacts(self, output_dir: Path, overwrite: bool = False) -> List[Path]:
        """Write all artifacts to files.
        
        Args:
            output_dir: Directory to write files to
            overwrite: Whether to overwrite existing files
            
        Returns:
            List of written file paths
            
        Raises:
            CodeGenerationError: If any file operations fail
        """
        written_files = []
        
        for artifact in self.artifacts:
            try:
                artifact.write_to_file(output_dir, overwrite)
                if artifact.file_path:
                    written_files.append(output_dir / artifact.file_path.name)
            except CodeGenerationError as e:
                self.add_error(str(e))
                if self.status == GenerationStatus.SUCCESS:
                    self.status = GenerationStatus.PARTIAL
        
        return written_files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        data = asdict(self)
        
        # Convert enum to string
        data['status'] = self.status.value
        
        # Convert artifacts
        data['artifacts'] = [a.to_dict() for a in self.artifacts]
        
        return data


class GeneratorBase(ABC):
    """Abstract base class for all generators."""
    
    def __init__(self, config=None, template_manager=None, context_builder=None):
        """Initialize generator with dependencies.
        
        Args:
            config: Pipeline configuration
            template_manager: Template manager instance
            context_builder: Template context builder instance
        """
        self.config = config
        self.template_manager = template_manager
        self.context_builder = context_builder
        
        # Generator metadata
        self.generator_type = self.__class__.__name__
        self.version = "1.0.0"
        
        # Performance tracking
        self._generation_start_time = 0.0
    
    @abstractmethod
    def generate(self, analysis_data: Dict[str, Any], **kwargs) -> GenerationResult:
        """Generate code artifacts from analysis data.
        
        Args:
            analysis_data: Parsed and analyzed data from RTL/pragmas
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with generated artifacts
            
        Raises:
            CodeGenerationError: If generation fails
        """
        pass
    
    @abstractmethod
    def get_supported_templates(self) -> List[str]:
        """Get list of templates this generator supports.
        
        Returns:
            List of template names
        """
        pass
    
    @abstractmethod
    def validate_input(self, analysis_data: Dict[str, Any]) -> None:
        """Validate input data for generation.
        
        Args:
            analysis_data: Data to validate
            
        Raises:
            ValidationError: If data is invalid
        """
        pass
    
    def create_artifact(self, name: str, artifact_type: ArtifactType, 
                       content: str, template_name: str = None, 
                       context_data: Dict[str, Any] = None) -> GeneratedArtifact:
        """Create a new artifact with standard metadata.
        
        Args:
            name: Artifact name
            artifact_type: Type of artifact
            content: Generated content
            template_name: Template used for generation
            context_data: Context data used
            
        Returns:
            GeneratedArtifact instance
        """
        artifact = GeneratedArtifact(
            name=name,
            type=artifact_type,
            content=content,
            template_name=template_name,
            context_data=context_data or {}
        )
        
        # Validate the artifact
        artifact.validate_content()
        
        return artifact
    
    def start_generation_timer(self) -> None:
        """Start timing generation operation."""
        import time
        self._generation_start_time = time.time()
    
    def get_generation_time(self) -> float:
        """Get elapsed generation time in seconds."""
        import time
        if self._generation_start_time > 0:
            return time.time() - self._generation_start_time
        return 0.0
    
    def create_result(self, status: GenerationStatus, message: str = "") -> GenerationResult:
        """Create a new generation result with standard metadata.
        
        Args:
            status: Generation status
            message: Status message
            
        Returns:
            GenerationResult instance
        """
        return GenerationResult(
            status=status,
            message=message,
            generation_time=self.get_generation_time(),
            generator_type=self.generator_type
        )
    
    def render_template(self, template_name: str, context: BaseContext) -> str:
        """Render a template with context.
        
        Args:
            template_name: Name of template to render
            context: Template context
            
        Returns:
            Rendered content
            
        Raises:
            CodeGenerationError: If template rendering fails
        """
        if not self.template_manager:
            raise CodeGenerationError(
                "Template manager not initialized",
                generator_type=self.generator_type,
                suggestion="Initialize generator with template_manager"
            )
        
        try:
            return self.template_manager.render_template(template_name, context.to_dict())
        except Exception as e:
            raise CodeGenerationError(
                f"Failed to render template '{template_name}': {e}",
                generator_type=self.generator_type,
                template_name=template_name,
                suggestion="Check template syntax and context data"
            )
    
    def build_context(self, analysis_data: Dict[str, Any], **kwargs) -> BaseContext:
        """Build template context from analysis data.
        
        Args:
            analysis_data: Analyzed data
            **kwargs: Additional context parameters
            
        Returns:
            Template context instance
            
        Raises:
            CodeGenerationError: If context building fails
        """
        if not self.context_builder:
            raise CodeGenerationError(
                "Context builder not initialized",
                generator_type=self.generator_type,
                suggestion="Initialize generator with context_builder"
            )
        
        # This is overridden by specific generators
        raise NotImplementedError("Subclasses must implement build_context")
    
    def get_generator_info(self) -> Dict[str, Any]:
        """Get generator information and capabilities.
        
        Returns:
            Generator metadata
        """
        return {
            'type': self.generator_type,
            'version': self.version,
            'supported_templates': self.get_supported_templates(),
            'config_required': self.config is not None,
            'template_manager_required': self.template_manager is not None,
            'context_builder_required': self.context_builder is not None
        }


def create_generation_result(status: GenerationStatus, message: str = "", 
                           generator_type: str = "") -> GenerationResult:
    """Factory function to create generation results."""
    return GenerationResult(
        status=status,
        message=message,
        generator_type=generator_type
    )


def create_artifact(name: str, artifact_type: ArtifactType, content: str) -> GeneratedArtifact:
    """Factory function to create artifacts."""
    return GeneratedArtifact(
        name=name,
        type=artifact_type,
        content=content
    )