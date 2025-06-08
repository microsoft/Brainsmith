"""
Enhanced Data Structures for Hardware Kernel Generator Pipeline.

This module provides comprehensive data structures that support the enhanced
HWKG pipeline with dataflow integration, validation, and rich metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from pathlib import Path
from enum import Enum
import time
from datetime import datetime

# Import dataflow components
try:
    from ...dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
    from ...dataflow.core.dataflow_model import DataflowModel
    from ...dataflow.core.validation import ValidationResult, create_validation_result
    DATAFLOW_AVAILABLE = True
except ImportError:
    DATAFLOW_AVAILABLE = False
    # Create placeholder types
    class DataflowInterface: pass
    class DataflowInterfaceType: pass
    class DataflowModel: pass
    ValidationResult = Dict[str, Any]

from .enhanced_config import PipelineConfig, GeneratorType
from .enhanced_generator_base import GenerationResult


class PipelineStage(Enum):
    """Enumeration of pipeline execution stages."""
    INITIALIZATION = "initialization"
    RTL_PARSING = "rtl_parsing"
    COMPILER_DATA_PARSING = "compiler_data_parsing"
    INTERFACE_ANALYSIS = "interface_analysis"
    PRAGMA_PROCESSING = "pragma_processing"
    DATAFLOW_MODEL_BUILDING = "dataflow_model_building"
    TEMPLATE_CONTEXT_BUILDING = "template_context_building"
    CODE_GENERATION = "code_generation"
    VALIDATION = "validation"
    OUTPUT_WRITING = "output_writing"
    CLEANUP = "cleanup"
    COMPLETE = "complete"
    ERROR = "error"


class ProcessingStatus(Enum):
    """Status of processing operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RTLSignal:
    """Enhanced representation of an RTL signal."""
    
    # Basic signal information
    name: str
    direction: str  # input, output, inout
    width: int
    signal_type: str = "wire"  # wire, reg, logic, etc.
    
    # Extended metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_location: Optional[Dict[str, Any]] = None  # file, line, column
    
    # Signal classification
    is_clock: bool = False
    is_reset: bool = False
    is_control: bool = False
    is_data: bool = True
    
    # Interface association
    interface_name: Optional[str] = None
    interface_role: Optional[str] = None  # tdata, tvalid, tready, etc.
    
    # Dataflow integration
    dataflow_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Auto-detect signal characteristics
        if not any([self.is_clock, self.is_reset, self.is_control]):
            self._classify_signal()
    
    def _classify_signal(self) -> None:
        """Automatically classify signal based on name patterns."""
        name_lower = self.name.lower()
        
        # Clock detection
        if any(pattern in name_lower for pattern in ['clk', 'clock', 'ck']):
            self.is_clock = True
            self.is_data = False
            self.is_control = True
        
        # Reset detection
        elif any(pattern in name_lower for pattern in ['rst', 'reset', 'aresetn']):
            self.is_reset = True
            self.is_data = False
            self.is_control = True
        
        # Control signal detection
        elif any(pattern in name_lower for pattern in ['valid', 'ready', 'enable', 'start', 'done']):
            self.is_control = True
            self.is_data = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary representation."""
        return {
            "name": self.name,
            "direction": self.direction,
            "width": self.width,
            "signal_type": self.signal_type,
            "is_clock": self.is_clock,
            "is_reset": self.is_reset,
            "is_control": self.is_control,
            "is_data": self.is_data,
            "interface_name": self.interface_name,
            "interface_role": self.interface_role,
            "attributes": self.attributes,
            "source_location": self.source_location,
            "dataflow_metadata": self.dataflow_metadata
        }


@dataclass
class RTLInterface:
    """Enhanced representation of an RTL interface."""
    
    # Basic interface information
    name: str
    interface_type: str  # axi_stream, axi_lite, custom, etc.
    signals: List[RTLSignal] = field(default_factory=list)
    
    # Interface metadata
    parameters: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Classification and direction
    direction: Optional[str] = None  # input, output, inout
    protocol_version: Optional[str] = None
    
    # Dataflow integration
    dataflow_interface: Optional[DataflowInterface] = None
    conversion_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validation_result: Optional[ValidationResult] = None
    is_validated: bool = False
    
    def get_signals_by_direction(self, direction: str) -> List[RTLSignal]:
        """Get signals with specific direction."""
        return [signal for signal in self.signals if signal.direction == direction]
    
    def get_signals_by_role(self, role: str) -> List[RTLSignal]:
        """Get signals with specific interface role."""
        return [signal for signal in self.signals if signal.interface_role == role]
    
    def get_data_width(self) -> int:
        """Calculate total data width for the interface."""
        data_signals = [sig for sig in self.signals if sig.is_data]
        return sum(sig.width for sig in data_signals)
    
    def validate(self) -> ValidationResult:
        """Validate interface structure and completeness."""
        if DATAFLOW_AVAILABLE:
            self.validation_result = create_validation_result()
        else:
            self.validation_result = {"errors": [], "warnings": [], "success": True}
        
        # Validate required signals for interface type
        if self.interface_type == "axi_stream":
            self._validate_axi_stream()
        elif self.interface_type == "axi_lite":
            self._validate_axi_lite()
        
        self.is_validated = True
        return self.validation_result
    
    def _validate_axi_stream(self) -> None:
        """Validate AXI-Stream interface requirements."""
        required_signals = ["tdata", "tvalid", "tready"]
        signal_roles = {sig.interface_role for sig in self.signals if sig.interface_role}
        
        missing_signals = [role for role in required_signals if role not in signal_roles]
        if missing_signals:
            error_message = f"AXI-Stream interface '{self.name}' missing required signals: {missing_signals}"
            error_context = {"interface_type": self.interface_type, "missing": missing_signals}
            
            if DATAFLOW_AVAILABLE:
                from ...dataflow.core.validation import ValidationError, ValidationSeverity
                error = ValidationError(
                    component="RTLInterface",
                    error_type="missing_signals",
                    message=error_message,
                    severity=ValidationSeverity.ERROR,
                    context=error_context
                )
                self.validation_result.add_error(error)
            else:
                error = {
                    "component": "RTLInterface",
                    "error_type": "missing_signals",
                    "message": error_message,
                    "context": error_context
                }
                self.validation_result["errors"].append(error)
                self.validation_result["success"] = False
    
    def _validate_axi_lite(self) -> None:
        """Validate AXI-Lite interface requirements."""
        required_signals = ["awaddr", "awvalid", "awready", "wdata", "wvalid", "wready"]
        signal_roles = {sig.interface_role for sig in self.signals if sig.interface_role}
        
        missing_signals = [role for role in required_signals if role not in signal_roles]
        if missing_signals:
            error_message = f"AXI-Lite interface '{self.name}' missing required signals: {missing_signals}"
            error_context = {"interface_type": self.interface_type, "missing": missing_signals}
            
            if DATAFLOW_AVAILABLE:
                from ...dataflow.core.validation import ValidationError, ValidationSeverity
                error = ValidationError(
                    component="RTLInterface",
                    error_type="missing_signals",
                    message=error_message,
                    severity=ValidationSeverity.ERROR,
                    context=error_context
                )
                self.validation_result.add_error(error)
            else:
                error = {
                    "component": "RTLInterface",
                    "error_type": "missing_signals", 
                    "message": error_message,
                    "context": error_context
                }
                self.validation_result["errors"].append(error)
                self.validation_result["success"] = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interface to dictionary representation."""
        return {
            "name": self.name,
            "interface_type": self.interface_type,
            "direction": self.direction,
            "protocol_version": self.protocol_version,
            "signal_count": len(self.signals),
            "signals": [signal.to_dict() for signal in self.signals],
            "data_width": self.get_data_width(),
            "parameters": self.parameters,
            "attributes": self.attributes,
            "is_validated": self.is_validated,
            "validation_success": (
                self.validation_result.success if DATAFLOW_AVAILABLE and self.validation_result
                else self.validation_result.get("success", True) if self.validation_result
                else None
            ),
            "conversion_metadata": self.conversion_metadata
        }


@dataclass
class RTLModule:
    """Enhanced representation of an RTL module."""
    
    # Basic module information
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    interfaces: List[RTLInterface] = field(default_factory=list)
    
    # Additional signals not part of interfaces
    internal_signals: List[RTLSignal] = field(default_factory=list)
    
    # Module metadata
    source_file: Optional[Path] = None
    module_type: str = "hardware_kernel"  # hardware_kernel, wrapper, utility
    description: Optional[str] = None
    
    # Pragmas and directives
    pragmas: List[Dict[str, Any]] = field(default_factory=list)
    directives: List[Dict[str, Any]] = field(default_factory=list)
    
    # Dataflow integration
    dataflow_model: Optional[DataflowModel] = None
    
    def get_interface(self, name: str) -> Optional[RTLInterface]:
        """Get interface by name."""
        for interface in self.interfaces:
            if interface.name == name:
                return interface
        return None
    
    def get_interfaces_by_type(self, interface_type: str) -> List[RTLInterface]:
        """Get interfaces of specific type."""
        return [iface for iface in self.interfaces if iface.interface_type == interface_type]
    
    def get_all_signals(self) -> List[RTLSignal]:
        """Get all signals from interfaces and internal signals."""
        all_signals = list(self.internal_signals)
        for interface in self.interfaces:
            all_signals.extend(interface.signals)
        return all_signals
    
    def get_pragmas_by_type(self, pragma_type: str) -> List[Dict[str, Any]]:
        """Get pragmas of specific type."""
        return [pragma for pragma in self.pragmas if pragma.get("type") == pragma_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert module to dictionary representation."""
        return {
            "name": self.name,
            "module_type": self.module_type,
            "description": self.description,
            "source_file": str(self.source_file) if self.source_file else None,
            "parameter_count": len(self.parameters),
            "parameters": self.parameters,
            "interface_count": len(self.interfaces),
            "interfaces": [iface.to_dict() for iface in self.interfaces],
            "internal_signal_count": len(self.internal_signals),
            "pragma_count": len(self.pragmas),
            "pragmas": self.pragmas,
            "directive_count": len(self.directives),
            "directives": self.directives
        }


@dataclass
class ParsedRTLData:
    """Container for parsed RTL data with validation and metadata."""
    
    # Parsed content
    modules: List[RTLModule] = field(default_factory=list)
    top_module: Optional[str] = None
    
    # Parsing metadata
    source_files: List[Path] = field(default_factory=list)
    parsing_time: float = 0.0
    parser_version: str = "2.0.0"
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    parsing_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validation_result: Optional[ValidationResult] = None
    is_validated: bool = False
    
    def get_top_module(self) -> Optional[RTLModule]:
        """Get the top-level module."""
        if self.top_module:
            for module in self.modules:
                if module.name == self.top_module:
                    return module
        
        # If no explicit top module, return first module
        return self.modules[0] if self.modules else None
    
    def get_module(self, name: str) -> Optional[RTLModule]:
        """Get module by name."""
        for module in self.modules:
            if module.name == name:
                return module
        return None
    
    def validate(self) -> ValidationResult:
        """Validate parsed RTL data."""
        if DATAFLOW_AVAILABLE:
            self.validation_result = create_validation_result()
        else:
            self.validation_result = {"errors": [], "warnings": [], "success": True}
        
        # Validate modules
        for module in self.modules:
            for interface in module.interfaces:
                interface_result = interface.validate()
                if DATAFLOW_AVAILABLE:
                    self.validation_result.merge(interface_result)
                else:
                    self.validation_result["errors"].extend(interface_result.get("errors", []))
                    self.validation_result["warnings"].extend(interface_result.get("warnings", []))
                    if not interface_result.get("success", True):
                        self.validation_result["success"] = False
        
        # Validate top module exists
        if not self.get_top_module():
            error = {
                "component": "ParsedRTLData",
                "error_type": "missing_top_module",
                "message": "No top module found in parsed RTL data",
                "context": {"module_count": len(self.modules)}
            }
            if DATAFLOW_AVAILABLE:
                self.validation_result.add_error(error)
            else:
                self.validation_result["errors"].append(error)
                self.validation_result["success"] = False
        
        self.is_validated = True
        return self.validation_result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "module_count": len(self.modules),
            "modules": [module.to_dict() for module in self.modules],
            "top_module": self.top_module,
            "source_file_count": len(self.source_files),
            "source_files": [str(f) for f in self.source_files],
            "parsing_time": self.parsing_time,
            "parser_version": self.parser_version,
            "is_validated": self.is_validated,
            "validation_success": (
                self.validation_result.success if DATAFLOW_AVAILABLE and self.validation_result
                else self.validation_result.get("success", True) if self.validation_result
                else None
            ),
            "metadata": self.metadata,
            "parsing_statistics": self.parsing_statistics
        }


@dataclass
class CompilerData:
    """Enhanced compiler data with validation and metadata."""
    
    # Core compiler data
    function_name: str
    domain: str = "finn"
    
    # Function parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Datatype information
    input_datatypes: List[str] = field(default_factory=list)
    output_datatypes: List[str] = field(default_factory=list)
    weight_datatypes: List[str] = field(default_factory=list)
    
    # Shape information
    input_shapes: List[List[int]] = field(default_factory=list)
    output_shapes: List[List[int]] = field(default_factory=list)
    weight_shapes: List[List[int]] = field(default_factory=list)
    
    # Parallelism configuration
    parallelism_config: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization settings
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    source_file: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "function_name": self.function_name,
            "domain": self.domain,
            "parameters": self.parameters,
            "input_datatypes": self.input_datatypes,
            "output_datatypes": self.output_datatypes,
            "weight_datatypes": self.weight_datatypes,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "weight_shapes": self.weight_shapes,
            "parallelism_config": self.parallelism_config,
            "optimization_config": self.optimization_config,
            "source_file": str(self.source_file) if self.source_file else None,
            "metadata": self.metadata
        }


@dataclass
class PipelineInputs:
    """Input specification for pipeline execution."""
    
    # Required inputs
    rtl_file_path: Path
    compiler_data_path: Path
    config: PipelineConfig
    
    # Optional inputs
    custom_doc_path: Optional[Path] = None
    onnx_metadata: Optional[Dict[str, Any]] = None
    output_dir: Optional[Path] = None
    
    # Processing options
    skip_validation: bool = False
    force_regeneration: bool = False
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string paths to Path objects
        if isinstance(self.rtl_file_path, str):
            self.rtl_file_path = Path(self.rtl_file_path)
        if isinstance(self.compiler_data_path, str):
            self.compiler_data_path = Path(self.compiler_data_path)
        if isinstance(self.custom_doc_path, str):
            self.custom_doc_path = Path(self.custom_doc_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Set output directory from config if not provided
        if self.output_dir is None:
            self.output_dir = self.config.generation.output_dir
    
    def validate(self) -> ValidationResult:
        """Validate pipeline inputs."""
        if DATAFLOW_AVAILABLE:
            result = create_validation_result()
        else:
            result = {"errors": [], "warnings": [], "success": True}
        
        # Check required files exist
        if not self.rtl_file_path.exists():
            error = {
                "component": "PipelineInputs",
                "error_type": "missing_file",
                "message": f"RTL file not found: {self.rtl_file_path}",
                "context": {"file_type": "rtl", "path": str(self.rtl_file_path)}
            }
            if DATAFLOW_AVAILABLE:
                result.add_error(error)
            else:
                result["errors"].append(error)
                result["success"] = False
        
        if not self.compiler_data_path.exists():
            error = {
                "component": "PipelineInputs",
                "error_type": "missing_file",
                "message": f"Compiler data file not found: {self.compiler_data_path}",
                "context": {"file_type": "compiler_data", "path": str(self.compiler_data_path)}
            }
            if DATAFLOW_AVAILABLE:
                result.add_error(error)
            else:
                result["errors"].append(error)
                result["success"] = False
        
        # Validate configuration
        try:
            self.config.validate()
        except Exception as e:
            error = {
                "component": "PipelineInputs",
                "error_type": "config_validation",
                "message": f"Configuration validation failed: {e}",
                "context": {"config_section": "unknown"}
            }
            if DATAFLOW_AVAILABLE:
                result.add_error(error)
            else:
                result["errors"].append(error)
                result["success"] = False
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rtl_file_path": str(self.rtl_file_path),
            "compiler_data_path": str(self.compiler_data_path),
            "custom_doc_path": str(self.custom_doc_path) if self.custom_doc_path else None,
            "output_dir": str(self.output_dir),
            "skip_validation": self.skip_validation,
            "force_regeneration": self.force_regeneration,
            "onnx_metadata": self.onnx_metadata,
            "config": self.config.to_dict(),
            "metadata": self.metadata
        }


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""
    
    # Stage information
    stage: PipelineStage
    status: ProcessingStatus
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    # Results
    output_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Status information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def complete(self, output_data: Optional[Any] = None) -> None:
        """Mark stage as completed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = ProcessingStatus.COMPLETED
        if output_data is not None:
            self.output_data = output_data
    
    def fail(self, error_message: str) -> None:
        """Mark stage as failed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = ProcessingStatus.FAILED
        self.error_message = error_message
    
    def skip(self, reason: str) -> None:
        """Mark stage as skipped."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = ProcessingStatus.SKIPPED
        self.metadata["skip_reason"] = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "error_message": self.error_message,
            "warning_count": len(self.warnings),
            "warnings": self.warnings,
            "metadata": self.metadata,
            "has_output": self.output_data is not None
        }


@dataclass
class PipelineResults:
    """Complete results of pipeline execution with comprehensive tracking."""
    
    # Core results
    parsed_rtl: Optional[ParsedRTLData] = None
    compiler_data: Optional[CompilerData] = None
    dataflow_model: Optional[DataflowModel] = None
    generated_artifacts: List[Any] = field(default_factory=list)  # GeneratedArtifact list
    
    # Processing stages
    stage_results: Dict[PipelineStage, StageResult] = field(default_factory=dict)
    current_stage: PipelineStage = PipelineStage.INITIALIZATION
    
    # Overall status
    success: bool = False
    complete: bool = False
    
    # Timing
    pipeline_start_time: float = field(default_factory=time.time)
    pipeline_end_time: Optional[float] = None
    total_duration: Optional[float] = None
    
    # Output information
    output_files: List[Path] = field(default_factory=list)
    output_directory: Optional[Path] = None
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metrics and statistics
    metrics: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    
    def start_stage(self, stage: PipelineStage) -> StageResult:
        """Start a new pipeline stage."""
        self.current_stage = stage
        stage_result = StageResult(stage=stage, status=ProcessingStatus.IN_PROGRESS)
        self.stage_results[stage] = stage_result
        return stage_result
    
    def complete_pipeline(self, success: bool = True) -> None:
        """Mark pipeline as completed."""
        self.pipeline_end_time = time.time()
        self.total_duration = self.pipeline_end_time - self.pipeline_start_time
        self.success = success
        self.complete = True
        self.current_stage = PipelineStage.COMPLETE if success else PipelineStage.ERROR
        
        # Calculate performance statistics
        self._calculate_performance_stats()
    
    def add_error(self, error: str) -> None:
        """Add error message."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str) -> None:
        """Add warning message."""
        self.warnings.append(warning)
    
    def add_output_file(self, file_path: Path) -> None:
        """Add output file to results."""
        self.output_files.append(file_path)
    
    def _calculate_performance_stats(self) -> None:
        """Calculate performance statistics."""
        completed_stages = [
            result for result in self.stage_results.values()
            if result.status == ProcessingStatus.COMPLETED and result.duration is not None
        ]
        
        if completed_stages:
            durations = [result.duration for result in completed_stages]
            self.performance_stats = {
                "completed_stages": len(completed_stages),
                "total_stage_time": sum(durations),
                "average_stage_time": sum(durations) / len(durations),
                "longest_stage": max(durations),
                "shortest_stage": min(durations),
                "stage_breakdown": {
                    result.stage.value: result.duration 
                    for result in completed_stages
                }
            }
    
    def get_stage_result(self, stage: PipelineStage) -> Optional[StageResult]:
        """Get result for specific stage."""
        return self.stage_results.get(stage)
    
    def get_failed_stages(self) -> List[StageResult]:
        """Get all failed stages."""
        return [
            result for result in self.stage_results.values()
            if result.status == ProcessingStatus.FAILED
        ]
    
    def get_completion_percentage(self) -> float:
        """Calculate pipeline completion percentage."""
        total_stages = len(PipelineStage) - 2  # Exclude COMPLETE and ERROR
        completed_stages = len([
            result for result in self.stage_results.values()
            if result.status in [ProcessingStatus.COMPLETED, ProcessingStatus.SKIPPED]
        ])
        return (completed_stages / total_stages) * 100 if total_stages > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to comprehensive dictionary representation."""
        return {
            "success": self.success,
            "complete": self.complete,
            "current_stage": self.current_stage.value,
            "completion_percentage": self.get_completion_percentage(),
            
            # Timing
            "pipeline_start_time": self.pipeline_start_time,
            "pipeline_end_time": self.pipeline_end_time,
            "total_duration": self.total_duration,
            
            # Results summary
            "parsed_rtl_available": self.parsed_rtl is not None,
            "compiler_data_available": self.compiler_data is not None,
            "dataflow_model_available": self.dataflow_model is not None,
            "artifact_count": len(self.generated_artifacts),
            "output_file_count": len(self.output_files),
            
            # Error tracking
            "error_count": len(self.errors),
            "errors": self.errors,
            "warning_count": len(self.warnings),
            "warnings": self.warnings,
            
            # Stage information
            "stage_count": len(self.stage_results),
            "stage_results": {
                stage.value: result.to_dict() 
                for stage, result in self.stage_results.items()
            },
            "failed_stage_count": len(self.get_failed_stages()),
            
            # Performance
            "performance_stats": self.performance_stats,
            "metrics": self.metrics,
            
            # Output
            "output_directory": str(self.output_directory) if self.output_directory else None,
            "output_files": [str(f) for f in self.output_files]
        }


# Factory functions
def create_pipeline_inputs(
    rtl_file: Union[str, Path],
    compiler_data: Union[str, Path],
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> PipelineInputs:
    """Create pipeline inputs with validation."""
    if config is None:
        from .enhanced_config import create_default_config
        config = create_default_config()
    
    return PipelineInputs(
        rtl_file_path=Path(rtl_file),
        compiler_data_path=Path(compiler_data),
        config=config,
        **kwargs
    )


def create_pipeline_results() -> PipelineResults:
    """Create a new pipeline results container."""
    return PipelineResults()