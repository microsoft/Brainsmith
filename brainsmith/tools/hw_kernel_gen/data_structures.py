"""
Data structures for the Hardware Kernel Generator pipeline.

This module defines the standardized data structures used throughout the
pipeline for passing data between components and tracking analysis results.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union, Set
from pathlib import Path
from enum import Enum
import datetime

from .errors import ValidationError


class PipelineStage(Enum):
    """Pipeline execution stages."""
    INITIALIZATION = "initialization"
    RTL_PARSING = "rtl_parsing" 
    PRAGMA_PROCESSING = "pragma_processing"
    INTERFACE_ANALYSIS = "interface_analysis"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    CONTEXT_BUILDING = "context_building"
    CODE_GENERATION = "code_generation"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


class InterfaceType(Enum):
    """Types of hardware interfaces."""
    CLOCK = "clock"
    RESET = "reset"
    CONTROL = "control"
    DATA = "data"
    AXI_STREAM = "axi_stream"
    AXI_LITE = "axi_lite"
    AXI_FULL = "axi_full"
    CUSTOM = "custom"


class SignalDirection(Enum):
    """Signal directions."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


@dataclass
class RTLSignal:
    """Represents a single RTL signal."""
    
    name: str
    direction: SignalDirection
    width: Optional[int] = None
    type: str = "wire"
    description: str = ""
    
    # Signal classification
    interface_type: InterfaceType = InterfaceType.DATA
    is_vector: bool = False
    
    # Source information
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields."""
        # Determine if signal is a vector
        if self.width is not None and self.width > 1:
            self.is_vector = True
        
        # Auto-classify signal type based on name
        name_lower = self.name.lower()
        
        if any(clk in name_lower for clk in ['clk', 'clock']):
            self.interface_type = InterfaceType.CLOCK
        elif any(rst in name_lower for rst in ['rst', 'reset']):
            self.interface_type = InterfaceType.RESET
        elif any(ctrl in name_lower for ctrl in ['ap_', 'ctrl', 'control', 'start', 'done', 'idle', 'ready']):
            self.interface_type = InterfaceType.CONTROL
        elif 's_axis' in name_lower or 'm_axis' in name_lower:
            self.interface_type = InterfaceType.AXI_STREAM
        elif 's_axi' in name_lower or 'm_axi' in name_lower:
            if 'lite' in name_lower:
                self.interface_type = InterfaceType.AXI_LITE
            else:
                self.interface_type = InterfaceType.AXI_FULL


@dataclass
class RTLInterface:
    """Represents a complete hardware interface."""
    
    name: str
    interface_type: InterfaceType
    signals: List[RTLSignal] = field(default_factory=list)
    
    # Interface properties
    direction: SignalDirection = SignalDirection.INPUT
    data_width: Optional[int] = None
    is_required: bool = True
    
    # AXI-specific properties
    axi_protocol: Optional[str] = None  # "AXI4", "AXI4-Lite", "AXI4-Stream"
    axi_signals: Dict[str, str] = field(default_factory=dict)  # signal_type -> signal_name
    
    # Documentation
    description: str = ""
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def add_signal(self, signal: RTLSignal) -> None:
        """Add a signal to this interface."""
        self.signals.append(signal)
        
        # Update interface properties based on signals
        if signal.width and (not self.data_width or signal.width > self.data_width):
            self.data_width = signal.width
    
    def get_signals_by_direction(self, direction: SignalDirection) -> List[RTLSignal]:
        """Get signals with specific direction."""
        return [s for s in self.signals if s.direction == direction]
    
    def get_control_signals(self) -> List[RTLSignal]:
        """Get control signals in this interface."""
        return [s for s in self.signals if s.interface_type == InterfaceType.CONTROL]
    
    def get_data_signals(self) -> List[RTLSignal]:
        """Get data signals in this interface."""
        return [s for s in self.signals if s.interface_type == InterfaceType.DATA]


@dataclass
class RTLModule:
    """Represents an RTL module."""
    
    name: str
    file_path: Path
    interfaces: List[RTLInterface] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Module metadata
    is_top_level: bool = False
    description: str = ""
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    
    # Source information
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    
    # Analysis results
    complexity_score: float = 0.0
    estimated_resources: Dict[str, int] = field(default_factory=dict)
    
    def add_interface(self, interface: RTLInterface) -> None:
        """Add an interface to this module."""
        self.interfaces.append(interface)
    
    def get_interfaces_by_type(self, interface_type: InterfaceType) -> List[RTLInterface]:
        """Get interfaces of a specific type."""
        return [i for i in self.interfaces if i.interface_type == interface_type]
    
    def get_all_signals(self) -> List[RTLSignal]:
        """Get all signals from all interfaces."""
        signals = []
        for interface in self.interfaces:
            signals.extend(interface.signals)
        return signals


@dataclass
class ParsedRTLData:
    """Results of RTL parsing and analysis."""
    
    # Source files
    rtl_files: List[Path] = field(default_factory=list)
    main_file: Optional[Path] = None
    
    # Parsed modules
    modules: List[RTLModule] = field(default_factory=list)
    top_module: Optional[RTLModule] = None
    
    # Global elements
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    includes: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Parsing metadata
    parse_time: float = 0.0
    parser_version: str = "1.0.0"
    parsed_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    # Issues and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_module(self, module: RTLModule) -> None:
        """Add a module to the parsed data."""
        self.modules.append(module)
        
        # Auto-detect top module if not set
        if module.is_top_level and not self.top_module:
            self.top_module = module
    
    def get_module_by_name(self, name: str) -> Optional[RTLModule]:
        """Get module by name."""
        for module in self.modules:
            if module.name == name:
                return module
        return None
    
    def get_all_interfaces(self) -> List[RTLInterface]:
        """Get all interfaces from all modules."""
        interfaces = []
        for module in self.modules:
            interfaces.extend(module.interfaces)
        return interfaces
    
    def get_all_signals(self) -> List[RTLSignal]:
        """Get all signals from all modules."""
        signals = []
        for module in self.modules:
            signals.extend(module.get_all_signals())
        return signals
    
    def validate(self) -> bool:
        """Validate parsed RTL data."""
        self.errors.clear()
        self.warnings.clear()
        
        # Check for required elements
        if not self.modules:
            self.errors.append("No modules found in RTL files")
            return False
        
        if not self.top_module:
            self.warnings.append("No top-level module identified")
        
        # Validate modules
        for module in self.modules:
            if not module.interfaces:
                self.warnings.append(f"Module '{module.name}' has no interfaces")
        
        # Check for required control signals in top module
        if self.top_module:
            all_signals = self.top_module.get_all_signals()
            has_clock = any(s.interface_type == InterfaceType.CLOCK for s in all_signals)
            has_reset = any(s.interface_type == InterfaceType.RESET for s in all_signals)
            
            if not has_clock:
                self.warnings.append("No clock signal found in top module")
            if not has_reset:
                self.warnings.append("No reset signal found in top module")
        
        return len(self.errors) == 0


@dataclass
class PragmaInfo:
    """Information about a parsed pragma."""
    
    type: str
    content: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Source information
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    raw_text: str = ""
    
    # Processing status
    is_processed: bool = False
    processing_errors: List[str] = field(default_factory=list)


@dataclass
class AnalyzedInterfaces:
    """Results of interface analysis."""
    
    # Interface analysis
    detected_interfaces: List[RTLInterface] = field(default_factory=list)
    interface_groups: Dict[str, List[RTLInterface]] = field(default_factory=dict)
    
    # Protocol analysis
    axi_interfaces: List[RTLInterface] = field(default_factory=list)
    streaming_interfaces: List[RTLInterface] = field(default_factory=list)
    control_interfaces: List[RTLInterface] = field(default_factory=list)
    
    # Connectivity analysis
    interface_connections: Dict[str, List[str]] = field(default_factory=dict)
    data_flow_graph: Dict[str, List[str]] = field(default_factory=dict)
    
    # Analysis metadata
    analysis_time: float = 0.0
    analyzer_version: str = "1.0.0"
    analyzed_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    # Issues
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_interface(self, interface: RTLInterface) -> None:
        """Add an analyzed interface."""
        self.detected_interfaces.append(interface)
        
        # Categorize interface
        if interface.interface_type == InterfaceType.AXI_STREAM:
            self.axi_interfaces.append(interface)
            self.streaming_interfaces.append(interface)
        elif interface.interface_type in [InterfaceType.AXI_LITE, InterfaceType.AXI_FULL]:
            self.axi_interfaces.append(interface)
        elif interface.interface_type == InterfaceType.CONTROL:
            self.control_interfaces.append(interface)
    
    def get_interfaces_by_type(self, interface_type: InterfaceType) -> List[RTLInterface]:
        """Get interfaces of a specific type."""
        return [i for i in self.detected_interfaces if i.interface_type == interface_type]


@dataclass
class PipelineInputs:
    """Input data for the HWKG pipeline."""
    
    # Primary inputs
    rtl_files: List[Path] = field(default_factory=list)
    main_rtl_file: Optional[Path] = None
    
    # Configuration
    module_name: str = ""
    top_module_name: str = ""
    output_directory: Path = field(default_factory=lambda: Path("./generated"))
    
    # Generation options
    generator_type: str = "hw_custom_op"
    generate_testbench: bool = False
    generate_documentation: bool = True
    
    # Analysis options
    analyze_interfaces: bool = True
    analyze_dependencies: bool = True
    analyze_timing: bool = False
    
    # Template options
    custom_templates: Dict[str, Path] = field(default_factory=dict)
    template_overrides: Dict[str, str] = field(default_factory=dict)
    
    # FINN-specific options
    finn_datatype: str = "float32"
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    
    # Advanced options
    optimization_level: str = "2"
    target_device: str = ""
    clock_frequency: float = 100.0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    user: str = ""
    description: str = ""
    
    def validate(self) -> None:
        """Validate pipeline inputs."""
        errors = []
        
        # Check required fields
        if not self.rtl_files:
            errors.append("No RTL files specified")
        
        if not self.module_name:
            errors.append("Module name is required")
        
        # Validate file paths
        for rtl_file in self.rtl_files:
            if not rtl_file.exists():
                errors.append(f"RTL file not found: {rtl_file}")
        
        if self.main_rtl_file and not self.main_rtl_file.exists():
            errors.append(f"Main RTL file not found: {self.main_rtl_file}")
        
        # Validate custom templates
        for name, template_path in self.custom_templates.items():
            if not template_path.exists():
                errors.append(f"Custom template '{name}' not found: {template_path}")
        
        if errors:
            raise ValidationError(
                f"Pipeline input validation failed: {'; '.join(errors)}",
                validation_type="pipeline_inputs",
                suggestion="Check file paths and required parameters"
            )


@dataclass
class PipelineResults:
    """Results of the complete HWKG pipeline execution."""
    
    # Execution metadata
    pipeline_id: str = ""
    status: str = "unknown"
    started_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    completed_at: str = ""
    total_time: float = 0.0
    
    # Input summary
    inputs: Optional[PipelineInputs] = None
    
    # Stage results
    rtl_data: Optional[ParsedRTLData] = None
    interface_analysis: Optional[AnalyzedInterfaces] = None
    pragmas: List[PragmaInfo] = field(default_factory=list)
    
    # Generation results
    generated_files: List[Path] = field(default_factory=list)
    generation_result: Optional[Any] = None  # GenerationResult from generator_base
    
    # Performance metrics
    stage_times: Dict[PipelineStage, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    lines_generated: int = 0
    files_generated: int = 0
    templates_used: int = 0
    
    # Issues tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stage_errors: Dict[PipelineStage, List[str]] = field(default_factory=dict)
    
    # Output paths
    output_directory: Optional[Path] = None
    log_file: Optional[Path] = None
    
    def mark_stage_complete(self, stage: PipelineStage, duration: float) -> None:
        """Mark a pipeline stage as complete."""
        self.stage_times[stage] = duration
    
    def add_error(self, error: str, stage: Optional[PipelineStage] = None) -> None:
        """Add an error to the results."""
        self.errors.append(error)
        
        if stage:
            if stage not in self.stage_errors:
                self.stage_errors[stage] = []
            self.stage_errors[stage].append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the results."""
        self.warnings.append(warning)
    
    def finalize(self) -> None:
        """Finalize the pipeline results."""
        self.completed_at = datetime.datetime.now().isoformat()
        
        # Calculate total time
        if self.stage_times:
            self.total_time = sum(self.stage_times.values())
        
        # Update file counts
        self.files_generated = len(self.generated_files)
        
        # Determine final status
        if self.errors:
            self.status = "failed"
        elif self.warnings:
            self.status = "completed_with_warnings"
        else:
            self.status = "success"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline results."""
        return {
            'pipeline_id': self.pipeline_id,
            'status': self.status,
            'total_time': self.total_time,
            'files_generated': self.files_generated,
            'lines_generated': self.lines_generated,
            'templates_used': self.templates_used,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'stages_completed': len(self.stage_times),
            'output_directory': str(self.output_directory) if self.output_directory else None
        }
    
    def has_errors(self) -> bool:
        """Check if pipeline has any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if pipeline has any warnings."""
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        data = asdict(self)
        
        # Convert Path objects to strings
        if self.output_directory:
            data['output_directory'] = str(self.output_directory)
        if self.log_file:
            data['log_file'] = str(self.log_file)
        
        data['generated_files'] = [str(f) for f in self.generated_files]
        
        # Convert enums to strings
        stage_times_str = {}
        for stage, time_val in self.stage_times.items():
            stage_times_str[stage.value] = time_val
        data['stage_times'] = stage_times_str
        
        stage_errors_str = {}
        for stage, errors in self.stage_errors.items():
            stage_errors_str[stage.value] = errors
        data['stage_errors'] = stage_errors_str
        
        return data


def create_pipeline_inputs(rtl_files: List[Union[str, Path]], 
                          module_name: str, **kwargs) -> PipelineInputs:
    """Factory function to create pipeline inputs."""
    # Convert string paths to Path objects
    rtl_paths = [Path(f) if isinstance(f, str) else f for f in rtl_files]
    
    return PipelineInputs(
        rtl_files=rtl_paths,
        module_name=module_name,
        **kwargs
    )


def create_pipeline_results(pipeline_id: str = "", inputs: Optional[PipelineInputs] = None) -> PipelineResults:
    """Factory function to create pipeline results."""
    if not pipeline_id:
        import uuid
        pipeline_id = str(uuid.uuid4())[:8]
    
    return PipelineResults(
        pipeline_id=pipeline_id,
        inputs=inputs
    )