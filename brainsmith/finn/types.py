"""
Core data types and structures for FINN Integration Engine.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    AREA = "area"
    POWER = "power"
    BALANCED = "balanced"

class BuildStatus(Enum):
    """FINN build status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ModelOpsConfig:
    """Model operations configuration for FINN"""
    supported_ops: List[str] = field(default_factory=list)
    custom_ops: Dict[str, Any] = field(default_factory=dict)
    frontend_cleanup: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'supported_ops': self.supported_ops,
            'custom_ops': self.custom_ops,
            'frontend_cleanup': self.frontend_cleanup,
            'preprocessing_steps': self.preprocessing_steps,
            'validation_rules': self.validation_rules
        }
    
    def copy(self) -> 'ModelOpsConfig':
        return ModelOpsConfig(
            supported_ops=self.supported_ops.copy(),
            custom_ops=self.custom_ops.copy(),
            frontend_cleanup=self.frontend_cleanup.copy(),
            preprocessing_steps=self.preprocessing_steps.copy(),
            validation_rules=self.validation_rules.copy()
        )

@dataclass
class ModelTransformsConfig:
    """Model transforms configuration for FINN"""
    transforms_sequence: List[str] = field(default_factory=list)
    optimization_level: str = "standard"
    target_platform: str = "zynq"
    quantization_config: Dict[str, Any] = field(default_factory=dict)
    graph_optimizations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'transforms_sequence': self.transforms_sequence,
            'optimization_level': self.optimization_level,
            'target_platform': self.target_platform,
            'quantization_config': self.quantization_config,
            'graph_optimizations': self.graph_optimizations
        }
    
    def copy(self) -> 'ModelTransformsConfig':
        return ModelTransformsConfig(
            transforms_sequence=self.transforms_sequence.copy(),
            optimization_level=self.optimization_level,
            target_platform=self.target_platform,
            quantization_config=self.quantization_config.copy(),
            graph_optimizations=self.graph_optimizations.copy()
        )

@dataclass
class HwKernelsConfig:
    """Hardware kernels configuration for FINN"""
    kernel_selection_plan: Dict[str, str] = field(default_factory=dict)
    custom_kernels: Dict[str, Any] = field(default_factory=dict)
    folding_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resource_sharing: Dict[str, Any] = field(default_factory=dict)
    memory_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'kernel_selection_plan': self.kernel_selection_plan,
            'custom_kernels': self.custom_kernels,
            'folding_config': self.folding_config,
            'resource_sharing': self.resource_sharing,
            'memory_config': self.memory_config
        }
    
    def copy(self) -> 'HwKernelsConfig':
        return HwKernelsConfig(
            kernel_selection_plan=self.kernel_selection_plan.copy(),
            custom_kernels=self.custom_kernels.copy(),
            folding_config=self.folding_config.copy(),
            resource_sharing=self.resource_sharing.copy(),
            memory_config=self.memory_config.copy()
        )

@dataclass
class HwOptimizationConfig:
    """Hardware optimization configuration for FINN"""
    optimization_strategy: str = "balanced"
    performance_targets: Dict[str, float] = field(default_factory=dict)
    power_constraints: Dict[str, float] = field(default_factory=dict)
    timing_constraints: Dict[str, float] = field(default_factory=dict)
    resource_constraints: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimization_strategy': self.optimization_strategy,
            'performance_targets': self.performance_targets,
            'power_constraints': self.power_constraints,
            'timing_constraints': self.timing_constraints,
            'resource_constraints': self.resource_constraints
        }
    
    def copy(self) -> 'HwOptimizationConfig':
        return HwOptimizationConfig(
            optimization_strategy=self.optimization_strategy,
            performance_targets=self.performance_targets.copy(),
            power_constraints=self.power_constraints.copy(),
            timing_constraints=self.timing_constraints.copy(),
            resource_constraints=self.resource_constraints.copy()
        )

@dataclass
class FINNInterfaceConfig:
    """Complete FINN interface configuration across all four categories"""
    model_ops: ModelOpsConfig = field(default_factory=ModelOpsConfig)
    model_transforms: ModelTransformsConfig = field(default_factory=ModelTransformsConfig)
    hw_kernels: HwKernelsConfig = field(default_factory=HwKernelsConfig)
    hw_optimization: HwOptimizationConfig = field(default_factory=HwOptimizationConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for FINN consumption"""
        return {
            'model_ops': self.model_ops.to_dict(),
            'model_transforms': self.model_transforms.to_dict(),
            'hw_kernels': self.hw_kernels.to_dict(),
            'hw_optimization': self.hw_optimization.to_dict(),
            'metadata': self.metadata
        }
    
    def copy(self) -> 'FINNInterfaceConfig':
        """Create deep copy of configuration"""
        return FINNInterfaceConfig(
            model_ops=self.model_ops.copy(),
            model_transforms=self.model_transforms.copy(),
            hw_kernels=self.hw_kernels.copy(),
            hw_optimization=self.hw_optimization.copy(),
            metadata=self.metadata.copy()
        )

@dataclass
class PerformanceMetrics:
    """Performance metrics from FINN build"""
    throughput: float = 0.0
    latency: float = 0.0
    power: float = 0.0
    efficiency: float = 0.0
    clock_frequency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'throughput': self.throughput,
            'latency': self.latency,
            'power': self.power,
            'efficiency': self.efficiency,
            'clock_frequency': self.clock_frequency
        }

@dataclass
class ResourceAnalysis:
    """Resource utilization analysis"""
    lut_usage: Dict[str, float] = field(default_factory=dict)
    dsp_usage: Dict[str, float] = field(default_factory=dict)
    bram_usage: Dict[str, float] = field(default_factory=dict)
    ff_usage: Dict[str, float] = field(default_factory=dict)
    global_utilization: Dict[str, float] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lut_usage': self.lut_usage,
            'dsp_usage': self.dsp_usage,
            'bram_usage': self.bram_usage,
            'ff_usage': self.ff_usage,
            'global_utilization': self.global_utilization,
            'bottlenecks': self.bottlenecks,
            'optimization_suggestions': self.optimization_suggestions
        }

@dataclass
class TimingAnalysis:
    """Timing analysis results"""
    critical_paths: List[Dict[str, Any]] = field(default_factory=list)
    timing_margins: Dict[str, float] = field(default_factory=dict)
    timing_bottlenecks: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'critical_paths': self.critical_paths,
            'timing_margins': self.timing_margins,
            'timing_bottlenecks': self.timing_bottlenecks,
            'optimization_suggestions': self.optimization_suggestions
        }

@dataclass
class BuildEnvironment:
    """FINN build environment configuration"""
    finn_root: str = ""
    vivado_path: str = ""
    target_device: str = "xc7z020clg400-1"
    clock_period: float = 10.0
    build_dir: str = ""
    temp_dir: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'finn_root': self.finn_root,
            'vivado_path': self.vivado_path,
            'target_device': self.target_device,
            'clock_period': self.clock_period,
            'build_dir': self.build_dir,
            'temp_dir': self.temp_dir
        }

@dataclass
class FINNBuildResult:
    """Result from FINN build execution"""
    status: BuildStatus = BuildStatus.PENDING
    success: bool = False
    build_time: float = 0.0
    output_dir: str = ""
    log_files: List[str] = field(default_factory=list)
    synthesis_reports: List[str] = field(default_factory=list)
    timing_reports: List[str] = field(default_factory=list)
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value if hasattr(self.status, 'value') else str(self.status),
            'success': self.success,
            'build_time': self.build_time,
            'output_dir': self.output_dir,
            'log_files': self.log_files,
            'synthesis_reports': self.synthesis_reports,
            'timing_reports': self.timing_reports,
            'error_message': self.error_message,
            'warnings': self.warnings
        }

@dataclass
class EnhancedFINNResult:
    """Enhanced FINN build result with comprehensive analysis"""
    original_result: FINNBuildResult
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    resource_analysis: ResourceAnalysis = field(default_factory=ResourceAnalysis)
    timing_analysis: TimingAnalysis = field(default_factory=TimingAnalysis)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    optimization_opportunities: List[str] = field(default_factory=list)
    enhanced_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        return self.original_result.success
    
    @property
    def build_time(self) -> float:
        return self.original_result.build_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_result': self.original_result.to_dict(),
            'performance_metrics': self.performance_metrics.to_dict(),
            'resource_analysis': self.resource_analysis.to_dict(),
            'timing_analysis': self.timing_analysis.to_dict(),
            'quality_metrics': self.quality_metrics,
            'optimization_opportunities': self.optimization_opportunities,
            'enhanced_timestamp': self.enhanced_timestamp.isoformat()
        }