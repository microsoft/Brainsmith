"""
FINN Infrastructure Types

Type definitions for the FINN infrastructure layer.
Provides data structures for FINN configuration, results, and 4-hooks preparation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class FINNDevice(Enum):
    """Supported FPGA devices for FINN builds."""
    PYNQ_Z1 = "pynq-z1"
    PYNQ_Z2 = "pynq-z2"
    ULTRA96 = "ultra96"
    ZCU104 = "zcu104"
    ALVEO_U250 = "alveo-u250"
    ALVEO_U200 = "alveo-u200"


class FINNOptimization(Enum):
    """FINN optimization levels."""
    NONE = "none"
    BASIC = "basic"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class FINNConfig:
    """Configuration for FINN builds."""
    device: FINNDevice = FINNDevice.PYNQ_Z1
    optimization_level: FINNOptimization = FINNOptimization.BASIC
    enable_vivado_synth: bool = True
    enable_rtlsim: bool = False
    target_clock_ns: float = 10.0
    enable_debug: bool = False
    additional_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_config is None:
            self.additional_config = {}
    
    def to_core_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for core FINN interface."""
        return {
            "device": self.device.value,
            "optimization_level": self.optimization_level.value,
            "enable_vivado_synth": self.enable_vivado_synth,
            "enable_rtlsim": self.enable_rtlsim,
            "target_clock_ns": self.target_clock_ns,
            "enable_debug": self.enable_debug,
            **self.additional_config
        }


@dataclass
class FINNBuildMetrics:
    """Metrics from FINN build process."""
    resource_utilization: Dict[str, float] = None
    timing_metrics: Dict[str, float] = None
    power_estimate: Optional[float] = None
    throughput: Optional[float] = None
    latency: Optional[float] = None
    accuracy: Optional[float] = None
    build_time: Optional[float] = None
    
    def __post_init__(self):
        if self.resource_utilization is None:
            self.resource_utilization = {}
        if self.timing_metrics is None:
            self.timing_metrics = {}


@dataclass
class FINNResult:
    """Result of FINN build operation."""
    success: bool
    model_path: str
    output_dir: str
    build_metrics: Optional[FINNBuildMetrics] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    build_artifacts: Dict[str, str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.build_artifacts is None:
            self.build_artifacts = {}
    
    @classmethod
    def from_core_result(cls, core_result: Dict[str, Any]) -> 'FINNResult':
        """Create FINNResult from core FINN interface result."""
        metrics = None
        if core_result.get("metrics"):
            metrics = FINNBuildMetrics(
                resource_utilization=core_result["metrics"].get("resources", {}),
                timing_metrics=core_result["metrics"].get("timing", {}),
                power_estimate=core_result["metrics"].get("power"),
                throughput=core_result["metrics"].get("throughput"),
                latency=core_result["metrics"].get("latency"),
                accuracy=core_result["metrics"].get("accuracy"),
                build_time=core_result["metrics"].get("build_time")
            )
        
        return cls(
            success=core_result.get("success", False),
            model_path=core_result.get("model_path", ""),
            output_dir=core_result.get("output_dir", ""),
            build_metrics=metrics,
            error_message=core_result.get("error"),
            warnings=core_result.get("warnings", []),
            build_artifacts=core_result.get("artifacts", {})
        )


@dataclass
class FINNHooksConfig:
    """Configuration for future 4-hooks FINN interface."""
    
    def prepare_config(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare design point configuration for 4-hooks interface.
        
        This method provides a foundation for future 4-hooks interface
        integration while maintaining current functionality.
        
        Args:
            design_point: Design point parameters
            
        Returns:
            Configuration dictionary prepared for 4-hooks interface
        """
        # Extract relevant parameters for 4-hooks preparation
        hooks_config = {
            "dataflow_config": self._extract_dataflow_config(design_point),
            "synthesis_config": self._extract_synthesis_config(design_point),
            "verification_config": self._extract_verification_config(design_point),
            "deployment_config": self._extract_deployment_config(design_point)
        }
        
        return hooks_config
    
    def _extract_dataflow_config(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dataflow-related configuration."""
        return {
            "folding_factors": design_point.get("folding_factors", {}),
            "mem_mode": design_point.get("mem_mode", "internal_decoupled"),
            "resType": design_point.get("resType", "lut"),
            "mvau_wwidth_max": design_point.get("mvau_wwidth_max", 36)
        }
    
    def _extract_synthesis_config(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """Extract synthesis-related configuration."""
        return {
            "fpga_part": design_point.get("fpga_part", "xc7z020clg400-1"),
            "target_clock_ns": design_point.get("target_clock_ns", 10.0),
            "enable_zynq_ps": design_point.get("enable_zynq_ps", True)
        }
    
    def _extract_verification_config(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """Extract verification-related configuration."""
        return {
            "enable_rtlsim": design_point.get("enable_rtlsim", False),
            "enable_cppsim": design_point.get("enable_cppsim", True),
            "verify_steps": design_point.get("verify_steps", ["cppsim"])
        }
    
    def _extract_deployment_config(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """Extract deployment-related configuration."""
        return {
            "platform": design_point.get("platform", "zynq-iodma"),
            "enable_debug": design_point.get("enable_debug", False),
            "driver_mode": design_point.get("driver_mode", "python")
        }


# Type aliases for better readability
FINNConfigDict = Dict[str, Any]
FINNMetricsDict = Dict[str, Any]
FINNResultDict = Dict[str, Any]