"""
Enhanced configuration system for Brainsmith platform.

This module provides configuration classes that support both simple compilation
and advanced design space exploration workflows.
"""

import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .design_space import DesignPoint


@dataclass
class CompilerConfig:
    """Enhanced configuration with DSE support and extensibility."""
    
    # Core compilation parameters
    blueprint: str = ""
    output_dir: str = "./build"
    
    # Model parameters
    model_path: str = ""
    
    # Metric collection settings
    collect_comprehensive_metrics: bool = True
    export_research_data: bool = False
    save_intermediate_results: bool = False
    
    # Design space exploration settings
    dse_enabled: bool = False
    parameter_sweep: Optional[Dict[str, List[Any]]] = None
    single_design_point: Optional[DesignPoint] = None
    
    # FINN integration settings
    finn_hooks_override: Optional[Dict[str, Any]] = None
    
    # Traditional FINN parameters (for backward compatibility)
    target_fps: int = 3000
    synth_clk_period_ns: float = 3.33
    board: str = "V80"
    folding_config_file: Optional[str] = None
    auto_fifo_depths: bool = True
    verification_atol: float = 1e-5
    
    # Build settings
    enable_verification: bool = True
    enable_rtlsim: bool = True
    parallel_builds: int = 1
    
    # Debug and logging
    verbose: bool = False
    debug: bool = False
    log_level: str = "INFO"
    
    # Custom settings (extensible)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_design_point(self) -> DesignPoint:
        """Convert configuration to design point."""
        point = DesignPoint()
        
        # Map configuration parameters to design point
        point.set_parameter('target_fps', self.target_fps)
        point.set_parameter('clk_period_ns', self.synth_clk_period_ns)
        point.set_parameter('board', self.board)
        
        # Add FINN hooks if provided
        if self.finn_hooks_override:
            point.finn_hooks = self.finn_hooks_override.copy()
        
        # Add custom settings
        for key, value in self.custom_settings.items():
            point.set_parameter(key, value)
        
        # Add single design point overrides
        if self.single_design_point:
            point.parameters.update(self.single_design_point.parameters)
            point.finn_hooks.update(self.single_design_point.finn_hooks)
        
        return point
    
    def update_from_design_point(self, design_point: DesignPoint):
        """Update configuration from design point."""
        # Update traditional parameters
        if 'target_fps' in design_point.parameters:
            self.target_fps = design_point.parameters['target_fps']
        
        if 'clk_period_ns' in design_point.parameters:
            self.synth_clk_period_ns = design_point.parameters['clk_period_ns']
        
        if 'board' in design_point.parameters:
            self.board = design_point.parameters['board']
        elif 'platform' in design_point.parameters:
            self.board = design_point.parameters['platform']
        
        # Update FINN hooks
        if design_point.finn_hooks:
            if self.finn_hooks_override is None:
                self.finn_hooks_override = {}
            self.finn_hooks_override.update(design_point.finn_hooks)
    
    def get_output_subdir(self, suffix: str = "") -> str:
        """Get output subdirectory with optional suffix."""
        if suffix:
            return os.path.join(self.output_dir, suffix)
        return self.output_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        
        # Handle special objects
        if self.single_design_point:
            data['single_design_point'] = self.single_design_point.to_dict()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompilerConfig':
        """Create from dictionary."""
        # Handle special objects
        if 'single_design_point' in data and data['single_design_point']:
            data['single_design_point'] = DesignPoint.from_dict(data['single_design_point'])
        
        return cls(**data)
    
    def save_to_file(self, filepath: str):
        """Save configuration to YAML file."""
        data = self.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'CompilerConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)


@dataclass
class ParameterSweepConfig:
    """Configuration for parameter sweep operations."""
    
    # Parameters to sweep
    parameters: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Output organization
    base_output_dir: str = "./parameter_sweep"
    create_subdirs: bool = True
    subdir_pattern: str = "config_{index:03d}"
    
    # Execution settings
    parallel_builds: int = 1
    continue_on_error: bool = True
    save_partial_results: bool = True
    
    # Analysis settings
    generate_comparison_report: bool = True
    export_csv: bool = True
    find_pareto_frontier: bool = False
    
    def get_total_configurations(self) -> int:
        """Calculate total number of configurations."""
        if not self.parameters:
            return 0
        
        total = 1
        for values in self.parameters.values():
            total *= len(values)
        return total
    
    def estimate_total_time(self, avg_build_time: float = 180.0) -> float:
        """Estimate total execution time in seconds."""
        total_configs = self.get_total_configurations()
        
        if self.parallel_builds > 1:
            return (total_configs * avg_build_time) / self.parallel_builds
        else:
            return total_configs * avg_build_time
    
    def get_output_dir(self, config_index: int) -> str:
        """Get output directory for specific configuration."""
        if self.create_subdirs:
            subdir = self.subdir_pattern.format(index=config_index)
            return os.path.join(self.base_output_dir, subdir)
        else:
            return self.base_output_dir


@dataclass
class DSEConfig:
    """Configuration for design space exploration."""
    
    # DSE strategy
    strategy: str = "random"  # "random", "grid", "lhs", "external"
    max_evaluations: int = 50
    
    # Sampling settings
    random_seed: Optional[int] = None
    sampling_strategy: str = "random"  # "random", "latin_hypercube", "sobol"
    
    # External tool integration
    external_tool_interface: Optional[str] = None
    external_tool_config: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization objectives
    objectives: List[str] = field(default_factory=lambda: ["throughput_ops_sec"])
    objective_directions: List[str] = field(default_factory=lambda: ["maximize"])
    
    # Constraint handling
    enforce_hard_constraints: bool = True
    constraint_violation_penalty: float = 1000.0
    
    # Output and analysis
    base_output_dir: str = "./dse_exploration"
    save_all_results: bool = True
    generate_analysis_report: bool = True
    
    # Convergence criteria
    early_stopping: bool = False
    convergence_patience: int = 10
    min_improvement: float = 0.01
    
    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get summary of exploration configuration."""
        return {
            'strategy': self.strategy,
            'max_evaluations': self.max_evaluations,
            'objectives': self.objectives,
            'random_seed': self.random_seed,
            'early_stopping': self.early_stopping
        }


def load_config_from_file(filepath: str) -> CompilerConfig:
    """Load configuration from YAML or JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    file_ext = Path(filepath).suffix.lower()
    
    if file_ext in ['.yml', '.yaml']:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
    elif file_ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_ext}")
    
    return CompilerConfig.from_dict(data)


def create_parameter_sweep_config(parameters: Dict[str, List[Any]], 
                                 output_dir: str = "./parameter_sweep",
                                 **kwargs) -> ParameterSweepConfig:
    """Create parameter sweep configuration with sensible defaults."""
    return ParameterSweepConfig(
        parameters=parameters,
        base_output_dir=output_dir,
        **kwargs
    )


def create_dse_config(strategy: str = "random",
                     max_evaluations: int = 50,
                     objectives: List[str] = None,
                     output_dir: str = "./dse_exploration",
                     **kwargs) -> DSEConfig:
    """Create DSE configuration with sensible defaults."""
    if objectives is None:
        objectives = ["throughput_ops_sec"]
    
    return DSEConfig(
        strategy=strategy,
        max_evaluations=max_evaluations,
        objectives=objectives,
        base_output_dir=output_dir,
        **kwargs
    )


# Legacy compatibility wrapper
@dataclass
class BrainsmithConfig:
    """Legacy configuration class for backward compatibility."""
    
    def __init__(self):
        """Initialize with default values matching legacy interface."""
        self.target_fps = 3000
        self.synth_clk_period_ns = 3.33
        self.board = "V80"
        self.output_dir = "./build"
        self.folding_config_file = None
        self.auto_fifo_depths = True
        self.verification_atol = 1e-5
    
    def to_compiler_config(self, blueprint: str = "", model_path: str = "") -> CompilerConfig:
        """Convert to new CompilerConfig format."""
        return CompilerConfig(
            blueprint=blueprint,
            model_path=model_path,
            output_dir=self.output_dir,
            target_fps=self.target_fps,
            synth_clk_period_ns=self.synth_clk_period_ns,
            board=self.board,
            folding_config_file=self.folding_config_file,
            auto_fifo_depths=self.auto_fifo_depths,
            verification_atol=self.verification_atol
        )


def validate_config(config: CompilerConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check required fields
    if not config.blueprint:
        issues.append("Blueprint name is required")
    
    if not config.output_dir:
        issues.append("Output directory is required")
    
    # Check parameter sweep configuration
    if config.dse_enabled and config.parameter_sweep:
        if not config.parameter_sweep:
            issues.append("Parameter sweep enabled but no parameters specified")
        
        # Check parameter values are lists
        for param_name, values in config.parameter_sweep.items():
            if not isinstance(values, list):
                issues.append(f"Parameter '{param_name}' values must be a list")
            if len(values) == 0:
                issues.append(f"Parameter '{param_name}' has no values")
    
    # Check FINN parameters
    if config.target_fps <= 0:
        issues.append("Target FPS must be positive")
    
    if config.synth_clk_period_ns <= 0:
        issues.append("Clock period must be positive")
    
    # Check board name
    valid_boards = ["V80", "ZCU104", "U250", "VCK190"]  # Common FINN boards
    if config.board not in valid_boards:
        issues.append(f"Board '{config.board}' may not be supported. Valid options: {valid_boards}")
    
    return issues


def get_default_config() -> CompilerConfig:
    """Get default configuration with sensible values."""
    return CompilerConfig(
        blueprint="",
        output_dir="./build",
        target_fps=3000,
        synth_clk_period_ns=3.33,
        board="V80",
        collect_comprehensive_metrics=True,
        enable_verification=True,
        enable_rtlsim=True,
        verbose=False
    )