############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Kernel model v2 with separate input/output interfaces"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any, Set
import math
from .base import BaseModel, ParameterBinding
from .input_interface import InputInterface
from .output_interface import OutputInterface
from .types import prod, SDIMParameterInfo, Shape
from .relationships import RelationType

@dataclass
class KernelModel(BaseModel):
    """Runtime kernel model with separate input/output interfaces
    
    Key features:
    - Separate input_models and output_models lists
    - SDIM configuration only for inputs
    - Output rates computed from kernel behavior
    """
    
    # Separate interface storage
    input_models: List[InputInterface] = field(default_factory=list)
    output_models: List[OutputInterface] = field(default_factory=list)
    parameter_binding: ParameterBinding = field(default_factory=lambda: ParameterBinding({}))
    
    # Timing characteristics
    latency_cycles: Tuple[int, int] = (1, 1)
    calculation_ii: Optional[int] = None
    execution_ii: Optional[int] = None
    
    # Pipeline characteristics
    priming_cycles: int = 0
    flush_cycles: int = 0
    pipeline_depth: int = 1
    
    # Resource usage
    resources: Dict[str, float] = field(default_factory=dict)
    power_watts: float = 0.0
    
    # Performance characteristics
    clock_freq_mhz: float = 100.0
    actual_efficiency: float = 1.0
    
    def __init__(self,
                 input_models: List[InputInterface] = None,
                 output_models: List[OutputInterface] = None,
                 definition: Optional['KernelDefinition'] = None,
                 parameter_binding: Optional[ParameterBinding] = None,
                 **kwargs):
        """Initialize kernel model
        
        Args:
            input_models: List of input interfaces with concrete datatypes
            output_models: List of output interfaces with concrete datatypes
            definition: Parent kernel definition (optional)
            parameter_binding: Parameter bindings (optional)
            **kwargs: Additional properties
        """
        super().__init__(definition)
        
        self.input_models = input_models or []
        self.output_models = output_models or []
        self.parameter_binding = parameter_binding or ParameterBinding({})
        
        # Validate that all interfaces have concrete datatypes
        for inp in self.input_models:
            if not hasattr(inp, 'datatype') or inp.datatype is None:
                raise ValueError(f"Input interface '{inp.definition.name}' missing concrete datatype")
        
        for out in self.output_models:
            if not hasattr(out, 'datatype') or out.datatype is None:
                raise ValueError(f"Output interface '{out.definition.name}' missing concrete datatype")
        
        # Initialize other fields from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.__post_init__()
    
    def __post_init__(self):
        """Initialize model with optimized setup"""
        # Build name mappings for quick lookup
        self._input_map = {inp.definition.name if inp.definition else f"input_{i}": inp 
                          for i, inp in enumerate(self.input_models)}
        self._output_map = {out.definition.name if out.definition else f"output_{i}": out 
                           for i, out in enumerate(self.output_models)}
        
        # Cache for performance calculations
        self._cached_metrics = {}
        
        # Compute initial output rates
        self.compute_output_rates()
    
    @property
    def name(self) -> str:
        """Get kernel name from definition"""
        return self.definition.name if self.definition else "unnamed_kernel"
    
    def get_input_model(self, name: str) -> Optional[InputInterface]:
        """Get input model by name"""
        return self._input_map.get(name)
    
    def get_output_model(self, name: str) -> Optional[OutputInterface]:
        """Get output model by name"""
        return self._output_map.get(name)
    
    def get_sdim_parameters(self) -> Dict[str, SDIMParameterInfo]:
        """Get SDIM parameters that need configuration (inputs only)"""
        parameters = {}
        
        # Analyze relationships to find constraints between inputs
        hidden_interfaces = set()
        dimension_constraints = {}
        
        if self.definition:
            for rel in self.definition.relationships:
                # Only consider input-input relationships
                if (rel.source_interface in self._input_map and 
                    rel.target_interface in self._input_map):
                    
                    if rel.relation == RelationType.EQUAL:
                        # EQUAL hides entire target interface
                        hidden_interfaces.add(rel.target_interface)
                        
                    elif rel.relation == RelationType.DEPENDENT:
                        # DEPENDENT constrains specific dimension
                        if rel.target_interface not in dimension_constraints:
                            dimension_constraints[rel.target_interface] = {}
                        if rel.target_dim is not None:
                            dimension_constraints[rel.target_interface][rel.target_dim] = "dependent"
        
        # Build parameter info for inputs only
        for name, inp in self._input_map.items():
            if name in hidden_interfaces:
                continue
            
            n_dims = len(inp.block_dims[0])  # Number of dimensions in the first phase
            dim_constraints = dimension_constraints.get(name, {})
            free_dims = [i for i in range(n_dims) if i not in dim_constraints]
            
            if free_dims:  # Only expose if there are free dimensions
                parameters[name] = SDIMParameterInfo(
                    interface_name=name,
                    total_dimensions=n_dims,
                    free_dimensions=free_dims,
                    constrained_dimensions=dim_constraints,
                    block_dims=inp.block_dims[0]
                )
        
        return parameters
    
    def configure_sdim(self, config: Dict[str, Union[int, List[int], Dict[int, int]]]) -> None:
        """Configure SDIM for input interfaces only
        
        Args:
            config: Maps input names to SDIM specifications
        """
        # Validate all interfaces are inputs
        for intf_name in config:
            if intf_name not in self._input_map:
                if intf_name in self._output_map:
                    raise ValueError(
                        f"Cannot configure SDIM for output interface '{intf_name}'. "
                        "Only input interfaces support SDIM configuration."
                    )
                else:
                    raise ValueError(f"Interface '{intf_name}' not found")
        
        # Apply configurations
        configured_interfaces = set()
        
        for intf_name, sdim_spec in config.items():
            inp = self._input_map[intf_name]
            # Initialize SDIM to 1 for each dimension in the first phase
            current_sdim = list(inp.sdim) if inp._sdim else [1] * len(inp.block_dims[0])
            
            if isinstance(sdim_spec, int):
                # Uniform for all free dimensions
                params = self.get_sdim_parameters()
                if intf_name in params:
                    for dim in params[intf_name].free_dimensions:
                        # Only apply if it doesn't exceed block dimension
                        # Access the dimension within the first phase
                        if sdim_spec <= inp.block_dims[0][dim]:
                            current_sdim[dim] = sdim_spec
                else:
                    # If no parameter info, apply to dims where it fits
                    # Iterate over dimensions in the first phase
                    for i, bd in enumerate(inp.block_dims[0]):
                        if sdim_spec <= bd:
                            current_sdim[i] = sdim_spec
            
            elif isinstance(sdim_spec, dict):
                # Sparse specification
                for dim, value in sdim_spec.items():
                    if value is not None:
                        current_sdim[dim] = value
            
            else:
                # Full specification
                for i, value in enumerate(sdim_spec):
                    if value is not None:
                        current_sdim[i] = value
            
            inp.sdim = current_sdim
            configured_interfaces.add(intf_name)
        
        # Propagate through input-input relationships
        if self.definition:
            self._propagate_sdim_constraints(configured_interfaces)
        
        # Validate configuration
        self._validate_sdim_configuration()
        
        # Update output rates based on new input configuration
        self.compute_output_rates()
        
        # Clear caches
        self.clear_cache()
    
    def _propagate_sdim_constraints(self, configured: Set[str]) -> None:
        """Propagate SDIM through input-input relationships only"""
        if not self.definition:
            return
        
        changed = True
        iterations = 0
        max_iterations = len(self.input_models) * 2
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for rel in self.definition.relationships:
                # Only process input-input relationships
                if (rel.source_interface not in self._input_map or
                    rel.target_interface not in self._input_map):
                    continue
                
                if rel.target_interface in configured:
                    continue
                
                source = self._input_map[rel.source_interface]
                target = self._input_map[rel.target_interface]
                
                if source._sdim is None:
                    continue
                
                if rel.relation == RelationType.EQUAL:
                    # Check if dimension-specific or full equality
                    if rel.source_dim is not None and rel.target_dim is not None:
                        # Dimension-specific equality
                        target_sdim = list(target.sdim)
                        source_dim_value = source.sdim[rel.source_dim]
                        if target_sdim[rel.target_dim] != source_dim_value:
                            target_sdim[rel.target_dim] = source_dim_value
                            target.sdim = target_sdim
                            changed = True
                    else:
                        # Full SDIM equality (only if dimensions match)
                        if len(source.sdim) == len(target.sdim):
                            if target._sdim is None or target._sdim != source._sdim:
                                target.sdim = source.sdim
                                changed = True
                
                elif rel.relation == RelationType.DEPENDENT:
                    # Dimension-specific dependency
                    if rel.source_dim is not None and rel.target_dim is not None:
                        target_sdim = list(target.sdim)
                        
                        if rel.dependency_type == "scaled" and rel.factor:
                            new_value = int(source.sdim[rel.source_dim] * rel.factor)
                        else:
                            new_value = source.sdim[rel.source_dim]
                        
                        if target_sdim[rel.target_dim] != new_value:
                            target_sdim[rel.target_dim] = new_value
                            target.sdim = target_sdim
                            changed = True
    
    def _validate_sdim_configuration(self) -> None:
        """Validate SDIM configuration for input-input relationships"""
        if not self.definition:
            return
        
        errors = []
        
        for rel in self.definition.relationships:
            # Only validate input-input relationships
            if (rel.source_interface not in self._input_map or
                rel.target_interface not in self._input_map):
                continue
            
            source = self._input_map[rel.source_interface]
            target = self._input_map[rel.target_interface]
            
            try:
                if rel.relation == RelationType.EQUAL:
                    # Check if dimension-specific or full equality
                    if rel.source_dim is not None and rel.target_dim is not None:
                        # Dimension-specific equality
                        source_val = source.sdim[rel.source_dim]
                        target_val = target.sdim[rel.target_dim]
                        if source_val != target_val:
                            errors.append(
                                f"EQUAL constraint violated: {rel.source_interface}[{rel.source_dim}]="
                                f"{source_val} != {rel.target_interface}[{rel.target_dim}]={target_val}"
                            )
                    else:
                        # Full SDIM equality
                        if source.sdim != target.sdim:
                            errors.append(
                                f"EQUAL constraint violated: {rel.source_interface}.sdim="
                                f"{source.sdim} != {rel.target_interface}.sdim={target.sdim}"
                            )
                
                elif rel.relation == RelationType.DEPENDENT:
                    if rel.source_dim is not None and rel.target_dim is not None:
                        expected = source.sdim[rel.source_dim]
                        actual = target.sdim[rel.target_dim]
                        if actual != expected:
                            errors.append(
                                f"DEPENDENT constraint violated: {rel.target_interface}["
                                f"{rel.target_dim}]={actual} should equal {rel.source_interface}["
                                f"{rel.source_dim}]={expected}"
                            )
            except Exception as e:
                errors.append(f"Error validating {rel.describe()}: {e}")
        
        if errors:
            raise ValueError("SDIM configuration validation failed:\n" + "\n".join(errors))
    
    def compute_output_rates(self) -> None:
        """Compute output streaming rates based on input SDIM and kernel behavior
        
        This is a placeholder that should be overridden by specific kernel types.
        Default behavior: match the rate of the first input.
        """
        if not self.input_models or not self.output_models:
            return
        
        # Default: outputs stream at same rate as first input
        first_input_rate = self.input_models[0].streaming_bandwidth
        
        for output in self.output_models:
            output.set_streaming_rate(first_input_rate)
    
    def get_sdim_state(self) -> Dict[str, Shape]:
        """Get current SDIM values for all inputs"""
        state = {}
        for name, inp in self._input_map.items():
            state[name] = inp.sdim
        return state
    
    def initiation_interval(self) -> int:
        """Compute kernel initiation interval"""
        if "initiation_interval" not in self._cached_metrics:
            if self.calculation_ii is not None:
                ii = self.calculation_ii
            else:
                # Default: maximum II across inputs
                max_ii = 1
                for inp in self.input_models:
                    max_ii = max(max_ii, inp.initiation_interval)
                ii = max_ii
            
            self._cached_metrics["initiation_interval"] = ii
        
        return self._cached_metrics["initiation_interval"]
    
    def throughput_fps(self) -> float:
        """Compute kernel throughput in inferences per second"""
        if "throughput_fps" not in self._cached_metrics:
            cycles_per_inf = self.initiation_interval()
            
            # Apply efficiency factor
            effective_cycles = cycles_per_inf / self.actual_efficiency
            
            # Convert to inferences/second
            clock_hz = self.clock_freq_mhz * 1e6
            fps = clock_hz / effective_cycles
            
            self._cached_metrics["throughput_fps"] = fps
        
        return self._cached_metrics["throughput_fps"]
    
    def calculate_performance_metrics(self, frequency_mhz: float = 100.0) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {
            "kernel_name": self.name,
            "inputs": {},
            "outputs": {},
            "aggregate": {}
        }
        
        # Input metrics
        total_input_bandwidth_bits = 0
        max_input_ii = 0
        
        for name, inp in self._input_map.items():
            inp_metrics = inp.calculate_performance_metrics()
            metrics["inputs"][name] = inp_metrics
            total_input_bandwidth_bits += inp.bandwidth_bits
            max_input_ii = max(max_input_ii, inp.initiation_interval)
        
        # Output metrics
        total_output_bandwidth_bits = 0
        
        for name, out in self._output_map.items():
            out_metrics = out.calculate_performance_metrics()
            metrics["outputs"][name] = out_metrics
            total_output_bandwidth_bits += out.bandwidth_bits
        
        # Aggregate metrics
        metrics["aggregate"] = {
            "initiation_interval": self.initiation_interval(),
            "total_input_bandwidth_bits": total_input_bandwidth_bits,
            "total_output_bandwidth_bits": total_output_bandwidth_bits,
            "total_bandwidth_mbps": ((total_input_bandwidth_bits + total_output_bandwidth_bits) * 
                                   frequency_mhz) / 8.0,
            "throughput_fps": self.throughput_fps()
        }
        
        return metrics
    
    def clear_cache(self) -> None:
        """Clear all cached performance metrics"""
        self._cached_metrics.clear()
        # Clear input caches
        for inp in self.input_models:
            inp._invalidate_performance_cache()
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"KernelModel(name='{self.name}', "
            f"inputs={len(self.input_models)}, "
            f"outputs={len(self.output_models)}, "
            f"throughput={self.throughput_fps():.1f}fps)"
        )
