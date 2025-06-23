############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Kernel model for performance calculations and runtime behavior"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any, Set
import math
from .base import BaseModel, ParameterBinding
from .interface_model import InterfaceModel
from .types import InterfaceDirection, prod
from .relationships import RelationType


@dataclass 
class KernelModel(BaseModel):
    """Runtime model of a kernel optimized for performance calculations
    
    Focuses on timing, throughput, and resource modeling. Assumes valid
    configuration and skips validation for performance.
    """
    
    # Runtime configuration
    interface_models: List[InterfaceModel] = field(default_factory=list)
    parameter_binding: ParameterBinding = field(default_factory=lambda: ParameterBinding({}))
    
    # Timing characteristics (actual measured or estimated values)
    latency_cycles: Tuple[int, int] = (1, 1)  # (worst_case, average)
    calculation_ii: Optional[int] = None  # Initiation interval for one calculation
    execution_ii: Optional[int] = None    # Initiation interval for one execution
    
    # Pipeline characteristics
    priming_cycles: int = 0  # Cycles to fill pipeline
    flush_cycles: int = 0    # Cycles to drain pipeline
    pipeline_depth: int = 1  # Depth of processing pipeline
    
    # Resource usage (actual measured values)
    resources: Dict[str, float] = field(default_factory=dict)
    power_watts: float = 0.0  # Estimated power consumption
    
    # Performance characteristics
    clock_freq_mhz: float = 100.0  # Operating frequency
    actual_efficiency: float = 1.0  # Actual vs theoretical efficiency
    
    def __init__(self, interface_models: List[InterfaceModel] = None,
                 parameter_binding: ParameterBinding = None,
                 definition: Optional['KernelDefinition'] = None,
                 **kwargs):
        """Initialize kernel model"""
        # Call parent constructor
        super().__init__(definition)
        
        # Set fields
        self.interface_models = interface_models or []
        self.parameter_binding = parameter_binding or ParameterBinding({})
        
        # Initialize default values
        self.latency_cycles = kwargs.get('latency_cycles', (1, 1))
        self.calculation_ii = kwargs.get('calculation_ii', None)
        self.execution_ii = kwargs.get('execution_ii', None)
        self.priming_cycles = kwargs.get('priming_cycles', 0)
        self.flush_cycles = kwargs.get('flush_cycles', 0)
        self.pipeline_depth = kwargs.get('pipeline_depth', 1)
        self.resources = kwargs.get('resources', {})
        self.power_watts = kwargs.get('power_watts', 0.0)
        self.clock_freq_mhz = kwargs.get('clock_freq_mhz', 100.0)
        self.actual_efficiency = kwargs.get('actual_efficiency', 1.0)
        
        # Call post init
        self.__post_init__()
    
    def __post_init__(self):
        """Initialize model with optimized setup"""
        # Build interface name mapping for quick lookup
        self._interface_map = {intf.definition.name if intf.definition else f"intf_{i}": intf 
                              for i, intf in enumerate(self.interface_models)}
        
        # Cache for performance calculations
        self._cached_metrics = {}
        
        # Validate basic timing constraints
        if self.latency_cycles[0] < self.latency_cycles[1]:
            self.latency_cycles = (self.latency_cycles[1], self.latency_cycles[1])
    
    @property
    def name(self) -> str:
        """Get kernel name from definition"""
        return self.definition.name if self.definition else "unnamed_kernel"
    
    @property
    def input_models(self) -> List[InterfaceModel]:
        """Get all input interface models"""
        return [intf for intf in self.interface_models 
                if intf.definition and intf.definition.direction == InterfaceDirection.INPUT]
    
    @property
    def output_models(self) -> List[InterfaceModel]:
        """Get all output interface models"""
        return [intf for intf in self.interface_models 
                if intf.definition and intf.definition.direction == InterfaceDirection.OUTPUT]
    
    @property
    def weight_models(self) -> List[InterfaceModel]:
        """Get all weight interface models"""
        return [intf for intf in self.interface_models 
                if intf.definition and intf.definition.direction == InterfaceDirection.WEIGHT]
    
    @property
    def config_models(self) -> List[InterfaceModel]:
        """Get all config interface models"""
        return [intf for intf in self.interface_models 
                if intf.definition and intf.definition.direction == InterfaceDirection.CONFIG]
    
    def get_interface_model(self, name: str) -> Optional[InterfaceModel]:
        """Get interface model by name"""
        return self._interface_map.get(name)
    
    def initiation_interval(self) -> int:
        """Compute kernel initiation interval"""
        if "initiation_interval" not in self._cached_metrics:
            if self.calculation_ii is not None:
                ii = self.calculation_ii
            else:
                # Default: maximum II across input interfaces
                max_ii = 1
                for intf in self.input_models:
                    max_ii = max(max_ii, max(intf.ii_pattern))
                ii = max_ii
            
            self._cached_metrics["initiation_interval"] = ii
        
        return self._cached_metrics["initiation_interval"]
    
    def execution_interval(self) -> int:
        """Compute execution interval for processing all weights"""
        if "execution_interval" not in self._cached_metrics:
            if self.execution_ii is not None:
                exec_ii = self.execution_ii
            else:
                # Default: calculation_ii * number of weight blocks
                calc_ii = self.initiation_interval()
                
                if not self.weight_models:
                    exec_ii = calc_ii
                else:
                    # Assume all weights processed sequentially
                    total_weight_blocks = 1
                    for w_model in self.weight_models:
                        total_weight_blocks *= w_model.tokens_per_inference
                    exec_ii = calc_ii * total_weight_blocks
            
            self._cached_metrics["execution_interval"] = exec_ii
        
        return self._cached_metrics["execution_interval"]
    
    def inference_latency(self, batch_size: int = 1) -> int:
        """Compute total inference latency including pipeline costs"""
        if batch_size == 1 and "inference_latency" in self._cached_metrics:
            return self._cached_metrics["inference_latency"]
        
        if batch_size == 1:
            # For single inference, use the worst-case latency
            total_latency = self.latency_cycles[0]  # worst-case
        else:
            # For multiple inferences, compute based on execution cycles
            exec_cycles = 0
            for i_model in self.input_models:
                n_blocks = i_model.tokens_per_inference * batch_size
                exec_cycles = max(exec_cycles, n_blocks * self.execution_interval())
            
            # Add pipeline costs
            total_latency = self.priming_cycles + exec_cycles + self.flush_cycles
        
        if batch_size == 1:
            self._cached_metrics["inference_latency"] = total_latency
        
        return total_latency
    
    def throughput_fps(self) -> float:
        """Compute kernel throughput in inferences per second"""
        if "throughput_fps" not in self._cached_metrics:
            # Cycles per inference (steady state, no pipeline costs)
            cycles_per_inf = self.execution_interval()
            
            # Account for multiple input interfaces
            for i_model in self.input_models:
                cycles_per_inf = max(cycles_per_inf, 
                                   i_model.tokens_per_inference * self.execution_interval())
            
            # Apply efficiency factor
            effective_cycles = cycles_per_inf / self.actual_efficiency
            
            # Convert to inferences/second
            clock_hz = self.clock_freq_mhz * 1e6
            fps = clock_hz / effective_cycles
            
            self._cached_metrics["throughput_fps"] = fps
        
        return self._cached_metrics["throughput_fps"]
    
    def throughput_gops(self) -> float:
        """Compute throughput in GOPS (billions of operations per second)"""
        if "throughput_gops" not in self._cached_metrics:
            # Get total operations from parameter binding or estimate
            total_ops = self.parameter_binding.get_value("total_operations")
            
            if total_ops is None:
                # Estimate based on interfaces (rough approximation)
                total_ops = 1
                for i_model in self.input_models:
                    total_ops *= prod(i_model.tensor_dims)
                for w_model in self.weight_models:
                    total_ops *= prod(w_model.tensor_dims)
            
            fps = self.throughput_fps()
            gops = (total_ops * fps) / 1e9
            
            self._cached_metrics["throughput_gops"] = gops
        
        return self._cached_metrics["throughput_gops"]
    
    def bandwidth_requirements(self) -> Dict[str, float]:
        """Compute bandwidth requirements per interface in MB/s"""
        if "bandwidth_requirements" not in self._cached_metrics:
            bandwidth = {}
            
            for intf in self.interface_models:
                if intf.definition and intf.definition.direction != InterfaceDirection.CONFIG:
                    # Effective bandwidth accounting for utilization
                    mb_per_s = intf.effective_bandwidth(self.clock_freq_mhz)
                    bandwidth[intf.definition.name] = mb_per_s
            
            self._cached_metrics["bandwidth_requirements"] = bandwidth
        
        return self._cached_metrics["bandwidth_requirements"]
    
    def total_bandwidth_mbps(self) -> float:
        """Compute total bandwidth requirement in MB/s"""
        return sum(self.bandwidth_requirements().values())
    
    def estimate_resources(self) -> Dict[str, float]:
        """Estimate resource usage scaling from base values"""
        if self.resources:
            return self.resources.copy()
        
        # Simple estimates based on interface parallelism if no measured values
        estimates = {
            "LUT": 0,
            "FF": 0, 
            "DSP": 0,
            "BRAM": 0
        }
        
        # Scale based on total parallelism
        total_par = sum(intf.ipar for intf in self.interface_models)
        
        for intf in self.interface_models:
            par = intf.ipar
            bits = intf.definition.dtype.bits if intf.definition else 8
            
            # Datapath logic scaling
            estimates["LUT"] += par * bits * 12
            estimates["FF"] += par * bits * 6
            
            # DSPs for arithmetic (mainly for weight interfaces)
            if intf.definition and intf.definition.direction == InterfaceDirection.WEIGHT:
                estimates["DSP"] += math.ceil(par / 2)
            
            # Buffer memory scaling
            estimates["BRAM"] += math.ceil((par * bits * 128) / 36000)  # 36Kb BRAMs
        
        return estimates
    
    def apply_parallelism(self, ipar_config: Dict[str, int]) -> None:
        """Apply iPar values to interfaces and propagate through relationships
        
        Args:
            ipar_config: Dictionary mapping interface names to iPar values
        """
        # Track which interfaces have explicit parallelism set
        explicitly_set = set()
        
        # Apply iPar to each specified interface
        for intf_name, ipar_value in ipar_config.items():
            if intf_name in self._interface_map:
                self._interface_map[intf_name].ipar = ipar_value
                explicitly_set.add(intf_name)
            else:
                # Try to match by interface type + index
                for intf in self.interface_models:
                    if intf.definition and intf.definition.name == intf_name:
                        intf.ipar = ipar_value
                        explicitly_set.add(intf_name)
                        break
        
        # Propagate parallelism through relationships if we have a definition
        if self.definition:
            self._propagate_parallelism(explicitly_set)
        
        # Clear performance caches as parallelism has changed
        self.clear_cache()
    
    def _propagate_parallelism(self, explicitly_set: Optional[set] = None) -> None:
        """Propagate parallelism through interface relationships
        
        This implements sophisticated propagation that handles:
        - EQUAL relationships with dimension-aware scaling
        - MULTIPLE relationships with factor-based scaling
        - Transitive propagation through relationship chains
        - Conflict resolution when multiple sources affect a target
        
        Args:
            explicitly_set: Set of interface names that have explicit parallelism
        """
        if not self.definition:
            return
        
        if explicitly_set is None:
            explicitly_set = set()
        
        # Track propagated values to detect conflicts
        propagated: Dict[str, List[Tuple[int, str]]] = {}  # interface -> [(ipar, source), ...]
        
        # Phase 1: Direct propagation from relationships
        for rel in self.definition.relationships:
            source_intf = self.get_interface_model(rel.source_interface)
            target_intf = self.get_interface_model(rel.target_interface)
            
            if not source_intf or not target_intf:
                continue
            
            # Skip if source has no parallelism to propagate
            if source_intf.ipar <= 1:
                continue
                
            # Calculate propagated value based on relationship type
            propagated_ipar = None
            
            if rel.relation == RelationType.EQUAL:
                # For EQUAL relationships, check if dimensions allow propagation
                if self._can_propagate_parallelism(source_intf, target_intf, rel):
                    scale_factor = self._calculate_propagation_scale(source_intf, target_intf, rel)
                    propagated_ipar = max(1, int(source_intf.ipar * scale_factor))
                    
            elif rel.relation == RelationType.MULTIPLE and rel.factor:
                # For MULTIPLE relationships, scale by factor
                if rel.factor >= 1:
                    # Target is larger, so can have more parallelism
                    propagated_ipar = int(source_intf.ipar * rel.factor)
                else:
                    # Target is smaller, so needs less parallelism
                    propagated_ipar = max(1, int(source_intf.ipar * rel.factor))
                    
            elif rel.relation == RelationType.DIVISIBLE:
                # For DIVISIBLE relationships, ensure target divides source parallelism
                if target_intf.ipar == 1:
                    # Find a divisor of source ipar for target
                    for divisor in range(source_intf.ipar, 0, -1):
                        if source_intf.ipar % divisor == 0:
                            propagated_ipar = divisor
                            break
            
            # Record propagation if calculated
            if propagated_ipar is not None:
                if rel.target_interface not in propagated:
                    propagated[rel.target_interface] = []
                propagated[rel.target_interface].append((propagated_ipar, rel.source_interface))
        
        # Phase 2: Resolve conflicts and apply propagated values
        for target_name, proposals in propagated.items():
            # Skip if explicitly set by user
            if target_name in explicitly_set:
                continue
                
            target_intf = self.get_interface_model(target_name)
            if not target_intf or target_intf.ipar > 1:
                # Skip if already has parallelism set
                continue
                
            if len(proposals) == 1:
                # No conflict, apply directly
                target_intf.ipar = proposals[0][0]
            else:
                # Multiple sources - resolve conflict
                # Strategy: Use the minimum to ensure all constraints are satisfied
                min_ipar = min(p[0] for p in proposals)
                target_intf.ipar = min_ipar
        
        # Phase 3: Iterative propagation for transitive relationships
        # Continue propagating until no more changes occur
        changed = True
        iterations = 0
        max_iterations = len(self.interface_models)  # Prevent infinite loops
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for rel in self.definition.relationships:
                source_intf = self.get_interface_model(rel.source_interface)
                target_intf = self.get_interface_model(rel.target_interface)
                
                if not source_intf or not target_intf:
                    continue
                    
                # Check if this iteration can propagate new parallelism
                # Skip if target was explicitly set
                if rel.target_interface in explicitly_set:
                    continue
                    
                if source_intf.ipar > 1 and target_intf.ipar == 1:
                    if rel.relation == RelationType.EQUAL and self._can_propagate_parallelism(source_intf, target_intf, rel):
                        scale_factor = self._calculate_propagation_scale(source_intf, target_intf, rel)
                        new_ipar = max(1, int(source_intf.ipar * scale_factor))
                        target_intf.ipar = new_ipar
                        changed = True
    
    def _can_propagate_parallelism(self, source: InterfaceModel, target: InterfaceModel, 
                                   relationship: 'DimensionRelationship') -> bool:
        """Check if parallelism can be propagated between interfaces"""
        # If no specific dimensions specified, check total compatibility
        if relationship.source_dim is None or relationship.target_dim is None:
            # For total size relationships, always allow propagation
            return True
            
        # Check if dimensions are compatible
        source_dim_idx = relationship.source_dim
        target_dim_idx = relationship.target_dim
        
        # Get block dims (handle CSDF by using first phase)
        source_blocks = source.block_dims[0] if isinstance(source.block_dims[0], (list, tuple)) else source.block_dims
        target_blocks = target.block_dims[0] if isinstance(target.block_dims[0], (list, tuple)) else target.block_dims
        
        # Verify dimension indices are valid
        if (source_dim_idx < len(source_blocks) and 
            target_dim_idx < len(target_blocks)):
            
            # Check if dimensions are related
            source_dim = source_blocks[source_dim_idx]
            target_dim = target_blocks[target_dim_idx]
            
            # Can propagate if dimensions are equal or compatible
            return source_dim == target_dim or target_dim % source_dim == 0
        
        return False
    
    def _calculate_propagation_scale(self, source: InterfaceModel, target: InterfaceModel,
                                   relationship: 'DimensionRelationship') -> float:
        """Calculate scaling factor for parallelism propagation"""
        # If no specific dimensions, no scaling
        if relationship.source_dim is None or relationship.target_dim is None:
            return 1.0
            
        source_dim_idx = relationship.source_dim
        target_dim_idx = relationship.target_dim
        
        # Get block dims (handle CSDF by using first phase)
        source_blocks = source.block_dims[0] if isinstance(source.block_dims[0], (list, tuple)) else source.block_dims
        target_blocks = target.block_dims[0] if isinstance(target.block_dims[0], (list, tuple)) else target.block_dims
        
        if (source_dim_idx < len(source_blocks) and 
            target_dim_idx < len(target_blocks)):
            
            source_dim = source_blocks[source_dim_idx]
            target_dim = target_blocks[target_dim_idx]
            
            # Scale based on dimension ratio
            if source_dim > 0:
                return min(1.0, target_dim / source_dim)
        
        return 1.0
    
    def get_interface_model(self, interface_name: str) -> Optional[InterfaceModel]:
        """Get interface model by name"""
        return self._interface_map.get(interface_name)
    
    def get_parallelism_state(self) -> Dict[str, int]:
        """Get current parallelism values for all interfaces"""
        state = {}
        for intf in self.interface_models:
            if intf.definition:
                state[intf.definition.name] = intf.ipar
        return state
    
    def update_clock_frequency(self, freq_mhz: float) -> None:
        """Update clock frequency and clear dependent caches"""
        self.clock_freq_mhz = freq_mhz
        # Clear bandwidth-related caches
        self._cached_metrics.pop("bandwidth_requirements", None)
        self._cached_metrics.pop("throughput_fps", None)
        self._cached_metrics.pop("throughput_gops", None)
    
    def clear_cache(self) -> None:
        """Clear all cached performance metrics"""
        self._cached_metrics.clear()
        # Also clear interface caches
        for intf in self.interface_models:
            intf.clear_cache()
    
    def estimate_power(self) -> Dict[str, float]:
        """Estimate power consumption breakdown"""
        if self.power_watts > 0:
            return {"total_watts": self.power_watts}
        
        # Simple power estimation
        resources = self.estimate_resources()
        
        # Rough power estimates (typical FPGA values)
        power_breakdown = {
            "static_watts": 0.5,  # Base static power
            "lut_watts": resources["LUT"] * 0.000001,  # 1µW per LUT
            "dsp_watts": resources["DSP"] * 0.001,     # 1mW per DSP
            "bram_watts": resources["BRAM"] * 0.002,   # 2mW per BRAM
            "io_watts": self.total_bandwidth_mbps() * 0.0001,  # IO power
        }
        
        power_breakdown["total_watts"] = sum(power_breakdown.values())
        return power_breakdown
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        # Interface metrics
        intf_metrics = {}
        for intf in self.interface_models:
            if intf.definition:
                intf_metrics[intf.definition.name] = intf.calculate_performance_metrics()
        
        # Kernel-level metrics
        metrics = {
            # Basic properties
            "name": self.name,
            "n_interfaces": len(self.interface_models),
            "clock_freq_mhz": self.clock_freq_mhz,
            "actual_efficiency": self.actual_efficiency,
            
            # Timing
            "latency_cycles": self.latency_cycles,
            "initiation_interval": self.initiation_interval(),
            "execution_interval": self.execution_interval(),
            "inference_latency_cycles": self.inference_latency(),
            "priming_cycles": self.priming_cycles,
            "flush_cycles": self.flush_cycles,
            "pipeline_depth": self.pipeline_depth,
            
            # Throughput
            "throughput_fps": self.throughput_fps(),
            "throughput_gops": self.throughput_gops(),
            
            # Resources and Power
            "resource_estimates": self.estimate_resources(),
            "power_estimates": self.estimate_power(),
            
            # Bandwidth
            "bandwidth_requirements_mbps": self.bandwidth_requirements(),
            "total_bandwidth_mbps": self.total_bandwidth_mbps(),
            
            # Interface details
            "interface_metrics": intf_metrics,
            
            # Parameters
            "parameters": self.parameter_binding.parameters,
            "constants": self.parameter_binding.constants,
        }
        
        return metrics
    
    def simulate_execution(self, n_inferences: int = 1, 
                          detailed: bool = False) -> Dict[str, Any]:
        """Simulate kernel execution for performance analysis
        
        Args:
            n_inferences: Number of inferences to simulate
            detailed: Include detailed per-cycle information
            
        Returns:
            Simulation results
        """
        simulation = {
            "n_inferences": n_inferences,
            "total_cycles": 0,
            "pipeline_utilization": 0.0,
            "interface_activity": {},
            "resource_utilization": {}
        }
        
        # Calculate total execution time
        if n_inferences == 1:
            total_cycles = self.inference_latency()
        else:
            # Pipeline multiple inferences
            first_latency = self.inference_latency()
            additional_cycles = (n_inferences - 1) * self.execution_interval()
            total_cycles = first_latency + additional_cycles
        
        simulation["total_cycles"] = total_cycles
        
        # Calculate pipeline utilization
        productive_cycles = n_inferences * self.execution_interval()
        pipeline_util = productive_cycles / total_cycles if total_cycles > 0 else 0
        simulation["pipeline_utilization"] = pipeline_util
        
        # Interface activity simulation
        for intf in self.interface_models:
            if intf.definition:
                tokens = intf.simulate_tokens(n_inferences)
                simulation["interface_activity"][intf.definition.name] = {
                    "total_tokens": len(tokens),
                    "total_cycles": sum(t["cycles"] for t in tokens),
                    "utilization": intf.actual_utilization,
                    "effective_rate": intf.effective_rate()
                }
                
                if detailed:
                    simulation["interface_activity"][intf.definition.name]["token_details"] = tokens
        
        # Resource utilization
        resources = self.estimate_resources()
        simulation["resource_utilization"] = {
            resource: value * self.actual_efficiency 
            for resource, value in resources.items()
        }
        
        return simulation
    
    def compare_with(self, other: 'KernelModel') -> Dict[str, Any]:
        """Compare performance with another kernel model
        
        Args:
            other: Other kernel model to compare with
            
        Returns:
            Comparison results
        """
        self_metrics = self.calculate_performance_metrics()
        other_metrics = other.calculate_performance_metrics()
        
        comparison = {
            "kernel_a": self.name,
            "kernel_b": other.name,
            "throughput_ratio": self_metrics["throughput_fps"] / other_metrics["throughput_fps"],
            "latency_ratio": self_metrics["inference_latency_cycles"] / other_metrics["inference_latency_cycles"],
            "resource_ratios": {},
            "bandwidth_ratio": self_metrics["total_bandwidth_mbps"] / other_metrics["total_bandwidth_mbps"],
        }
        
        # Resource comparisons
        for resource in ["LUT", "DSP", "BRAM"]:
            a_val = self_metrics["resource_estimates"].get(resource, 0)
            b_val = other_metrics["resource_estimates"].get(resource, 0)
            comparison["resource_ratios"][resource] = a_val / b_val if b_val > 0 else float('inf')
        
        return comparison
    
    def update_clock_frequency(self, freq_mhz: float):
        """Update clock frequency and clear dependent caches"""
        self.clock_freq_mhz = freq_mhz
        # Clear frequency-dependent caches
        self._cached_metrics.pop("throughput_fps", None)
        self._cached_metrics.pop("throughput_gops", None)
        self._cached_metrics.pop("bandwidth_requirements", None)
    
    def update_efficiency(self, efficiency: float):
        """Update actual efficiency and clear dependent caches"""
        self.actual_efficiency = max(0.0, min(1.0, efficiency))
        # Clear efficiency-dependent caches
        self._cached_metrics.pop("throughput_fps", None)
        self._cached_metrics.pop("throughput_gops", None)
    
    def clear_cache(self):
        """Clear all cached performance metrics"""
        self._cached_metrics.clear()
        # Also clear interface caches
        for intf in self.interface_models:
            intf.clear_cache()
    
    def __repr__(self) -> str:
        """String representation optimized for debugging"""
        n_intf = len(self.interface_models)
        fps = self.throughput_fps()
        
        return (
            f"KernelModel(name='{self.name}', "
            f"interfaces={n_intf}, "
            f"throughput={fps:.1f}fps, "
            f"latency={self.inference_latency()}cyc)"
        )