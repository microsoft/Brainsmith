############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unified kernel definition"""

from dataclasses import dataclass, field, replace
from typing import List, Dict, Tuple, Optional, Set, Union
import math
from .interface import Interface
from .relationships import (
    DimensionRelationship, ArchitecturalConstraint, ParameterDependency,
    ValidationResult, RelationType
)
from .validators import KernelValidator
from .expressions import evaluate_expression
from .types import InterfaceDirection, prod


@dataclass
class Kernel:
    """Hardware kernel with interfaces and native constraints
    
    Represents a hardware accelerator kernel with:
    - Multiple streaming interfaces (input/output/weight/config)
    - Timing characteristics (latency, initiation intervals)
    - Pipeline costs (priming/flush cycles)
    - Native relationship and constraint modeling
    - Resource estimates and architectural requirements
    """
    
    # Identity
    name: str
    hw_module: Optional[str] = None  # SystemVerilog module name
    
    # Interfaces
    interfaces: List[Interface] = field(default_factory=list)
    
    # Timing characteristics
    latency_cycles: Tuple[int, int] = (1, 1)  # (worst_case, average)
    calculation_ii: Optional[int] = None  # Initiation interval for one calculation
    execution_ii: Optional[int] = None    # Initiation interval for one execution
    
    # Pipeline costs
    priming_cycles: int = 0  # Cycles to fill pipeline
    flush_cycles: int = 0    # Cycles to drain pipeline
    
    # Native relationships and constraints
    relationships: List[DimensionRelationship] = field(default_factory=list)
    constraints: List[ArchitecturalConstraint] = field(default_factory=list)
    dependencies: List[ParameterDependency] = field(default_factory=list)
    
    # Architectural requirements
    requires_burst_alignment: bool = False
    requires_power_of_two: Set[str] = field(default_factory=set)  # Interface names
    memory_architecture: Optional[str] = None  # "distributed", "HBM", "DDR", etc.
    pipeline_style: Optional[str] = None  # "streaming", "batch", "hybrid"
    
    # Resource estimates
    resources: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate kernel configuration"""
        # Set hw_module to name if not specified
        if self.hw_module is None:
            self.hw_module = self.name
        
        # Validate interface names are unique
        names = [intf.name for intf in self.interfaces]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate interface names: {set(duplicates)}")
        
        # Validate timing
        if len(self.latency_cycles) != 2:
            raise ValueError("latency_cycles must be (worst_case, average) tuple")
        
        if self.latency_cycles[0] < self.latency_cycles[1]:
            raise ValueError("Worst-case latency must be >= average latency")
        
        if self.priming_cycles < 0 or self.flush_cycles < 0:
            raise ValueError("Pipeline costs must be non-negative")
    
    def validate(self) -> ValidationResult:
        """Validate kernel configuration including native constraints
        
        Returns:
            ValidationResult with any violations
        """
        # Create validator
        validator = KernelValidator(self.relationships, self.constraints, self.dependencies)
        
        # Build context for validation
        context = {
            'interfaces': {intf.name: intf for intf in self.interfaces},
            'parameters': {},  # Will be populated with runtime parameters
            'constants': {}    # Will be populated with design-time constants
        }
        
        # Validate all constraints
        result = validator.validate(context)
        
        return result
    
    def validate_and_raise(self) -> None:
        """Validate kernel and raise exception if invalid
        
        Raises:
            ValueError: If any constraint is violated
        """
        result = self.validate()
        result.raise_if_invalid(self.name)
    
    @property
    def input_interfaces(self) -> List[Interface]:
        """Get all input interfaces"""
        return [i for i in self.interfaces 
                if i.direction == InterfaceDirection.INPUT]
    
    @property
    def output_interfaces(self) -> List[Interface]:
        """Get all output interfaces"""
        return [i for i in self.interfaces 
                if i.direction == InterfaceDirection.OUTPUT]
    
    @property
    def weight_interfaces(self) -> List[Interface]:
        """Get all weight interfaces"""
        return [i for i in self.interfaces 
                if i.direction == InterfaceDirection.WEIGHT]
    
    @property
    def config_interfaces(self) -> List[Interface]:
        """Get all config interfaces"""
        return [i for i in self.interfaces 
                if i.direction == InterfaceDirection.CONFIG]
    
    @property
    def has_weights(self) -> bool:
        """Check if kernel has weight interfaces"""
        return len(self.weight_interfaces) > 0
    
    @property
    def is_stateful(self) -> bool:
        """Check if kernel maintains internal state"""
        return self.priming_cycles > 0 or self.flush_cycles > 0
    
    def get_interface(self, name: str) -> Interface:
        """Get interface by name
        
        Raises:
            KeyError: If interface not found
        """
        for intf in self.interfaces:
            if intf.name == name:
                return intf
        raise KeyError(f"Interface '{name}' not found in kernel '{self.name}'")
    
    def add_relationship(self, source: str, target: str,
                        relation: RelationType = RelationType.EQUAL,
                        source_dim: Optional[int] = None,
                        target_dim: Optional[int] = None,
                        factor: Optional[Union[int, float]] = None,
                        description: str = "") -> None:
        """Add a dimension relationship between interfaces
        
        Args:
            source: Source interface name
            target: Target interface name
            relation: Type of relationship
            source_dim: Source dimension index (None for total size)
            target_dim: Target dimension index (None for total size)
            factor: Factor for MULTIPLE relationships
            description: Human-readable description
        """
        rel = DimensionRelationship(
            source_interface=source,
            target_interface=target,
            relation=relation,
            source_dim=source_dim,
            target_dim=target_dim,
            factor=factor,
            description=description
        )
        self.relationships.append(rel)
        
        # Update interface dataflow metadata
        try:
            src_intf = self.get_interface(source)
            tgt_intf = self.get_interface(target)
            
            # Add dataflow connections for certain relationship types
            if relation in [RelationType.EQUAL, RelationType.MULTIPLE]:
                src_intf.add_produces(target)
                tgt_intf.add_consumes(source)
        except KeyError:
            pass  # Interface might not exist yet
    
    def add_constraint(self, name: str, expression: str,
                      operator: str, value: Union[int, float, str],
                      description: str = "") -> None:
        """Add an architectural constraint
        
        Args:
            name: Constraint name
            expression: Expression to evaluate
            operator: Comparison operator (==, <=, >=, etc.)
            value: Target value or expression
            description: Human-readable description
        """
        constraint = ArchitecturalConstraint(
            name=name,
            expression=expression,
            operator=operator,
            value=value,
            description=description
        )
        self.constraints.append(constraint)
    
    def add_dependency(self, dependent: str, expression: str,
                      description: str = "") -> None:
        """Add a parameter dependency
        
        Args:
            dependent: Name of dependent parameter
            expression: Expression to compute parameter value
            description: Human-readable description
        """
        dependency = ParameterDependency(
            dependent=dependent,
            expression=expression,
            description=description
        )
        self.dependencies.append(dependency)
    
    def get_relationships_for_interface(self, interface_name: str) -> List[DimensionRelationship]:
        """Get all relationships involving a specific interface"""
        return [rel for rel in self.relationships 
                if rel.source_interface == interface_name or rel.target_interface == interface_name]
    
    def get_dependent_interfaces(self, interface_name: str) -> Set[str]:
        """Get all interfaces that depend on the given interface"""
        dependents = set()
        for rel in self.relationships:
            if rel.source_interface == interface_name:
                dependents.add(rel.target_interface)
        return dependents
    
    def get_constraint_graph(self) -> Dict[str, Set[str]]:
        """Get constraint dependency graph
        
        Returns:
            Dict mapping interface names to their dependencies
        """
        graph = {}
        for rel in self.relationships:
            if rel.target_interface not in graph:
                graph[rel.target_interface] = set()
            graph[rel.target_interface].add(rel.source_interface)
        return graph
    
    def initiation_interval(self) -> int:
        """Compute kernel initiation interval
        
        Returns worst-case II across all input interfaces.
        Falls back to calculation_ii if specified.
        """
        if self.calculation_ii is not None:
            return self.calculation_ii
        
        # Default: maximum II across input interfaces
        if not self.input_interfaces:
            return 1
        
        max_ii = 1
        for intf in self.input_interfaces:
            max_ii = max(max_ii, max(intf.ii_pattern))
        
        return max_ii
    
    def execution_interval(self) -> int:
        """Compute execution interval for processing all weights
        
        Returns cycles for one complete execution with all weight blocks.
        Falls back to execution_ii if specified.
        """
        if self.execution_ii is not None:
            return self.execution_ii
        
        # Default: calculation_ii * number of weight blocks
        calc_ii = self.initiation_interval()
        
        if not self.weight_interfaces:
            return calc_ii
        
        # Assume all weights processed sequentially
        total_weight_blocks = 1
        for w_intf in self.weight_interfaces:
            total_weight_blocks *= w_intf.tokens_per_inference
        
        return calc_ii * total_weight_blocks
    
    def inference_latency(self, batch_size: int = 1) -> int:
        """Compute total inference latency
        
        Args:
            batch_size: Number of inferences to process
            
        Returns:
            Total cycles including pipeline costs
        """
        # Execution cycles for all input blocks
        exec_cycles = 0
        for i_intf in self.input_interfaces:
            n_blocks = i_intf.tokens_per_inference * batch_size
            exec_cycles = max(exec_cycles, n_blocks * self.execution_interval())
        
        # Add pipeline costs
        total = self.priming_cycles + exec_cycles + self.flush_cycles
        
        return total
    
    def throughput(self, clock_freq_mhz: float = 100.0) -> float:
        """Compute kernel throughput
        
        Args:
            clock_freq_mhz: Clock frequency in MHz
            
        Returns:
            Throughput in inferences/second
        """
        # Cycles per inference (steady state, no pipeline costs)
        cycles_per_inf = self.execution_interval()
        
        # Account for multiple input interfaces
        for i_intf in self.input_interfaces:
            cycles_per_inf = max(cycles_per_inf, 
                               i_intf.tokens_per_inference * self.execution_interval())
        
        # Convert to inferences/second
        return (clock_freq_mhz * 1e6) / cycles_per_inf
    
    def bandwidth_requirements(self) -> Dict[str, float]:
        """Compute bandwidth requirements per interface
        
        Returns:
            Dict mapping interface name to bandwidth in bits/cycle
        """
        bandwidth = {}
        
        for intf in self.interfaces:
            # Only streaming interfaces consume bandwidth
            if intf.direction != InterfaceDirection.CONFIG:
                bandwidth[intf.name] = intf.bandwidth_bits
        
        return bandwidth
    
    def estimate_resources(self) -> Dict[str, float]:
        """Estimate resource usage
        
        Returns resource dict if resources are specified,
        otherwise returns simple estimates based on interfaces.
        """
        if self.resources:
            return self.resources.copy()
        
        # Simple estimates based on interface parallelism
        estimates = {
            "LUT": 0,
            "FF": 0,
            "DSP": 0,
            "BRAM": 0
        }
        
        # Rough estimates per interface
        for intf in self.interfaces:
            par = intf.ipar
            bits = intf.dtype.bits
            
            # Datapath logic
            estimates["LUT"] += par * bits * 10  # Rough estimate
            estimates["FF"] += par * bits * 5
            
            # DSPs for arithmetic (mainly for weight interfaces)
            if intf.direction == InterfaceDirection.WEIGHT:
                estimates["DSP"] += math.ceil(par / 2)  # Assume DSP packing
            
            # Buffer memory (assume small FIFOs)
            estimates["BRAM"] += math.ceil((par * bits * 64) / 36000)  # 36Kb BRAMs
        
        return estimates
    
    def to_adfg_rates(self) -> Dict[str, List[int]]:
        """Convert to ADFG rate dictionary for scheduling
        
        Returns:
            Dict mapping interface names to CSDF rate patterns
        """
        rates = {}
        for intf in self.interfaces:
            if intf.direction != InterfaceDirection.CONFIG:
                rates[intf.name] = intf.rate_pattern
        return rates
    
    def apply_parallelism(self, config: Dict[str, int]) -> "Kernel":
        """Apply parallelism configuration to create new kernel instance
        
        Args:
            config: Dict mapping interface names to parallelism values
            
        Returns:
            New kernel with updated stream dimensions
        """
        new_interfaces = []
        
        for intf in self.interfaces:
            if intf.name in config:
                # Compute new stream dims to achieve target parallelism
                target_par = config[intf.name]
                
                # Simple approach: scale first dimension
                # TODO: More sophisticated dimension selection
                scale = target_par / intf.ipar
                new_stream_dims = list(intf.stream_dims)
                new_stream_dims[0] = int(new_stream_dims[0] * scale)
                
                new_intf = replace(intf, stream_dims=tuple(new_stream_dims))
                new_interfaces.append(new_intf)
            else:
                new_interfaces.append(intf)
        
        return replace(self, interfaces=new_interfaces)
    
    def __repr__(self) -> str:
        """String representation"""
        n_in = len(self.input_interfaces)
        n_out = len(self.output_interfaces)
        n_weight = len(self.weight_interfaces)
        
        return (
            f"Kernel(name='{self.name}', "
            f"interfaces=[{n_in} in, {n_out} out, {n_weight} weight], "
            f"latency={self.latency_cycles}, "
            f"pragmas={len(self.pragmas)})"
        )