"""
DataflowModel: Unified computational model for interface relationships

This module provides the core computational model implementing mathematical 
relationships between interfaces and parallelism parameters with unified
initiation interval calculations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from .interface_types import InterfaceType
from .dataflow_interface import DataflowInterface 
from .validation import ValidationResult, create_validation_result, ValidationError, ValidationSeverity

@dataclass
class InitiationIntervals:
    """
    Container for initiation interval and cycle latency calculations.
    
    Provides complete cycle timing specification for dataflow operations.
    """
    cII: Dict[str, int]  # Calculation Initiation Interval per input interface
    eII: Dict[str, int]  # Execution Initiation Interval per input interface  
    L: int               # Overall inference cycle latency (total cycles for complete operation)
    bottleneck_analysis: Dict[str, Any]  # Detailed performance bottleneck information

@dataclass 
class ParallelismBounds:
    """Valid bounds for parallelism parameters"""
    interface_name: str
    min_value: int
    max_value: int
    divisibility_constraints: List[int]  # Values that must divide evenly

@dataclass
class ParallelismConfiguration:
    """Complete parallelism configuration for a kernel"""
    iPar: Dict[str, int]  # Input parallelism per input interface
    wPar: Dict[str, int]  # Weight parallelism per weight interface
    derived_stream_dims: Dict[str, List[int]]  # Computed stream dimensions

class DataflowModel:
    """
    Core computational model implementing mathematical relationships
    between interfaces and parallelism parameters.
    """
    
    def __init__(self, interfaces: List[DataflowInterface], parameters: Dict[str, Any]):
        self.interfaces = self._organize_interfaces(interfaces)
        self.parameters = parameters
        self.constraints = self._extract_constraints()
        self.computation_graph = self._build_computation_graph()
        
        # Performance state - calculated when parallelism is applied
        self._current_parallelism: Optional[Dict[str, Dict[str, int]]] = None
        self._cached_intervals: Optional[InitiationIntervals] = None
        self._parallelism_applied = False
    
    def _organize_interfaces(self, interfaces: List[DataflowInterface]) -> Dict[str, DataflowInterface]:
        """Organize interfaces by name for easy access"""
        return {iface.name: iface for iface in interfaces}
    
    def _extract_constraints(self) -> List[Any]:
        """Extract constraints from all interfaces"""
        constraints = []
        for iface in self.interfaces.values():
            constraints.extend(iface.constraints)
        return constraints
    
    def _build_computation_graph(self) -> Dict[str, Any]:
        """Build computation graph representing interface relationships"""
        # For now, return basic metadata about interfaces
        # This can be extended for complex dependency analysis
        return {
            "input_count": len(self.input_interfaces),
            "output_count": len(self.output_interfaces),
            "weight_count": len(self.weight_interfaces)
        }
    
    @property
    def input_interfaces(self) -> List[DataflowInterface]:
        """All INPUT type interfaces"""
        return [iface for iface in self.interfaces.values() 
                if iface.interface_type == InterfaceType.INPUT]
    
    @property  
    def output_interfaces(self) -> List[DataflowInterface]:
        """All OUTPUT type interfaces"""
        return [iface for iface in self.interfaces.values() 
                if iface.interface_type == InterfaceType.OUTPUT]
    
    @property
    def weight_interfaces(self) -> List[DataflowInterface]:
        """All WEIGHT type interfaces"""
        return [iface for iface in self.interfaces.values() 
                if iface.interface_type == InterfaceType.WEIGHT]
    
    
    def apply_parallelism(self, iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals:
        """
        Apply parallelism parameters and recalculate all performance metrics atomically.
        
        This is the ONLY method that modifies interface stream_dims.
        All calculations are done consistently in one operation.
        
        Args:
            iPar: Input parallelism per input interface {interface_name: parallelism}
            wPar: Weight parallelism per weight interface {interface_name: parallelism}
            
        Returns:
            InitiationIntervals containing:
            - cII: Calculation Initiation Interval per input interface
            - eII: Execution Initiation Interval per input interface
            - L: Inference Cycle Latency
            - bottleneck_analysis: Performance bottleneck information
        """
        # Store parallelism configuration
        self._current_parallelism = {"iPar": iPar.copy(), "wPar": wPar.copy()}
        
        # Update all interface stream_dims atomically
        self._update_all_stream_dimensions(iPar, wPar)
        
        # Recalculate performance metrics
        self._cached_intervals = self._calculate_intervals_internal()
        self._parallelism_applied = True
        
        return self._cached_intervals
    
    def calculate_initiation_intervals(self, iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals:
        """
        Public API: Apply parallelism and return calculated intervals.
        
        This replaces the old method and ensures atomic updates.
        """
        return self.apply_parallelism(iPar, wPar)
    
    def _update_all_stream_dimensions(self, iPar: Dict[str, int], wPar: Dict[str, int]) -> None:
        """Update stream dimensions for all interfaces atomically."""
        input_interfaces = self.input_interfaces
        weight_interfaces = self.weight_interfaces
        output_interfaces = self.output_interfaces
        
        # Update input interface stream dimensions
        for input_if in input_interfaces:
            input_parallelism = iPar.get(input_if.name, 1)
            if len(input_if.stream_dims) > 0:
                input_if.stream_dims[0] = input_parallelism
        
        # Update weight interface stream dimensions
        for weight_if in weight_interfaces:
            weight_parallelism = wPar.get(weight_if.name, 1)
            if len(weight_if.stream_dims) > 0:
                # Calculate stream_dims_W = wPar * iPar * (block_dims_W / block_dims_I)
                # Use first input interface as reference
                if input_interfaces:
                    input_if = input_interfaces[0]
                    input_parallelism = iPar.get(input_if.name, 1)
                    
                    if (len(input_if.block_dims) > 0 and
                        len(weight_if.block_dims) > 0 and
                        input_if.block_dims[0] != 0):
                        scaling_factor = (weight_if.block_dims[0] // input_if.block_dims[0]
                                        if weight_if.block_dims[0] >= input_if.block_dims[0] else 1)
                    else:
                        scaling_factor = 1
                    
                    weight_if.stream_dims[0] = weight_parallelism * input_parallelism * scaling_factor
                else:
                    weight_if.stream_dims[0] = weight_parallelism
        
        # Update output interface stream dimensions based on bottleneck
        if input_interfaces and output_interfaces:
            # Find bottleneck input (highest eII)
            bottleneck_input = self._find_bottleneck_input(input_interfaces, weight_interfaces, iPar, wPar)
            bottleneck_parallelism = iPar.get(bottleneck_input.name, 1)
            
            for output_if in output_interfaces:
                if (len(output_if.stream_dims) > 0 and
                    len(bottleneck_input.block_dims) > 0 and
                    len(output_if.block_dims) > 0 and
                    bottleneck_input.block_dims[0] != 0):
                    
                    scaling_factor = (output_if.block_dims[0] // bottleneck_input.block_dims[0]
                                    if output_if.block_dims[0] >= bottleneck_input.block_dims[0] else 1)
                    output_if.stream_dims[0] = bottleneck_parallelism * scaling_factor
    
    def _find_bottleneck_input(self, input_interfaces: List[DataflowInterface],
                              weight_interfaces: List[DataflowInterface],
                              iPar: Dict[str, int], wPar: Dict[str, int]) -> DataflowInterface:
        """Find input interface with highest execution interval (bottleneck)."""
        max_eII = 0
        bottleneck_input = input_interfaces[0]
        
        for input_if in input_interfaces:
            input_name = input_if.name
            input_parallelism = iPar.get(input_name, 1)
            
            # Calculate cII for this input
            cII = input_if.calculate_cII()
            
            # Find maximum weight constraint
            max_weight_cycles = 1
            for weight_if in weight_interfaces:
                weight_name = weight_if.name
                weight_parallelism = wPar.get(weight_name, 1)
                weight_cycles = self._calculate_weight_cycles_simple(weight_if, weight_parallelism)
                max_weight_cycles = max(max_weight_cycles, weight_cycles)
            
            # Calculate eII
            eII = cII * max_weight_cycles
            
            if eII > max_eII:
                max_eII = eII
                bottleneck_input = input_if
        
        return bottleneck_input
    
    def _calculate_weight_cycles_simple(self, weight_if: DataflowInterface, weight_parallelism: int) -> int:
        """Calculate weight loading cycles."""
        weight_cycles = 1
        num_blocks = weight_if.get_num_blocks()
        for num_block in num_blocks:
            if weight_parallelism > 0:
                weight_cycles *= (num_block + weight_parallelism - 1) // weight_parallelism
        return max(weight_cycles, 1)
    
    def _calculate_intervals_internal(self) -> InitiationIntervals:
        """Calculate initiation intervals using current stream_dims."""
        input_interfaces = self.input_interfaces
        weight_interfaces = self.weight_interfaces
        
        if not input_interfaces:
            return InitiationIntervals(cII={}, eII={}, L=1, bottleneck_analysis={})
        
        cII_per_input = {}
        eII_per_input = {}
        
        # Calculate intervals using current stream_dims (already updated)
        for input_if in input_interfaces:
            input_name = input_if.name
            
            # Calculate cII using current stream_dims
            cII_per_input[input_name] = input_if.calculate_cII()
            
            # Find maximum weight constraint
            max_weight_cycles = 1
            for weight_if in weight_interfaces:
                weight_name = weight_if.name
                weight_parallelism = self._current_parallelism["wPar"].get(weight_name, 1)
                weight_cycles = self._calculate_weight_cycles_simple(weight_if, weight_parallelism)
                max_weight_cycles = max(max_weight_cycles, weight_cycles)
            
            # Calculate eII
            eII_per_input[input_name] = cII_per_input[input_name] * max_weight_cycles
        
        # Find bottleneck and calculate L
        bottleneck_input_name = max(eII_per_input.keys(), key=lambda name: eII_per_input[name])
        bottleneck_input = self.interfaces[bottleneck_input_name]
        
        num_blocks = np.prod(bottleneck_input.get_num_blocks())
        L = eII_per_input[bottleneck_input_name] * num_blocks
        
        bottleneck_analysis = {
            "bottleneck_input": bottleneck_input_name,
            "bottleneck_eII": eII_per_input[bottleneck_input_name],
            "bottleneck_cII": cII_per_input[bottleneck_input_name],
            "bottleneck_num_blocks": num_blocks,
            "total_inference_cycles": L
        }
        
        return InitiationIntervals(
            cII=cII_per_input,
            eII=eII_per_input,
            L=L,
            bottleneck_analysis=bottleneck_analysis
        )
    
    def get_current_intervals(self) -> Optional[InitiationIntervals]:
        """Get currently cached intervals (if parallelism has been applied)."""
        if self._parallelism_applied:
            return self._cached_intervals
        return None
    
    def reset_parallelism(self) -> None:
        """Reset all interfaces to default stream_dims."""
        for interface in self.interfaces.values():
            # Reset to default stream_dims (all 1s)
            interface.stream_dims = [1] * len(interface.stream_dims)
        
        self._current_parallelism = None
        self._cached_intervals = None
        self._parallelism_applied = False
    
    def validate_mathematical_constraints(self) -> ValidationResult:
        """
        Validate mathematical relationships between dimensions
        """
        result = create_validation_result()
        
        for interface in self.interfaces.values():
            interface_result = interface.validate_constraints()
            result.merge(interface_result)
        
        return result
    
    def get_parallelism_bounds(self) -> Dict[str, ParallelismBounds]:
        """
        Calculate valid bounds for iPar/wPar parameters for FINN optimization
        
        Returns dictionary mapping parameter names to their parallelism bounds
        """
        bounds = {}
        
        # Calculate bounds for input interfaces (iPar)
        for input_if in self.input_interfaces:
            min_val = 1
            max_val = np.prod(input_if.block_dims) if input_if.block_dims else 1
            
            # Divisibility constraints: iPar must divide block_dims values
            divisibility = []
            for block_dim in input_if.block_dims:
                # Find all divisors of block_dim
                divisors = [i for i in range(1, block_dim + 1) if block_dim % i == 0]
                divisibility.extend(divisors)
            
            bounds[f"{input_if.name}_iPar"] = ParallelismBounds(
                interface_name=input_if.name,
                min_value=min_val,
                max_value=max_val,
                divisibility_constraints=sorted(list(set(divisibility)))
            )
        
        # Calculate bounds for weight interfaces (wPar)  
        for weight_if in self.weight_interfaces:
            min_val = 1
            num_blocks = weight_if.get_num_blocks()
            max_val = np.prod(num_blocks) if num_blocks else 1
            
            # Divisibility constraints: wPar must divide num_blocks values
            divisibility = []
            for num_block in num_blocks:
                divisors = [i for i in range(1, num_block + 1) if num_block % i == 0]
                divisibility.extend(divisors)
            
            bounds[f"{weight_if.name}_wPar"] = ParallelismBounds(
                interface_name=weight_if.name,
                min_value=min_val,
                max_value=max_val,
                divisibility_constraints=sorted(list(set(divisibility)))
            )
        
        return bounds
    
    def optimize_parallelism(self, constraints: Dict[str, Any]) -> ParallelismConfiguration:
        """
        Find optimal parallelism configuration within given constraints
        """
        # This is a placeholder for optimization logic
        # In practice, this would implement search algorithms to find
        # optimal iPar/wPar values within resource constraints
        
        # For now, return default parallelism of 1 for all interfaces
        iPar = {iface.name: 1 for iface in self.input_interfaces}
        wPar = {iface.name: 1 for iface in self.weight_interfaces}
        
        # Calculate derived stream_dims
        derived_stream_dims = {}
        for iface in self.interfaces.values():
            derived_stream_dims[iface.name] = iface.stream_dims.copy()
        
        return ParallelismConfiguration(
            iPar=iPar,
            wPar=wPar,
            derived_stream_dims=derived_stream_dims
        )
