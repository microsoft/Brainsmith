"""
DataflowModel: Unified computational model for interface relationships

This module provides the core computational model implementing mathematical 
relationships between interfaces and parallelism parameters with unified
initiation interval calculations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from .dataflow_interface import DataflowInterface, DataflowInterfaceType
from .validation import ValidationResult, create_validation_result, ValidationError, ValidationSeverity

@dataclass
class InitiationIntervals:
    """Container for initiation interval calculations"""
    cII: Dict[str, int]  # Per input interface calculation intervals
    eII: Dict[str, int]  # Per input interface execution intervals
    L: int               # Overall inference latency
    bottleneck_analysis: Dict[str, Any]  # Performance bottleneck information

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
    derived_sDim: Dict[str, List[int]]  # Computed stream dimensions

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
            "weight_count": len(self.weight_interfaces),
            "config_count": len(self.config_interfaces)
        }
    
    @property
    def input_interfaces(self) -> List[DataflowInterface]:
        """All INPUT type interfaces"""
        return [iface for iface in self.interfaces.values() 
                if iface.interface_type == DataflowInterfaceType.INPUT]
    
    @property  
    def output_interfaces(self) -> List[DataflowInterface]:
        """All OUTPUT type interfaces"""
        return [iface for iface in self.interfaces.values() 
                if iface.interface_type == DataflowInterfaceType.OUTPUT]
    
    @property
    def weight_interfaces(self) -> List[DataflowInterface]:
        """All WEIGHT type interfaces"""
        return [iface for iface in self.interfaces.values() 
                if iface.interface_type == DataflowInterfaceType.WEIGHT]
    
    @property
    def config_interfaces(self) -> List[DataflowInterface]:
        """All CONFIG type interfaces"""
        return [iface for iface in self.interfaces.values() 
                if iface.interface_type == DataflowInterfaceType.CONFIG]
    
    @property
    def control_interfaces(self) -> List[DataflowInterface]:
        """All CONTROL type interfaces"""
        return [iface for iface in self.interfaces.values() 
                if iface.interface_type == DataflowInterfaceType.CONTROL]
    
    def calculate_initiation_intervals(self, iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals:
        """
        Unified calculation of cII, eII, L for given parallelism parameters.
        Handles both simple and multi-interface cases automatically.
        
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
        
        input_interfaces = self.input_interfaces
        weight_interfaces = self.weight_interfaces
        output_interfaces = self.output_interfaces
        
        if not input_interfaces:
            return InitiationIntervals(cII={}, eII={}, L=1, bottleneck_analysis={})
        
        cII_per_input = {}
        eII_per_input = {}
        
        # Calculate for each input interface
        for input_if in input_interfaces:
            input_name = input_if.name
            input_parallelism = iPar.get(input_name, 1)
            
            # Update input stream dimensions (make a copy to avoid modifying original)
            input_if_copy = self._copy_interface_with_parallelism(input_if, input_parallelism, None)
            
            # Calculate cII for this input: cII_i = ∏(tDim_i / sDim_i)
            cII_per_input[input_name] = self._calculate_cII(input_if_copy)
            
            # Find maximum weight constraint for this input
            max_weight_cycles = 1
            for weight_if in weight_interfaces:
                weight_name = weight_if.name
                weight_parallelism = wPar.get(weight_name, 1)
                
                # Update weight stream dimensions relative to this input
                # sDim_W = wPar * iPar * (tDim_W / tDim_I)
                weight_if_copy = self._copy_interface_with_weight_parallelism(
                    weight_if, weight_parallelism, input_parallelism, input_if
                )
                
                # Calculate weight cycles: ∏(qDim_W / wPar)
                weight_cycles = self._calculate_weight_cycles(weight_if_copy, weight_parallelism)
                max_weight_cycles = max(max_weight_cycles, weight_cycles)
            
            # Calculate eII for this input: eII_i = cII_i * max_weight_cycles
            eII_per_input[input_name] = cII_per_input[input_name] * max_weight_cycles
        
        # Determine bottleneck and overall latency
        bottleneck_input_name = max(eII_per_input.keys(), key=lambda name: eII_per_input[name])
        bottleneck_input = self.interfaces[bottleneck_input_name]
        
        # L = eII_bottleneck * ∏(qDim_bottleneck)
        L = eII_per_input[bottleneck_input_name] * np.prod(bottleneck_input.qDim)
        
        # Update output stream dimensions based on bottleneck
        bottleneck_parallelism = iPar.get(bottleneck_input_name, 1)
        self._update_output_stream_dimensions(output_interfaces, bottleneck_input, bottleneck_parallelism)
        
        bottleneck_analysis = {
            "bottleneck_input": bottleneck_input_name,
            "bottleneck_eII": eII_per_input[bottleneck_input_name],
            "bottleneck_qDim": bottleneck_input.qDim,
            "total_inputs": len(input_interfaces),
            "total_weights": len(weight_interfaces)
        }
        
        return InitiationIntervals(
            cII=cII_per_input,
            eII=eII_per_input,
            L=L,
            bottleneck_analysis=bottleneck_analysis
        )
    
    def _copy_interface_with_parallelism(self, interface: DataflowInterface, 
                                       input_parallelism: int, 
                                       weight_parallelism: Optional[int]) -> DataflowInterface:
        """Create a copy of interface with updated stream dimensions"""
        # Create a shallow copy and update sDim
        new_sDim = interface.sDim.copy()
        
        if interface.interface_type == DataflowInterfaceType.INPUT:
            new_sDim[0] = input_parallelism
        elif interface.interface_type == DataflowInterfaceType.WEIGHT and weight_parallelism is not None:
            new_sDim[0] = weight_parallelism
        
        # For simplicity, we'll modify the original interface's sDim
        # In a production implementation, we might want true copying
        original_sDim = interface.sDim.copy()
        interface.sDim = new_sDim
        
        return interface
    
    def _copy_interface_with_weight_parallelism(self, weight_if: DataflowInterface,
                                              weight_parallelism: int,
                                              input_parallelism: int,
                                              input_if: DataflowInterface) -> DataflowInterface:
        """Create copy of weight interface with computed stream dimensions"""
        # sDim_W = wPar * iPar * (tDim_W / tDim_I)
        if len(input_if.tDim) > 0 and input_if.tDim[0] != 0:
            scaling_factor = weight_if.tDim[0] // input_if.tDim[0] if weight_if.tDim[0] >= input_if.tDim[0] else 1
        else:
            scaling_factor = 1
            
        new_sDim = weight_if.sDim.copy()
        new_sDim[0] = weight_parallelism * input_parallelism * scaling_factor
        
        # Update the interface (temporary modification)
        original_sDim = weight_if.sDim.copy()
        weight_if.sDim = new_sDim
        
        return weight_if
    
    def _calculate_cII(self, interface: DataflowInterface) -> int:
        """Calculate calculation initiation interval for an interface"""
        cII = 1
        for tdim, sdim in zip(interface.tDim, interface.sDim):
            if sdim > 0:
                cII *= tdim // sdim
        return max(cII, 1)
    
    def _calculate_weight_cycles(self, weight_if: DataflowInterface, weight_parallelism: int) -> int:
        """Calculate weight loading cycles"""
        weight_cycles = 1
        for qdim in weight_if.qDim:
            if weight_parallelism > 0:
                weight_cycles *= (qdim + weight_parallelism - 1) // weight_parallelism
        return max(weight_cycles, 1)
    
    def _update_output_stream_dimensions(self, output_interfaces: List[DataflowInterface],
                                       bottleneck_input: DataflowInterface,
                                       bottleneck_parallelism: int) -> None:
        """Update output stream dimensions based on bottleneck input"""
        for output_if in output_interfaces:
            if len(output_if.sDim) > 0 and len(bottleneck_input.tDim) > 0:
                # sDim_O = iPar_bottleneck * (tDim_O / tDim_I_bottleneck)
                if bottleneck_input.tDim[0] != 0:
                    scaling_factor = output_if.tDim[0] // bottleneck_input.tDim[0] if output_if.tDim[0] >= bottleneck_input.tDim[0] else 1
                else:
                    scaling_factor = 1
                output_if.sDim[0] = bottleneck_parallelism * scaling_factor
    
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
            max_val = np.prod(input_if.tDim) if input_if.tDim else 1
            
            # Divisibility constraints: iPar must divide tDim values
            divisibility = []
            for tdim in input_if.tDim:
                # Find all divisors of tdim
                divisors = [i for i in range(1, tdim + 1) if tdim % i == 0]
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
            max_val = np.prod(weight_if.qDim) if weight_if.qDim else 1
            
            # Divisibility constraints: wPar must divide qDim values
            divisibility = []
            for qdim in weight_if.qDim:
                divisors = [i for i in range(1, qdim + 1) if qdim % i == 0]
                divisibility.extend(divisors)
            
            bounds[f"{weight_if.name}_wPar"] = ParallelismBounds(
                interface_name=weight_if.name,
                min_value=min_val,
                max_value=max_val,
                divisibility_constraints=sorted(list(set(divisibility)))
            )
        
        return bounds
    
    def get_resource_requirements(self, parallelism_config: ParallelismConfiguration) -> Dict[str, Any]:
        """
        Estimate resource requirements for given parallelism configuration
        """
        requirements = {
            "memory_bits": 0,
            "transfer_bandwidth": 0,
            "computation_cycles": 0
        }
        
        # Calculate memory requirements
        for interface in self.interfaces.values():
            if interface.interface_type in [DataflowInterfaceType.INPUT, DataflowInterfaceType.WEIGHT]:
                requirements["memory_bits"] += interface.get_memory_footprint()
        
        # Calculate bandwidth requirements
        for interface in self.interfaces.values():
            if interface.interface_type in [DataflowInterfaceType.INPUT, DataflowInterfaceType.OUTPUT, DataflowInterfaceType.WEIGHT]:
                requirements["transfer_bandwidth"] += interface.calculate_stream_width()
        
        # Calculate computation cycles using current parallelism
        intervals = self.calculate_initiation_intervals(
            parallelism_config.iPar, 
            parallelism_config.wPar
        )
        requirements["computation_cycles"] = intervals.L
        
        return requirements
    
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
        
        # Calculate derived sDim
        derived_sDim = {}
        for iface in self.interfaces.values():
            derived_sDim[iface.name] = iface.sDim.copy()
        
        return ParallelismConfiguration(
            iPar=iPar,
            wPar=wPar,
            derived_sDim=derived_sDim
        )
