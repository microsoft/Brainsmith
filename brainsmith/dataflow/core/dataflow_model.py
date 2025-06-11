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
            
            # Calculate cII for this input: cII_i = ∏(block_dims_i / stream_dims_i)
            cII_per_input[input_name] = input_if_copy.calculate_cII()
            
            # Find maximum weight constraint for this input
            max_weight_cycles = 1
            for weight_if in weight_interfaces:
                weight_name = weight_if.name
                weight_parallelism = wPar.get(weight_name, 1)
                
                # Update weight stream dimensions relative to this input
                # stream_dims_W = wPar * iPar * (block_dims_W / block_dims_I)
                weight_if_copy = self._copy_interface_with_weight_parallelism(
                    weight_if, weight_parallelism, input_parallelism, input_if
                )
                
                # Calculate weight cycles: ∏(tensor_dims_W / wPar)
                weight_cycles = self._calculate_weight_cycles(weight_if_copy, weight_parallelism)
                max_weight_cycles = max(max_weight_cycles, weight_cycles)
            
            # Calculate eII for this input: eII_i = cII_i * max_weight_cycles
            eII_per_input[input_name] = cII_per_input[input_name] * max_weight_cycles
        
        # Determine bottleneck and overall latency
        bottleneck_input_name = max(eII_per_input.keys(), key=lambda name: eII_per_input[name])
        bottleneck_input = self.interfaces[bottleneck_input_name]
        
        # L = eII_bottleneck * num_blocks_bottleneck  
        num_blocks = np.prod(bottleneck_input.get_num_blocks())
        L = eII_per_input[bottleneck_input_name] * num_blocks
        
        # Update output stream dimensions based on bottleneck
        bottleneck_parallelism = iPar.get(bottleneck_input_name, 1)
        self._update_output_stream_dimensions(output_interfaces, bottleneck_input, bottleneck_parallelism)
        
        bottleneck_analysis = {
            "bottleneck_input": bottleneck_input_name,
            "bottleneck_eII": eII_per_input[bottleneck_input_name],
            "bottleneck_cII": cII_per_input[bottleneck_input_name], 
            "bottleneck_num_blocks": num_blocks,
            "bottleneck_tensor_dims": bottleneck_input.tensor_dims,
            "bottleneck_block_dims": bottleneck_input.block_dims,
            "bottleneck_stream_dims": bottleneck_input.stream_dims,
            "total_cycles_breakdown": {
                "tensor_processing_cycles": eII_per_input[bottleneck_input_name],
                "total_tensor_blocks": num_blocks,
                "total_inference_cycles": L
            },
            "interface_counts": {
                "total_inputs": len(input_interfaces),
                "total_outputs": len(output_interfaces),
                "total_weights": len(weight_interfaces)
            }
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
        # Create a shallow copy and update stream_dims
        new_stream_dims = interface.stream_dims.copy()
        
        if interface.interface_type == InterfaceType.INPUT:
            new_stream_dims[0] = input_parallelism
        elif interface.interface_type == InterfaceType.WEIGHT and weight_parallelism is not None:
            new_stream_dims[0] = weight_parallelism
        
        # For simplicity, we'll modify the original interface's stream_dims
        # In a production implementation, we might want true copying
        original_stream_dims = interface.stream_dims.copy()
        interface.stream_dims = new_stream_dims
        
        return interface
    
    def _copy_interface_with_weight_parallelism(self, weight_if: DataflowInterface,
                                              weight_parallelism: int,
                                              input_parallelism: int,
                                              input_if: DataflowInterface) -> DataflowInterface:
        """Create copy of weight interface with computed stream dimensions"""
        # stream_dims_W = wPar * iPar * (block_dims_W / block_dims_I)
        if len(input_if.block_dims) > 0 and input_if.block_dims[0] != 0:
            scaling_factor = weight_if.block_dims[0] // input_if.block_dims[0] if weight_if.block_dims[0] >= input_if.block_dims[0] else 1
        else:
            scaling_factor = 1
            
        new_stream_dims = weight_if.stream_dims.copy()
        new_stream_dims[0] = weight_parallelism * input_parallelism * scaling_factor
        
        # Update the interface (temporary modification)
        original_stream_dims = weight_if.stream_dims.copy()
        weight_if.stream_dims = new_stream_dims
        
        return weight_if
    
    
    def _calculate_weight_cycles(self, weight_if: DataflowInterface, weight_parallelism: int) -> int:
        """Calculate weight loading cycles based on number of weight blocks to process"""
        weight_cycles = 1
        num_blocks = weight_if.get_num_blocks()
        for num_block in num_blocks:
            if weight_parallelism > 0:
                weight_cycles *= (num_block + weight_parallelism - 1) // weight_parallelism
        return max(weight_cycles, 1)
    
    def _update_output_stream_dimensions(self, output_interfaces: List[DataflowInterface],
                                       bottleneck_input: DataflowInterface,
                                       bottleneck_parallelism: int) -> None:
        """Update output stream dimensions based on bottleneck input"""
        for output_if in output_interfaces:
            if len(output_if.stream_dims) > 0 and len(bottleneck_input.block_dims) > 0:
                # stream_dims_O = iPar_bottleneck * (block_dims_O / block_dims_I_bottleneck)
                if bottleneck_input.block_dims[0] != 0:
                    scaling_factor = output_if.block_dims[0] // bottleneck_input.block_dims[0] if output_if.block_dims[0] >= bottleneck_input.block_dims[0] else 1
                else:
                    scaling_factor = 1
                output_if.stream_dims[0] = bottleneck_parallelism * scaling_factor
    
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
    
    def get_resource_requirements(self, parallelism_config: ParallelismConfiguration) -> Dict[str, Any]:
        """
        Estimate resource requirements for given parallelism configuration using ResourceAnalyzer
        """
        # Import ResourceAnalyzer here to avoid circular imports
        from .resource_analysis import ResourceAnalyzer
        
        # Create analyzer and get comprehensive requirements
        analyzer = ResourceAnalyzer()
        resource_req = analyzer.analyze_model(self, parallelism_config)
        
        # Calculate computation cycles using current parallelism
        intervals = self.calculate_initiation_intervals(
            parallelism_config.iPar, 
            parallelism_config.wPar
        )
        
        # Return comprehensive resource analysis
        return {
            "memory_bits": resource_req.memory_bits,
            "bandwidth_bits_per_cycle": resource_req.bandwidth_bits_per_cycle,
            "computation_cycles": intervals.L,
            "buffer_requirements": resource_req.buffer_requirements,
            "compute_units": resource_req.compute_units,
            "detailed_analysis": resource_req.get_summary(),
            "metadata": resource_req.metadata
        }
    
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
    
    def calculate_resource_efficiency(self, configurations: List[ParallelismConfiguration]) -> Dict[str, Any]:
        """
        Compare resource efficiency across multiple parallelism configurations
        
        Args:
            configurations: List of ParallelismConfiguration objects to compare
            
        Returns:
            Dict containing efficiency analysis and recommendations
        """
        # Import ResourceAnalyzer here to avoid circular imports
        from .resource_analysis import ResourceAnalyzer
        
        analyzer = ResourceAnalyzer()
        
        # Analyze each configuration
        config_results = []
        for config in configurations:
            requirements = analyzer.analyze_model(self, config)
            config_results.append((config, requirements))
        
        # Get comparison analysis
        comparison = analyzer.compare_configurations(config_results)
        
        # Add model-specific context
        comparison["model_summary"] = {
            "num_interfaces": len(self.interfaces),
            "input_interfaces": len(self.input_interfaces),
            "output_interfaces": len(self.output_interfaces), 
            "weight_interfaces": len(self.weight_interfaces),
            "total_configurations_analyzed": len(configurations)
        }
        
        return comparison
