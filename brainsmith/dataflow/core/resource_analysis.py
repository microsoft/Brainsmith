############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Resource Analysis Framework for Interface-Wise Dataflow Modeling
############################################################################

"""Resource analysis framework for dataflow modeling components.

This module provides resource requirement analysis for dataflow models and
interfaces, enabling memory footprint calculation, bandwidth analysis,
and performance estimation for hardware implementations.

Components:
- ResourceRequirements: Data structure for resource analysis results
- ResourceAnalyzer: Core analysis engine
- Integration with DataflowInterface and DataflowModel
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass
class ResourceRequirements:
    """Comprehensive resource requirement analysis.
    
    Contains detailed breakdown of memory, bandwidth, and compute requirements
    for dataflow interfaces and models.
    """
    memory_bits: int
    bandwidth_bits_per_cycle: int
    buffer_requirements: Dict[str, int] = field(default_factory=dict)
    compute_units: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get human-readable resource summary.
        
        Returns:
            Dict containing organized resource information
        """
        return {
            "memory": {
                "total_bits": self.memory_bits,
                "total_bytes": self.memory_bits // 8,
                "total_kb": self.memory_bits // (8 * 1024),
                "buffers": self.buffer_requirements
            },
            "bandwidth": {
                "bits_per_cycle": self.bandwidth_bits_per_cycle,
                "bytes_per_cycle": self.bandwidth_bits_per_cycle // 8,
                "gb_per_sec_at_100mhz": (self.bandwidth_bits_per_cycle * 100e6) / (8 * 1e9)
            },
            "compute": self.compute_units,
            "metadata": self.metadata
        }
    
    def __add__(self, other: 'ResourceRequirements') -> 'ResourceRequirements':
        """Add two resource requirements together."""
        combined_buffers = self.buffer_requirements.copy()
        for key, value in other.buffer_requirements.items():
            combined_buffers[key] = combined_buffers.get(key, 0) + value
            
        combined_compute = self.compute_units.copy()
        for key, value in other.compute_units.items():
            combined_compute[key] = combined_compute.get(key, 0) + value
            
        combined_metadata = self.metadata.copy()
        combined_metadata.update(other.metadata)
        
        return ResourceRequirements(
            memory_bits=self.memory_bits + other.memory_bits,
            bandwidth_bits_per_cycle=self.bandwidth_bits_per_cycle + other.bandwidth_bits_per_cycle,
            buffer_requirements=combined_buffers,
            compute_units=combined_compute,
            metadata=combined_metadata
        )


class ResourceAnalyzer:
    """Analyzes resource requirements for dataflow models and interfaces.
    
    Provides comprehensive analysis of memory, bandwidth, and compute
    requirements for hardware implementation planning.
    """
    
    def __init__(self, clock_frequency_hz: float = 100e6):
        """Initialize resource analyzer.
        
        Args:
            clock_frequency_hz: Target clock frequency for bandwidth calculations
        """
        self.clock_frequency_hz = clock_frequency_hz
        
    def analyze_interface(self, interface) -> ResourceRequirements:
        """Analyze resource requirements for single interface.
        
        Args:
            interface: DataflowInterface to analyze
            
        Returns:
            ResourceRequirements for the interface
        """
        # Import here to avoid circular imports
        from .dataflow_interface import DataflowInterface
        
        if not isinstance(interface, DataflowInterface):
            raise ValueError(f"Expected DataflowInterface, got {type(interface)}")
        
        # Calculate memory requirements
        total_elements = interface.calculate_total_elements()
        element_bits = interface.dtype.bitwidth
        memory_bits = total_elements * element_bits
        
        # Calculate bandwidth requirements (elements per cycle * bit width)
        elements_per_cycle = np.prod(interface.stream_dims)
        bandwidth_bits_per_cycle = elements_per_cycle * element_bits
        
        # Calculate buffer requirements
        buffer_requirements = self._calculate_buffer_requirements(interface)
        
        # Calculate compute requirements based on interface type
        compute_units = self._calculate_compute_requirements(interface)
        
        # Collect metadata
        metadata = {
            "interface_name": interface.name,
            "interface_type": interface.interface_type.value,
            "tensor_dims": interface.tensor_dims,
            "block_dims": interface.block_dims,
            "stream_dims": interface.stream_dims,
            "dtype": interface.dtype.finn_type,
            "total_elements": total_elements,
            "elements_per_cycle": elements_per_cycle,
            "num_blocks": interface.get_num_blocks(),
            "transfer_cycles": interface.get_transfer_cycles()
        }
        
        return ResourceRequirements(
            memory_bits=memory_bits,
            bandwidth_bits_per_cycle=bandwidth_bits_per_cycle,
            buffer_requirements=buffer_requirements,
            compute_units=compute_units,
            metadata=metadata
        )
    
    def analyze_model(self, model, parallelism_config) -> ResourceRequirements:
        """Analyze resource requirements for complete model.
        
        Args:
            model: DataflowModel to analyze
            parallelism_config: ParallelismConfiguration specifying iPar/wPar
            
        Returns:
            ResourceRequirements for the entire model
        """
        # Import here to avoid circular imports
        from .dataflow_model import DataflowModel
        
        total_requirements = ResourceRequirements(
            memory_bits=0,
            bandwidth_bits_per_cycle=0
        )
        
        # Analyze each interface
        interface_requirements = {}
        for interface in model.interfaces.values():
            # Apply parallelism configuration to interface
            interface_copy = self._apply_parallelism_to_interface(interface, parallelism_config)
            
            # Analyze the configured interface
            req = self.analyze_interface(interface_copy)
            interface_requirements[interface.name] = req
            
            # Add to total (bandwidth is cumulative, memory is cumulative)
            total_requirements += req
        
        # Add model-level metadata
        total_requirements.metadata.update({
            "model_interfaces": list(interface_requirements.keys()),
            "parallelism_config": {
                "iPar": getattr(parallelism_config, 'iPar', {}),
                "wPar": getattr(parallelism_config, 'wPar', {})
            },
            "interface_breakdown": {name: req.get_summary() 
                                  for name, req in interface_requirements.items()}
        })
        
        return total_requirements
    
    def compare_configurations(self, 
                             configs: List[Tuple[Any, ResourceRequirements]]) -> Dict[str, Any]:
        """Compare multiple parallelism configurations.
        
        Args:
            configs: List of (ParallelismConfiguration, ResourceRequirements) tuples
            
        Returns:
            Comparison analysis with recommendations
        """
        if not configs:
            return {"error": "No configurations provided"}
        
        analysis = {
            "configurations": [],
            "comparison": {},
            "recommendations": []
        }
        
        # Analyze each configuration
        for i, (config, requirements) in enumerate(configs):
            config_analysis = {
                "index": i,
                "config": {
                    "iPar": getattr(config, 'iPar', {}),
                    "wPar": getattr(config, 'wPar', {})
                },
                "resources": requirements.get_summary(),
                "efficiency_metrics": self._calculate_efficiency_metrics(requirements)
            }
            analysis["configurations"].append(config_analysis)
        
        # Find best configurations
        best_memory = min(configs, key=lambda x: x[1].memory_bits)
        best_bandwidth = min(configs, key=lambda x: x[1].bandwidth_bits_per_cycle)
        
        analysis["comparison"] = {
            "best_memory_efficiency": {
                "config_index": configs.index(best_memory),
                "memory_bits": best_memory[1].memory_bits
            },
            "best_bandwidth_efficiency": {
                "config_index": configs.index(best_bandwidth),
                "bandwidth_bits_per_cycle": best_bandwidth[1].bandwidth_bits_per_cycle
            }
        }
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(configs)
        
        return analysis
    
    def _calculate_buffer_requirements(self, interface) -> Dict[str, int]:
        """Calculate detailed buffer requirements for interface."""
        buffers = {}
        
        # Input/output buffers based on interface type
        if hasattr(interface, 'interface_type'):
            from .dataflow_interface import DataflowInterfaceType
            
            if interface.interface_type == DataflowInterfaceType.INPUT:
                # Input buffer needs to hold at least one block
                block_elements = np.prod(interface.block_dims)
                buffers["input_buffer_bits"] = block_elements * interface.dtype.bitwidth
                
                # Stream buffer for parallelism
                stream_elements = np.prod(interface.stream_dims)
                buffers["stream_buffer_bits"] = stream_elements * interface.dtype.bitwidth
                
            elif interface.interface_type == DataflowInterfaceType.WEIGHT:
                # Weight buffer typically needs full weight storage
                total_elements = interface.calculate_total_elements()
                buffers["weight_buffer_bits"] = total_elements * interface.dtype.bitwidth
                
            elif interface.interface_type == DataflowInterfaceType.OUTPUT:
                # Output buffer for accumulation and streaming
                block_elements = np.prod(interface.block_dims)
                buffers["output_buffer_bits"] = block_elements * interface.dtype.bitwidth
        
        return buffers
    
    def _calculate_compute_requirements(self, interface) -> Dict[str, int]:
        """Calculate compute unit requirements based on interface characteristics."""
        compute = {}
        
        if hasattr(interface, 'interface_type'):
            from .dataflow_interface import DataflowInterfaceType
            
            # Estimate compute units based on parallelism
            parallelism = np.prod(interface.stream_dims)
            
            if interface.interface_type == DataflowInterfaceType.INPUT:
                compute["processing_elements"] = parallelism
                
            elif interface.interface_type == DataflowInterfaceType.WEIGHT:
                # Weight processing typically needs MAC units
                compute["mac_units"] = parallelism
                
            elif interface.interface_type == DataflowInterfaceType.OUTPUT:
                compute["accumulator_units"] = parallelism
        
        return compute
    
    def _apply_parallelism_to_interface(self, interface, parallelism_config):
        """Apply parallelism configuration to interface (create copy)."""
        # Create a copy to avoid modifying original
        interface_copy = type(interface)(
            name=interface.name,
            interface_type=interface.interface_type,
            tensor_dims=interface.tensor_dims.copy(),
            block_dims=interface.block_dims.copy(),
            stream_dims=interface.stream_dims.copy(),
            dtype=interface.dtype
        )
        
        # Apply parallelism settings
        iPar = getattr(parallelism_config, 'iPar', {})
        wPar = getattr(parallelism_config, 'wPar', {})
        
        if interface.name in iPar:
            interface_copy.apply_parallelism(iPar=iPar[interface.name])
        elif interface.name in wPar:
            interface_copy.apply_parallelism(wPar=wPar[interface.name])
        
        return interface_copy
    
    def _calculate_efficiency_metrics(self, requirements: ResourceRequirements) -> Dict[str, float]:
        """Calculate efficiency metrics for resource requirements."""
        summary = requirements.get_summary()
        
        metrics = {}
        
        # Memory efficiency (bits per KB)
        if summary["memory"]["total_kb"] > 0:
            metrics["memory_efficiency"] = summary["memory"]["total_bits"] / summary["memory"]["total_kb"]
        
        # Bandwidth efficiency (GB/s per bit/cycle)
        if requirements.bandwidth_bits_per_cycle > 0:
            metrics["bandwidth_efficiency"] = summary["bandwidth"]["gb_per_sec_at_100mhz"] / requirements.bandwidth_bits_per_cycle
        
        # Compute efficiency (if compute units available)
        total_compute = sum(requirements.compute_units.values())
        if total_compute > 0:
            metrics["compute_density"] = requirements.memory_bits / total_compute
        
        return metrics
    
    def _generate_recommendations(self, configs: List[Tuple[Any, ResourceRequirements]]) -> List[str]:
        """Generate optimization recommendations based on configuration analysis."""
        recommendations = []
        
        if len(configs) < 2:
            recommendations.append("More configurations needed for meaningful comparison")
            return recommendations
        
        # Analyze memory usage patterns
        memory_values = [req.memory_bits for _, req in configs]
        memory_range = max(memory_values) - min(memory_values)
        
        if memory_range > min(memory_values) * 0.5:  # >50% variation
            recommendations.append("Significant memory variation detected - consider memory-optimized configuration")
        
        # Analyze bandwidth patterns  
        bandwidth_values = [req.bandwidth_bits_per_cycle for _, req in configs]
        bandwidth_range = max(bandwidth_values) - min(bandwidth_values)
        
        if bandwidth_range > min(bandwidth_values) * 0.3:  # >30% variation
            recommendations.append("Bandwidth requirements vary significantly - balance parallelism carefully")
        
        # Check for extreme configurations
        if max(memory_values) > 1024 * 1024 * 8:  # > 1MB
            recommendations.append("High memory usage detected - consider reducing tensor dimensions or increasing block chunking")
        
        if max(bandwidth_values) > 1024 * 8:  # > 1KB per cycle
            recommendations.append("High bandwidth requirements - verify AXI interface can handle the load")
        
        return recommendations