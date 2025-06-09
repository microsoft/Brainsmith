"""
FINN Kernel Selection Algorithm
Intelligent kernel selection for optimal FINN configurations.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import itertools

from .registry import FINNKernelRegistry, SearchCriteria
from .database import FINNKernelInfo, OperatorType, BackendType
from .analysis import OperatorRequirement, TopologyAnalysis

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objectives for kernel selection"""
    THROUGHPUT = "throughput"
    LATENCY = "latency" 
    POWER = "power"
    AREA = "area"
    BALANCED = "balanced"

@dataclass
class PerformanceTargets:
    """Performance targets for optimization"""
    throughput: Optional[float] = None  # ops/sec
    latency: Optional[int] = None       # clock cycles
    power: Optional[float] = None       # watts
    area: Optional[int] = None          # LUTs
    frequency: float = 100e6            # clock frequency in Hz
    
    # Relative weights for multi-objective optimization
    throughput_weight: float = 0.4
    latency_weight: float = 0.3
    power_weight: float = 0.2
    area_weight: float = 0.1

@dataclass
class ResourceConstraints:
    """Resource constraints for FPGA implementation"""
    max_luts: Optional[int] = None
    max_ffs: Optional[int] = None
    max_dsps: Optional[int] = None
    max_brams: Optional[int] = None
    max_urams: Optional[int] = None
    
    # Utilization limits (0.0 to 1.0)
    lut_utilization_limit: float = 0.8
    dsp_utilization_limit: float = 0.9
    bram_utilization_limit: float = 0.8
    
    # Platform-specific constraints
    platform_family: str = "zynq"
    device_part: Optional[str] = None

@dataclass
class KernelParameterConfig:
    """Configuration parameters for a kernel"""
    pe_parallelism: int
    simd_width: int
    folding_factors: Dict[str, int]
    memory_mode: str
    ram_style: str = "auto"
    pipeline_depth: int = 1
    
    # Optimization-specific parameters
    enable_dsp_optimization: bool = True
    enable_memory_optimization: bool = True
    custom_directives: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ParameterConfiguration:
    """Complete parameter configuration for all kernels"""
    kernel_configs: Dict[str, KernelParameterConfig] = field(default_factory=dict)
    global_settings: Dict[str, Any] = field(default_factory=dict)
    optimization_directives: Dict[str, Any] = field(default_factory=dict)
    
    def add_kernel_config(self, kernel_name: str, config: KernelParameterConfig) -> None:
        """Add configuration for a specific kernel"""
        self.kernel_configs[kernel_name] = config
    
    def get_kernel_config(self, kernel_name: str) -> Optional[KernelParameterConfig]:
        """Get configuration for a specific kernel"""
        return self.kernel_configs.get(kernel_name)

@dataclass
class KernelSelection:
    """Selection of a kernel with its configuration"""
    kernel: FINNKernelInfo
    parameters: KernelParameterConfig
    estimated_performance: Dict[str, float]
    estimated_resources: Dict[str, int]
    selection_score: float
    selection_rationale: str

@dataclass
class SelectionPlan:
    """Complete kernel selection plan for a model"""
    selections: Dict[str, KernelSelection] = field(default_factory=dict)
    global_configuration: ParameterConfiguration = field(default_factory=ParameterConfiguration)
    estimated_total_performance: Dict[str, float] = field(default_factory=dict)
    estimated_total_resources: Dict[str, int] = field(default_factory=dict)
    optimization_score: float = 0.0
    
    def add_kernel_selection(self, layer_id: str, kernel: FINNKernelInfo, 
                           parameters: KernelParameterConfig) -> None:
        """Add kernel selection for a layer"""
        # Estimate performance and resources
        platform = {'clock_frequency': 100e6}
        param_dict = {
            'pe': parameters.pe_parallelism,
            'simd': parameters.simd_width
        }
        
        estimated_perf = kernel.estimate_performance(param_dict, platform)
        estimated_res = kernel.resource_requirements.estimate_resources(param_dict)
        
        selection = KernelSelection(
            kernel=kernel,
            parameters=parameters,
            estimated_performance=estimated_perf,
            estimated_resources=estimated_res.to_dict(),
            selection_score=0.0,  # Will be computed later
            selection_rationale=f"Selected for layer {layer_id}"
        )
        
        self.selections[layer_id] = selection
        self._update_totals()
    
    def _update_totals(self) -> None:
        """Update total performance and resource estimates"""
        self.estimated_total_resources = {
            'lut': 0, 'ff': 0, 'dsp': 0, 'bram': 0, 'uram': 0
        }
        
        # Sum resources across all selections
        for selection in self.selections.values():
            for resource, count in selection.estimated_resources.items():
                if resource in self.estimated_total_resources:
                    self.estimated_total_resources[resource] += count
        
        # Estimate total performance (simplified)
        total_throughput = sum(
            sel.estimated_performance.get('throughput', 0)
            for sel in self.selections.values()
        )
        
        self.estimated_total_performance = {
            'throughput': total_throughput,
            'latency': max(sel.estimated_performance.get('latency', 0) 
                          for sel in self.selections.values()) if self.selections else 0
        }

class PerformanceOptimizer:
    """Optimizes parameters for performance targets"""
    
    def __init__(self):
        self.optimization_strategies = {
            OptimizationObjective.THROUGHPUT: self._optimize_for_throughput,
            OptimizationObjective.LATENCY: self._optimize_for_latency,
            OptimizationObjective.POWER: self._optimize_for_power,
            OptimizationObjective.AREA: self._optimize_for_area,
            OptimizationObjective.BALANCED: self._optimize_balanced
        }
    
    def optimize_parameters(self, kernel: FINNKernelInfo, 
                          requirement: OperatorRequirement,
                          targets: PerformanceTargets,
                          constraints: ResourceConstraints,
                          objective: OptimizationObjective = OptimizationObjective.BALANCED) -> KernelParameterConfig:
        """Optimize kernel parameters for given objectives"""
        
        optimizer = self.optimization_strategies.get(objective, self._optimize_balanced)
        return optimizer(kernel, requirement, targets, constraints)
    
    def _optimize_for_throughput(self, kernel: FINNKernelInfo, requirement: OperatorRequirement,
                                targets: PerformanceTargets, constraints: ResourceConstraints) -> KernelParameterConfig:
        """Optimize parameters for maximum throughput"""
        
        best_config = None
        best_throughput = 0
        
        # Search parameter space
        pe_range = range(requirement.pe_requirements[0], 
                        min(requirement.pe_requirements[1], 64) + 1)
        simd_range = range(requirement.simd_requirements[0],
                          min(requirement.simd_requirements[1], 32) + 1)
        
        for pe, simd in itertools.product(pe_range, simd_range):
            # Check resource constraints
            if not self._meets_resource_constraints({'pe': pe, 'simd': simd}, 
                                                   kernel, constraints):
                continue
            
            # Estimate throughput
            platform = {'clock_frequency': targets.frequency}
            perf = kernel.performance_model.estimate_throughput({'pe': pe, 'simd': simd}, platform)
            
            if perf > best_throughput:
                best_throughput = perf
                best_config = KernelParameterConfig(
                    pe_parallelism=pe,
                    simd_width=simd,
                    folding_factors=self._compute_optimal_folding(kernel, requirement, pe, simd),
                    memory_mode=self._select_memory_mode(kernel, requirement, 'throughput'),
                    pipeline_depth=self._compute_pipeline_depth(pe, simd, 'throughput')
                )
        
        return best_config or self._get_default_config(kernel, requirement)
    
    def _optimize_for_latency(self, kernel: FINNKernelInfo, requirement: OperatorRequirement,
                             targets: PerformanceTargets, constraints: ResourceConstraints) -> KernelParameterConfig:
        """Optimize parameters for minimum latency"""
        
        best_config = None
        best_latency = float('inf')
        
        # For latency optimization, prefer higher parallelism
        pe_range = range(requirement.pe_requirements[1], 
                        requirement.pe_requirements[0] - 1, -1)  # Descending
        simd_range = range(requirement.simd_requirements[1],
                          requirement.simd_requirements[0] - 1, -1)  # Descending
        
        for pe, simd in itertools.product(pe_range, simd_range):
            if not self._meets_resource_constraints({'pe': pe, 'simd': simd}, 
                                                   kernel, constraints):
                continue
            
            platform = {'clock_frequency': targets.frequency}
            latency = kernel.performance_model.estimate_latency({'pe': pe, 'simd': simd}, platform)
            
            if latency < best_latency:
                best_latency = latency
                best_config = KernelParameterConfig(
                    pe_parallelism=pe,
                    simd_width=simd,
                    folding_factors=self._compute_optimal_folding(kernel, requirement, pe, simd),
                    memory_mode=self._select_memory_mode(kernel, requirement, 'latency'),
                    pipeline_depth=self._compute_pipeline_depth(pe, simd, 'latency')
                )
        
        return best_config or self._get_default_config(kernel, requirement)
    
    def _optimize_for_power(self, kernel: FINNKernelInfo, requirement: OperatorRequirement,
                           targets: PerformanceTargets, constraints: ResourceConstraints) -> KernelParameterConfig:
        """Optimize parameters for minimum power consumption"""
        
        # For power optimization, prefer lower parallelism with higher efficiency
        pe = min(requirement.pe_requirements[1] // 2, 16)  # Moderate parallelism
        simd = min(requirement.simd_requirements[1] // 2, 8)
        
        return KernelParameterConfig(
            pe_parallelism=max(1, pe),
            simd_width=max(1, simd),
            folding_factors=self._compute_optimal_folding(kernel, requirement, pe, simd),
            memory_mode='internal',  # Internal memory for lower power
            pipeline_depth=self._compute_pipeline_depth(pe, simd, 'power'),
            enable_dsp_optimization=False  # DSPs consume more power
        )
    
    def _optimize_for_area(self, kernel: FINNKernelInfo, requirement: OperatorRequirement,
                          targets: PerformanceTargets, constraints: ResourceConstraints) -> KernelParameterConfig:
        """Optimize parameters for minimum area (LUT usage)"""
        
        # For area optimization, use minimum parallelism
        pe = requirement.pe_requirements[0]
        simd = requirement.simd_requirements[0]
        
        return KernelParameterConfig(
            pe_parallelism=pe,
            simd_width=simd,
            folding_factors=self._compute_optimal_folding(kernel, requirement, pe, simd),
            memory_mode='external',  # External memory to save on-chip resources
            pipeline_depth=1,  # Minimum pipeline depth
            enable_memory_optimization=True
        )
    
    def _optimize_balanced(self, kernel: FINNKernelInfo, requirement: OperatorRequirement,
                          targets: PerformanceTargets, constraints: ResourceConstraints) -> KernelParameterConfig:
        """Optimize parameters for balanced performance"""
        
        best_config = None
        best_score = 0
        
        # Sample parameter space
        pe_samples = self._sample_range(requirement.pe_requirements[0], 
                                       requirement.pe_requirements[1], 8)
        simd_samples = self._sample_range(requirement.simd_requirements[0],
                                         requirement.simd_requirements[1], 6)
        
        for pe, simd in itertools.product(pe_samples, simd_samples):
            if not self._meets_resource_constraints({'pe': pe, 'simd': simd}, 
                                                   kernel, constraints):
                continue
            
            # Compute balanced score
            score = self._compute_balanced_score(kernel, {'pe': pe, 'simd': simd}, 
                                               targets, constraints)
            
            if score > best_score:
                best_score = score
                best_config = KernelParameterConfig(
                    pe_parallelism=pe,
                    simd_width=simd,
                    folding_factors=self._compute_optimal_folding(kernel, requirement, pe, simd),
                    memory_mode=self._select_memory_mode(kernel, requirement, 'balanced'),
                    pipeline_depth=self._compute_pipeline_depth(pe, simd, 'balanced')
                )
        
        return best_config or self._get_default_config(kernel, requirement)
    
    def _meets_resource_constraints(self, parameters: Dict[str, int], 
                                   kernel: FINNKernelInfo, 
                                   constraints: ResourceConstraints) -> bool:
        """Check if parameters meet resource constraints"""
        
        estimated_resources = kernel.resource_requirements.estimate_resources(parameters)
        
        if constraints.max_luts and estimated_resources.lut_count > constraints.max_luts:
            return False
        
        if constraints.max_dsps and estimated_resources.dsp_count > constraints.max_dsps:
            return False
        
        if constraints.max_brams and estimated_resources.bram_count > constraints.max_brams:
            return False
        
        return True
    
    def _compute_balanced_score(self, kernel: FINNKernelInfo, parameters: Dict[str, int],
                               targets: PerformanceTargets, constraints: ResourceConstraints) -> float:
        """Compute balanced optimization score"""
        
        platform = {'clock_frequency': targets.frequency}
        
        # Get performance estimates
        throughput = kernel.performance_model.estimate_throughput(parameters, platform)
        latency = kernel.performance_model.estimate_latency(parameters, platform)
        
        # Get resource estimates
        resources = kernel.resource_requirements.estimate_resources(parameters)
        
        # Normalize and weight scores
        throughput_score = min(1.0, throughput / 10000) * targets.throughput_weight
        latency_score = max(0.0, 1.0 - latency / 1000) * targets.latency_weight
        area_score = max(0.0, 1.0 - resources.lut_count / 50000) * targets.area_weight
        
        # Assume power scales with resource usage
        power_score = area_score * targets.power_weight
        
        return throughput_score + latency_score + area_score + power_score
    
    def _compute_optimal_folding(self, kernel: FINNKernelInfo, requirement: OperatorRequirement,
                                pe: int, simd: int) -> Dict[str, int]:
        """Compute optimal folding factors"""
        
        folding = {}
        
        if requirement.operator_type == "Convolution":
            # Spatial folding for convolution
            spatial_size = requirement.input_shape.height * requirement.input_shape.width
            folding['spatial_fold'] = min(spatial_size, pe * 2)
            
        elif requirement.operator_type == "MatMul":
            # Weight folding for matrix multiplication
            input_size = requirement.input_shape.width
            output_size = requirement.output_shape.width
            
            folding['input_fold'] = min(input_size, simd * 2)
            folding['output_fold'] = min(output_size, pe * 2)
        
        return folding
    
    def _select_memory_mode(self, kernel: FINNKernelInfo, requirement: OperatorRequirement,
                           optimization: str) -> str:
        """Select optimal memory mode"""
        
        if optimization in ['throughput', 'latency']:
            return 'internal'  # Faster access
        elif optimization == 'area':
            return 'external'  # Save on-chip memory
        else:
            # Balanced - choose based on data size
            data_size = requirement.input_shape.total_elements + requirement.output_shape.total_elements
            return 'internal' if data_size < 10000 else 'external'
    
    def _compute_pipeline_depth(self, pe: int, simd: int, optimization: str) -> int:
        """Compute optimal pipeline depth"""
        
        if optimization == 'throughput':
            return min(16, pe // 2 + simd // 4)  # Deeper pipeline for throughput
        elif optimization == 'latency':
            return max(1, pe // 8)  # Shallow pipeline for latency
        elif optimization == 'area':
            return 1  # Minimum pipeline for area
        else:
            return min(8, pe // 4 + 1)  # Balanced pipeline depth
    
    def _sample_range(self, min_val: int, max_val: int, num_samples: int) -> List[int]:
        """Sample values from range"""
        if max_val <= min_val:
            return [min_val]
        
        if max_val - min_val + 1 <= num_samples:
            return list(range(min_val, max_val + 1))
        
        step = (max_val - min_val) / (num_samples - 1)
        return [min_val + int(i * step) for i in range(num_samples)]
    
    def _get_default_config(self, kernel: FINNKernelInfo, requirement: OperatorRequirement) -> KernelParameterConfig:
        """Get default configuration when optimization fails"""
        
        pe_mid = (requirement.pe_requirements[0] + requirement.pe_requirements[1]) // 2
        simd_mid = (requirement.simd_requirements[0] + requirement.simd_requirements[1]) // 2
        
        return KernelParameterConfig(
            pe_parallelism=pe_mid,
            simd_width=simd_mid,
            folding_factors={},
            memory_mode='internal',
            pipeline_depth=2
        )

class FINNKernelSelector:
    """
    Intelligent kernel selection for optimal FINN configurations
    
    Main class that orchestrates kernel selection and parameter optimization
    for complete FINN model implementations.
    """
    
    def __init__(self, kernel_registry: FINNKernelRegistry):
        self.registry = kernel_registry
        self.performance_optimizer = PerformanceOptimizer()
        
        # Selection preferences
        self.backend_preferences = {
            'performance': BackendType.RTL,
            'flexibility': BackendType.HLS,
            'development': BackendType.PYTHON
        }
        
        logger.info("FINN Kernel Selector initialized")
    
    def select_optimal_kernels(self, 
                              requirements: List[OperatorRequirement],
                              targets: PerformanceTargets,
                              constraints: ResourceConstraints,
                              selection_strategy: str = 'balanced') -> SelectionPlan:
        """
        Select optimal FINN kernels for given requirements
        
        Args:
            requirements: Per-layer operator requirements
            targets: Performance targets (throughput, latency, power)
            constraints: Resource constraints (LUTs, DSPs, BRAM)
            selection_strategy: Selection strategy ('performance', 'area', 'balanced')
            
        Returns:
            SelectionPlan: Optimal kernel selection with configurations
        """
        logger.info(f"Selecting kernels for {len(requirements)} operators")
        
        plan = SelectionPlan()
        
        # Select kernel for each requirement
        for req in requirements:
            logger.debug(f"Selecting kernel for layer {req.layer_id} ({req.operator_type})")
            
            # Find candidate kernels
            candidates = self._find_candidate_kernels(req, constraints)
            
            if not candidates:
                logger.warning(f"No candidate kernels found for {req.layer_id}")
                continue
            
            # Evaluate and select best kernel
            best_kernel, best_params = self._select_best_kernel(
                candidates, req, targets, constraints, selection_strategy
            )
            
            if best_kernel:
                plan.add_kernel_selection(req.layer_id, best_kernel, best_params)
                logger.debug(f"Selected {best_kernel.name} for {req.layer_id}")
            else:
                logger.warning(f"Failed to select kernel for {req.layer_id}")
        
        # Global optimization
        plan = self._optimize_global_configuration(plan, targets, constraints)
        
        # Compute final scores
        plan.optimization_score = self._compute_plan_score(plan, targets, constraints)
        
        logger.info(f"Kernel selection complete: {len(plan.selections)} kernels selected")
        return plan
    
    def _find_candidate_kernels(self, requirement: OperatorRequirement,
                               constraints: ResourceConstraints) -> List[FINNKernelInfo]:
        """Find candidate kernels for operator requirement"""
        
        # Map requirement operator type to FINN operator type
        operator_mapping = {
            "Convolution": OperatorType.CONVOLUTION,
            "MatMul": OperatorType.MATMUL,
            "Thresholding": OperatorType.THRESHOLDING,
            "LayerNorm": OperatorType.LAYERNORM,
            "Pool": OperatorType.POOL,
            "ElementWise": OperatorType.ELEMENTWISE,
            "Reshape": OperatorType.RESHAPE,
            "Concat": OperatorType.CONCAT
        }
        
        operator_type = operator_mapping.get(requirement.operator_type)
        if not operator_type:
            logger.warning(f"Unsupported operator type: {requirement.operator_type}")
            return []
        
        # Create search criteria
        criteria = SearchCriteria(
            operator_type=operator_type,
            min_pe=requirement.pe_requirements[0],
            max_pe=requirement.pe_requirements[1],
            min_simd=requirement.simd_requirements[0],
            max_simd=requirement.simd_requirements[1],
            supported_datatype=requirement.data_type.value,
            max_lut_usage=constraints.max_luts,
            max_dsp_usage=constraints.max_dsps,
            performance_requirements=requirement.performance_requirements
        )
        
        # Search for matching kernels
        candidates = self.registry.search_kernels(criteria)
        
        logger.debug(f"Found {len(candidates)} candidate kernels for {requirement.operator_type}")
        return candidates
    
    def _select_best_kernel(self, candidates: List[FINNKernelInfo],
                           requirement: OperatorRequirement,
                           targets: PerformanceTargets,
                           constraints: ResourceConstraints,
                           strategy: str) -> Tuple[Optional[FINNKernelInfo], Optional[KernelParameterConfig]]:
        """Select best kernel from candidates"""
        
        best_kernel = None
        best_params = None
        best_score = 0
        
        # Determine optimization objective
        objective_mapping = {
            'performance': OptimizationObjective.THROUGHPUT,
            'latency': OptimizationObjective.LATENCY,
            'area': OptimizationObjective.AREA,
            'power': OptimizationObjective.POWER,
            'balanced': OptimizationObjective.BALANCED
        }
        
        objective = objective_mapping.get(strategy, OptimizationObjective.BALANCED)
        
        for kernel in candidates:
            # Optimize parameters for this kernel
            params = self.performance_optimizer.optimize_parameters(
                kernel, requirement, targets, constraints, objective
            )
            
            # Evaluate kernel with optimized parameters
            score = self._evaluate_kernel(kernel, params, requirement, targets, constraints)
            
            if score > best_score:
                best_score = score
                best_kernel = kernel
                best_params = params
        
        return best_kernel, best_params
    
    def _evaluate_kernel(self, kernel: FINNKernelInfo, params: KernelParameterConfig,
                        requirement: OperatorRequirement, targets: PerformanceTargets,
                        constraints: ResourceConstraints) -> float:
        """Evaluate kernel selection quality"""
        
        platform = {'clock_frequency': targets.frequency}
        param_dict = {'pe': params.pe_parallelism, 'simd': params.simd_width}
        
        # Get performance estimates
        estimated_perf = kernel.estimate_performance(param_dict, platform)
        
        # Get resource estimates
        estimated_res = kernel.resource_requirements.estimate_resources(param_dict)
        
        score = 0.0
        
        # Performance score
        if 'throughput' in estimated_perf:
            throughput_score = min(1.0, estimated_perf['throughput'] / 10000)
            score += throughput_score * targets.throughput_weight
        
        if 'latency' in estimated_perf:
            latency_score = max(0.0, 1.0 - estimated_perf['latency'] / 1000)
            score += latency_score * targets.latency_weight
        
        # Resource efficiency score
        lut_efficiency = max(0.0, 1.0 - estimated_res.lut_count / 50000)
        score += lut_efficiency * targets.area_weight
        
        # Quality bonuses
        score += kernel.reliability_score * 0.1
        score += kernel.test_coverage * 0.05
        
        # Verification status bonus
        if kernel.verification_status == "verified":
            score += 0.1
        
        return score
    
    def _optimize_global_configuration(self, plan: SelectionPlan, 
                                     targets: PerformanceTargets,
                                     constraints: ResourceConstraints) -> SelectionPlan:
        """Optimize global configuration across all selected kernels"""
        
        # Check global resource constraints
        if self._violates_global_constraints(plan, constraints):
            plan = self._resolve_resource_conflicts(plan, constraints)
        
        # Optimize inter-kernel communication
        plan = self._optimize_dataflow(plan)
        
        # Set global optimization directives
        plan.global_configuration.optimization_directives = {
            'global_clock_gating': True,
            'resource_sharing': True,
            'pipeline_balancing': True,
            'memory_optimization': True
        }
        
        return plan
    
    def _violates_global_constraints(self, plan: SelectionPlan, 
                                   constraints: ResourceConstraints) -> bool:
        """Check if plan violates global resource constraints"""
        
        total_resources = plan.estimated_total_resources
        
        if constraints.max_luts and total_resources['lut'] > constraints.max_luts:
            return True
        
        if constraints.max_dsps and total_resources['dsp'] > constraints.max_dsps:
            return True
        
        if constraints.max_brams and total_resources['bram'] > constraints.max_brams:
            return True
        
        return False
    
    def _resolve_resource_conflicts(self, plan: SelectionPlan,
                                   constraints: ResourceConstraints) -> SelectionPlan:
        """Resolve resource constraint violations"""
        
        # Simple strategy: reduce parallelism of highest resource consumers
        sorted_selections = sorted(
            plan.selections.items(),
            key=lambda x: sum(x[1].estimated_resources.values()),
            reverse=True
        )
        
        for layer_id, selection in sorted_selections:
            if not self._violates_global_constraints(plan, constraints):
                break
            
            # Reduce PE parallelism
            if selection.parameters.pe_parallelism > 1:
                selection.parameters.pe_parallelism //= 2
                
                # Recompute estimates
                param_dict = {
                    'pe': selection.parameters.pe_parallelism,
                    'simd': selection.parameters.simd_width
                }
                
                platform = {'clock_frequency': 100e6}
                selection.estimated_performance = selection.kernel.estimate_performance(param_dict, platform)
                selection.estimated_resources = selection.kernel.resource_requirements.estimate_resources(param_dict).to_dict()
        
        # Update plan totals
        plan._update_totals()
        
        return plan
    
    def _optimize_dataflow(self, plan: SelectionPlan) -> SelectionPlan:
        """Optimize dataflow between kernels"""
        
        # Simple dataflow optimization
        # In a real implementation, this would analyze data dependencies
        # and optimize memory transfers and synchronization
        
        plan.global_configuration.global_settings['dataflow_optimization'] = True
        plan.global_configuration.global_settings['memory_coalescing'] = True
        plan.global_configuration.global_settings['pipeline_insertion'] = True
        
        return plan
    
    def _compute_plan_score(self, plan: SelectionPlan, targets: PerformanceTargets,
                           constraints: ResourceConstraints) -> float:
        """Compute overall plan quality score"""
        
        if not plan.selections:
            return 0.0
        
        # Average individual selection scores
        avg_selection_score = np.mean([sel.selection_score for sel in plan.selections.values()])
        
        # Resource utilization efficiency
        total_resources = plan.estimated_total_resources
        max_resources = constraints.max_luts or 100000
        
        resource_efficiency = 1.0 - (total_resources.get('lut', 0) / max_resources)
        resource_efficiency = max(0.0, min(1.0, resource_efficiency))
        
        # Performance achievement
        total_perf = plan.estimated_total_performance
        target_throughput = targets.throughput or 1000
        
        perf_achievement = min(1.0, total_perf.get('throughput', 0) / target_throughput)
        
        # Combined score
        score = (0.5 * avg_selection_score + 
                0.3 * resource_efficiency + 
                0.2 * perf_achievement)
        
        return score

    def optimize_kernel_parameters(self, 
                                  kernels: List[FINNKernelInfo],
                                  targets: PerformanceTargets) -> ParameterConfiguration:
        """Optimize PE, SIMD, and other parameters for target performance"""
        
        config = ParameterConfiguration()
        
        for kernel in kernels:
            # Create dummy requirement for parameter optimization
            dummy_req = OperatorRequirement(
                layer_id=kernel.name,
                operator_type=kernel.operator_type.value,
                input_shape=None,  # Would need actual shape
                output_shape=None,  # Would need actual shape
                parameters={},
                constraints={},
                performance_requirements={'throughput': targets.throughput or 1000},
                data_type=None,  # Would need actual data type
                pe_requirements=kernel.parameterization.pe_range,
                simd_requirements=kernel.parameterization.simd_range,
                memory_requirements={},
                folding_constraints={}
            )
            
            # Default constraints
            default_constraints = ResourceConstraints(
                max_luts=100000,
                max_dsps=2000,
                max_brams=500
            )
            
            # Optimize parameters
            optimal_params = self.performance_optimizer.optimize_parameters(
                kernel, dummy_req, targets, default_constraints, OptimizationObjective.BALANCED
            )
            
            config.add_kernel_config(kernel.name, optimal_params)
        
        return config