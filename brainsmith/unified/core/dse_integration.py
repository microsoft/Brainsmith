############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
DSE Integration for Unified Framework

Provides automatic optimization capabilities using the Design Space Exploration
framework.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from brainsmith.core.dataflow.dse import (
    DesignSpaceExplorer, DSEConstraints, DSEResult
)


class UnifiedDSEMixin:
    """
    Mixin for DSE capabilities in unified operators.
    
    This mixin should be used with UnifiedHWCustomOp to add automatic
    optimization capabilities.
    """
    
    def optimize_for_target(self, target_spec: Dict[str, Any]):
        """
        Optimize operator for target specification.
        
        Args:
            target_spec: Dictionary containing:
                - target_throughput: Desired throughput in MHz (optional)
                - max_resources: Resource constraints dict (optional)
                  - LUT: Maximum LUTs
                  - BRAM: Maximum BRAMs (18K)
                  - DSP: Maximum DSPs
                  - URAM: Maximum URAMs
                - optimization_objective: "throughput" | "latency" | "balanced" | "resources"
                - bandwidth_limit: Maximum bandwidth in GB/s (optional)
                - search_strategy: "exhaustive" | "greedy" | "random" (default: "exhaustive")
                - max_search_time: Maximum search time in seconds (optional)
        """
        # Ensure we have required attributes
        if not hasattr(self, 'graph') or not hasattr(self, 'config_space'):
            raise RuntimeError("DSE mixin requires graph and config_space attributes")
        
        # Set DSE constraints from target spec
        constraints = self._create_constraints_from_spec(target_spec)
        
        # Select search strategy
        strategy = target_spec.get('search_strategy', 'exhaustive')
        if target_spec.get('optimization_objective') == 'balanced':
            strategy = 'pareto'
        
        # Create explorer
        explorer = DesignSpaceExplorer(
            self.graph,
            constraints,
            strategy=strategy
        )
        
        # Set search time limit if specified
        if 'max_search_time' in target_spec:
            explorer.max_search_time = target_spec['max_search_time']
        
        # Explore design space
        results = explorer.explore()
        
        if not results:
            raise RuntimeError("No valid configurations found for target specification")
        
        # Select best configuration based on objective
        best = self._select_best_config(results, target_spec)
        
        # Apply best configuration
        self._current_config = best.config
        self._optimized = True
        
        # Update node attributes from configuration
        self._update_attributes_from_config()
        
        # Store optimization metadata
        self._optimization_metadata = {
            'target_spec': target_spec,
            'selected_config': best,
            'explored_configs': len(results),
            'pareto_configs': len(explorer.find_pareto_optimal(
                results,
                objectives=['throughput', 'resource_usage'],
                directions=['maximize', 'minimize']
            )) if len(results) > 1 else 1
        }
    
    def _create_constraints_from_spec(self, target_spec: Dict[str, Any]) -> DSEConstraints:
        """Create DSE constraints from target specification."""
        constraints = DSEConstraints()
        
        # Resource constraints
        if "max_resources" in target_spec:
            resources = target_spec["max_resources"]
            constraints.max_luts = resources.get("LUT", float('inf'))
            constraints.max_brams = resources.get("BRAM", float('inf'))
            constraints.max_dsps = resources.get("DSP", float('inf'))
            constraints.max_urams = resources.get("URAM", float('inf'))
        
        # Performance constraints
        if "target_throughput" in target_spec:
            constraints.min_throughput = target_spec["target_throughput"]
        
        if "bandwidth_limit" in target_spec:
            constraints.max_bandwidth = target_spec["bandwidth_limit"]
        
        # Add any kernel-specific constraints
        if hasattr(self, 'kernel_def') and self.kernel_def.metadata.get('dse_constraints'):
            kernel_constraints = self.kernel_def.metadata['dse_constraints']
            for key, value in kernel_constraints.items():
                setattr(constraints, key, value)
        
        return constraints
    
    def _select_best_config(self, results: List[DSEResult], 
                           target_spec: Dict[str, Any]) -> DSEResult:
        """Select best configuration based on optimization objective."""
        objective = target_spec.get('optimization_objective', 'balanced')
        
        if objective == 'throughput':
            # Maximize throughput
            return max(results, key=lambda r: r.performance.throughput)
        
        elif objective == 'latency':
            # Minimize latency
            return min(results, key=lambda r: r.performance.latency)
        
        elif objective == 'resources':
            # Minimize resource usage
            return min(results, key=lambda r: self._calculate_resource_score(r))
        
        else:  # balanced
            # Select from Pareto frontier
            return self._select_balanced_config(results, target_spec)
    
    def _calculate_resource_score(self, result: DSEResult) -> float:
        """Calculate weighted resource utilization score."""
        resources = result.performance.resource_usage
        
        # Weighted sum of resource usage
        # Weights can be adjusted based on resource scarcity
        score = (
            resources.get('LUT', 0) * 1.0 +
            resources.get('BRAM', 0) * 100.0 +  # BRAMs are more valuable
            resources.get('DSP', 0) * 50.0 +    # DSPs are valuable
            resources.get('URAM', 0) * 200.0    # URAMs are most valuable
        )
        
        return score
    
    def _select_balanced_config(self, results: List[DSEResult],
                               target_spec: Dict[str, Any]) -> DSEResult:
        """Select balanced configuration from Pareto frontier."""
        # Create a temporary explorer to use its find_pareto_optimal method
        temp_explorer = DesignSpaceExplorer(self.graph, DSEConstraints())
        
        # Find Pareto optimal configurations
        pareto_configs = temp_explorer.find_pareto_optimal(
            results,
            objectives=['throughput', 'resource_usage'],
            directions=['maximize', 'minimize']
        )
        
        if not pareto_configs:
            # Fallback to first result if Pareto computation fails
            return results[0]
        
        # If target throughput specified, find closest match on Pareto frontier
        if 'target_throughput' in target_spec:
            target_tp = target_spec['target_throughput']
            return min(pareto_configs, 
                      key=lambda r: abs(r.performance.throughput - target_tp))
        
        # Otherwise, select middle point from Pareto frontier
        # This gives a balanced trade-off
        return pareto_configs[len(pareto_configs) // 2]
    
    def _update_attributes_from_config(self):
        """Update node attributes from current configuration."""
        if not self._current_config:
            return
        
        # Get kernel configuration
        kernel_name = self.kernel.name if hasattr(self, 'kernel') else list(self._current_config.kernel_configs.keys())[0]
        kernel_config = self._current_config.kernel_configs[kernel_name]
        
        # Update parallelism attributes
        for intf_name, parallelism in kernel_config.interface_parallelism.items():
            attr_name = f"{intf_name}_parallelism"
            if hasattr(self, 'set_nodeattr'):
                self.set_nodeattr(attr_name, int(parallelism))
        
        # Update stream shapes if needed
        for intf_name, stream_shape in kernel_config.stream_shapes.items():
            # Stream shapes are computed from parallelism, so no need to store separately
            pass
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get detailed optimization report.
        
        Returns dictionary with optimization details including:
        - Target specification
        - Selected configuration
        - Performance metrics
        - Resource utilization
        - Pareto frontier information
        """
        if not hasattr(self, '_optimization_metadata'):
            return {
                'optimized': False,
                'message': 'Operator has not been optimized yet'
            }
        
        metadata = self._optimization_metadata
        selected = metadata['selected_config']
        
        return {
            'optimized': True,
            'target_specification': metadata['target_spec'],
            'explored_configurations': metadata['explored_configs'],
            'pareto_optimal_configs': metadata['pareto_configs'],
            'selected_configuration': {
                'parallelism': selected.config.kernel_configs[self.kernel.name].interface_parallelism,
                'performance': {
                    'throughput': selected.performance.throughput,
                    'latency': selected.performance.latency,
                    'initiation_interval': selected.performance.initiation_interval
                },
                'resources': selected.performance.resource_usage,
                'is_schedulable': selected.performance.is_schedulable
            }
        }
    
    def suggest_optimization_targets(self) -> List[Dict[str, Any]]:
        """
        Suggest optimization targets based on current configuration.
        
        Returns list of suggested target specifications that might
        yield better results.
        """
        suggestions = []
        
        # Get current performance if available
        if hasattr(self, 'get_performance_metrics'):
            current_perf = self.get_performance_metrics()
            current_tp = current_perf.get('throughput', 0)
            
            # Suggest higher throughput targets
            suggestions.append({
                'name': 'Higher Throughput',
                'target_spec': {
                    'target_throughput': current_tp * 1.5,
                    'optimization_objective': 'throughput'
                },
                'description': f'Target 50% higher throughput ({current_tp * 1.5:.1f} MHz)'
            })
            
            # Suggest balanced optimization
            suggestions.append({
                'name': 'Balanced Performance',
                'target_spec': {
                    'optimization_objective': 'balanced',
                    'search_strategy': 'pareto'
                },
                'description': 'Find optimal trade-off between performance and resources'
            })
            
            # Suggest resource minimization
            suggestions.append({
                'name': 'Minimize Resources',
                'target_spec': {
                    'target_throughput': current_tp * 0.8,
                    'optimization_objective': 'resources'
                },
                'description': f'Maintain 80% throughput ({current_tp * 0.8:.1f} MHz) with minimal resources'
            })
        
        return suggestions