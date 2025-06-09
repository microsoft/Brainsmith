"""
Hardware Optimization Manager for FINN Integration Engine.

Handles configuration of FINN hardware optimization including:
- Optimization strategy selection
- Performance targets configuration
- Resource and timing constraints
- Power optimization settings
"""

import logging
from typing import Dict, Any, List

from .types import HwOptimizationConfig

logger = logging.getLogger(__name__)

class HwOptimizationManager:
    """Manager for FINN hardware optimization configuration"""
    
    def __init__(self):
        self.optimization_strategies = self._load_optimization_strategies()
        self.constraint_profiles = self._load_constraint_profiles()
        self.optimization_techniques = self._load_optimization_techniques()
    
    def configure(self,
                 optimization_strategy: str = "balanced",
                 performance_targets: Dict[str, float] = None,
                 power_constraints: Dict[str, float] = None) -> HwOptimizationConfig:
        """Configure hardware optimization for FINN"""
        
        if performance_targets is None:
            performance_targets = {}
        if power_constraints is None:
            power_constraints = {}
        
        # Validate and adjust strategy
        validated_strategy = self._validate_strategy(optimization_strategy)
        
        # Configure performance targets
        configured_targets = self._configure_performance_targets(
            validated_strategy, performance_targets
        )
        
        # Configure power constraints
        configured_power = self._configure_power_constraints(
            validated_strategy, power_constraints
        )
        
        # Configure timing constraints
        timing_constraints = self._configure_timing_constraints(
            validated_strategy, configured_targets
        )
        
        # Configure resource constraints
        resource_constraints = self._configure_resource_constraints(
            validated_strategy, configured_targets
        )
        
        return HwOptimizationConfig(
            optimization_strategy=validated_strategy,
            performance_targets=configured_targets,
            power_constraints=configured_power,
            timing_constraints=timing_constraints,
            resource_constraints=resource_constraints
        )
    
    def _load_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load available optimization strategies"""
        return {
            'throughput': {
                'focus': 'maximum_throughput',
                'priority_order': ['throughput', 'latency', 'area', 'power'],
                'techniques': ['aggressive_pipelining', 'high_parallelism', 'resource_duplication'],
                'trade_offs': {'area': 'high', 'power': 'high'}
            },
            'latency': {
                'focus': 'minimum_latency',
                'priority_order': ['latency', 'throughput', 'area', 'power'],
                'techniques': ['deep_pipelining', 'parallel_execution', 'memory_optimization'],
                'trade_offs': {'area': 'medium', 'power': 'medium'}
            },
            'area': {
                'focus': 'minimum_area',
                'priority_order': ['area', 'power', 'throughput', 'latency'],
                'techniques': ['resource_sharing', 'time_multiplexing', 'compression'],
                'trade_offs': {'throughput': 'low', 'latency': 'high'}
            },
            'power': {
                'focus': 'minimum_power',
                'priority_order': ['power', 'area', 'throughput', 'latency'],
                'techniques': ['clock_gating', 'voltage_scaling', 'activity_reduction'],
                'trade_offs': {'throughput': 'medium', 'latency': 'medium'}
            },
            'balanced': {
                'focus': 'balanced_optimization',
                'priority_order': ['throughput', 'area', 'latency', 'power'],
                'techniques': ['moderate_pipelining', 'selective_sharing', 'adaptive_clocking'],
                'trade_offs': {'all': 'medium'}
            }
        }
    
    def _load_constraint_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load constraint profiles for different scenarios"""
        return {
            'edge_device': {
                'power_budget': 5.0,  # Watts
                'area_budget': 0.3,   # Fraction of device
                'timing_margin': 0.1, # 10% margin
                'optimization_focus': 'power'
            },
            'datacenter': {
                'power_budget': 50.0,
                'area_budget': 0.8,
                'timing_margin': 0.05,
                'optimization_focus': 'throughput'
            },
            'embedded': {
                'power_budget': 2.0,
                'area_budget': 0.5,
                'timing_margin': 0.15,
                'optimization_focus': 'area'
            },
            'research': {
                'power_budget': 20.0,
                'area_budget': 0.9,
                'timing_margin': 0.2,
                'optimization_focus': 'balanced'
            }
        }
    
    def _load_optimization_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Load available optimization techniques"""
        return {
            'aggressive_pipelining': {
                'description': 'Deep pipeline for maximum throughput',
                'impact': {'throughput': 'high', 'latency': 'medium', 'area': 'high'},
                'applicability': ['throughput', 'balanced']
            },
            'resource_sharing': {
                'description': 'Share resources to minimize area',
                'impact': {'area': 'high', 'power': 'medium', 'throughput': 'low'},
                'applicability': ['area', 'power']
            },
            'parallel_execution': {
                'description': 'Parallel processing units',
                'impact': {'throughput': 'high', 'area': 'high', 'power': 'high'},
                'applicability': ['throughput', 'latency']
            },
            'clock_gating': {
                'description': 'Dynamic clock control for power saving',
                'impact': {'power': 'high', 'area': 'low'},
                'applicability': ['power', 'balanced']
            },
            'memory_optimization': {
                'description': 'Optimize memory access patterns',
                'impact': {'latency': 'medium', 'power': 'medium', 'throughput': 'medium'},
                'applicability': ['latency', 'balanced']
            }
        }
    
    def _validate_strategy(self, strategy: str) -> str:
        """Validate and return optimization strategy"""
        if strategy in self.optimization_strategies:
            logger.debug(f"Using optimization strategy: {strategy}")
            return strategy
        else:
            logger.warning(f"Unknown strategy: {strategy}, using balanced")
            return 'balanced'
    
    def _configure_performance_targets(self, 
                                     strategy: str,
                                     user_targets: Dict[str, float]) -> Dict[str, float]:
        """Configure performance targets based on strategy and user inputs"""
        
        strategy_info = self.optimization_strategies[strategy]
        
        # Default targets based on strategy
        default_targets = self._get_default_targets(strategy)
        
        # Merge with user targets
        configured_targets = default_targets.copy()
        configured_targets.update(user_targets)
        
        # Validate targets
        validated_targets = self._validate_performance_targets(configured_targets)
        
        logger.debug(f"Configured performance targets: {validated_targets}")
        return validated_targets
    
    def _get_default_targets(self, strategy: str) -> Dict[str, float]:
        """Get default performance targets for strategy"""
        defaults = {
            'throughput': {
                'throughput': 1000.0,  # ops/sec
                'latency': 50.0,       # ms
                'efficiency': 0.8
            },
            'latency': {
                'latency': 10.0,
                'throughput': 500.0,
                'efficiency': 0.7
            },
            'area': {
                'area_efficiency': 0.9,
                'throughput': 200.0,
                'latency': 100.0
            },
            'power': {
                'power_efficiency': 0.9,
                'throughput': 300.0,
                'latency': 80.0
            },
            'balanced': {
                'throughput': 500.0,
                'latency': 30.0,
                'efficiency': 0.75
            }
        }
        return defaults.get(strategy, defaults['balanced'])
    
    def _validate_performance_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        """Validate performance targets"""
        validated = {}
        
        # Throughput validation
        if 'throughput' in targets:
            throughput = max(1.0, min(100000.0, targets['throughput']))
            validated['throughput'] = throughput
        
        # Latency validation
        if 'latency' in targets:
            latency = max(0.1, min(10000.0, targets['latency']))
            validated['latency'] = latency
        
        # Efficiency validation
        if 'efficiency' in targets:
            efficiency = max(0.1, min(1.0, targets['efficiency']))
            validated['efficiency'] = efficiency
        
        # Other targets
        for key, value in targets.items():
            if key not in validated and isinstance(value, (int, float)):
                if value >= 0:
                    validated[key] = float(value)
        
        return validated
    
    def _configure_power_constraints(self, 
                                   strategy: str,
                                   user_constraints: Dict[str, float]) -> Dict[str, float]:
        """Configure power constraints"""
        
        # Default power constraints based on strategy
        defaults = {
            'max_power': 10.0,      # Watts
            'max_dynamic_power': 8.0,
            'max_static_power': 2.0,
            'power_efficiency_target': 0.8
        }
        
        if strategy == 'power':
            # Stricter power constraints for power optimization
            defaults['max_power'] = 5.0
            defaults['max_dynamic_power'] = 3.0
            defaults['power_efficiency_target'] = 0.9
        elif strategy == 'throughput':
            # Relaxed power constraints for throughput optimization
            defaults['max_power'] = 20.0
            defaults['max_dynamic_power'] = 16.0
            defaults['power_efficiency_target'] = 0.6
        
        # Merge with user constraints
        configured = defaults.copy()
        configured.update(user_constraints)
        
        # Validate constraints
        validated = self._validate_power_constraints(configured)
        
        logger.debug(f"Configured power constraints: {validated}")
        return validated
    
    def _validate_power_constraints(self, constraints: Dict[str, float]) -> Dict[str, float]:
        """Validate power constraints"""
        validated = {}
        
        for key, value in constraints.items():
            if isinstance(value, (int, float)) and value > 0:
                validated[key] = float(value)
            else:
                logger.warning(f"Invalid power constraint: {key}={value}")
        
        return validated
    
    def _configure_timing_constraints(self, 
                                    strategy: str,
                                    performance_targets: Dict[str, float]) -> Dict[str, float]:
        """Configure timing constraints"""
        
        # Base timing constraints
        constraints = {
            'target_frequency': 100.0,  # MHz
            'setup_margin': 0.1,        # 10%
            'hold_margin': 0.05,        # 5%
            'max_path_delay': 8.0       # ns
        }
        
        # Adjust based on strategy
        if strategy == 'throughput':
            constraints['target_frequency'] = 200.0
            constraints['setup_margin'] = 0.05
        elif strategy == 'latency':
            constraints['target_frequency'] = 250.0
            constraints['setup_margin'] = 0.05
        elif strategy == 'area':
            constraints['target_frequency'] = 50.0
            constraints['setup_margin'] = 0.15
        elif strategy == 'power':
            constraints['target_frequency'] = 75.0
            constraints['setup_margin'] = 0.15
        
        # Consider performance targets
        if 'latency' in performance_targets:
            # Higher frequency for lower latency requirements
            target_latency = performance_targets['latency']
            if target_latency < 20.0:
                constraints['target_frequency'] = min(300.0, constraints['target_frequency'] * 1.5)
        
        logger.debug(f"Configured timing constraints: {constraints}")
        return constraints
    
    def _configure_resource_constraints(self, 
                                      strategy: str,
                                      performance_targets: Dict[str, float]) -> Dict[str, float]:
        """Configure resource constraints"""
        
        # Base resource constraints (as fractions of device resources)
        constraints = {
            'max_lut_usage': 0.8,
            'max_dsp_usage': 0.8,
            'max_bram_usage': 0.8,
            'max_ff_usage': 0.9
        }
        
        # Adjust based on strategy
        if strategy == 'area':
            # Stricter resource constraints for area optimization
            constraints.update({
                'max_lut_usage': 0.5,
                'max_dsp_usage': 0.6,
                'max_bram_usage': 0.6
            })
        elif strategy == 'throughput':
            # Relaxed resource constraints for throughput
            constraints.update({
                'max_lut_usage': 0.9,
                'max_dsp_usage': 0.9,
                'max_bram_usage': 0.9
            })
        
        logger.debug(f"Configured resource constraints: {constraints}")
        return constraints
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available optimization strategies"""
        return list(self.optimization_strategies.keys())
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """Get information about a specific strategy"""
        return self.optimization_strategies.get(strategy, {})
    
    def get_optimization_techniques(self, strategy: str) -> List[str]:
        """Get applicable optimization techniques for strategy"""
        strategy_info = self.optimization_strategies.get(strategy, {})
        return strategy_info.get('techniques', [])
    
    def validate_configuration(self, config: HwOptimizationConfig) -> bool:
        """Validate hardware optimization configuration"""
        # Check strategy
        if config.optimization_strategy not in self.optimization_strategies:
            logger.error(f"Invalid optimization strategy: {config.optimization_strategy}")
            return False
        
        # Validate targets
        if not self._validate_performance_targets(config.performance_targets):
            logger.error("Invalid performance targets")
            return False
        
        # Validate constraints
        if not self._validate_power_constraints(config.power_constraints):
            logger.error("Invalid power constraints")
            return False
        
        return True
    
    def get_recommended_strategy(self, 
                               use_case: str,
                               constraints: Dict[str, float]) -> str:
        """Get recommended optimization strategy for use case"""
        
        # Use case mapping
        use_case_mapping = {
            'edge_inference': 'power',
            'datacenter_inference': 'throughput',
            'real_time': 'latency',
            'resource_constrained': 'area',
            'general_purpose': 'balanced'
        }
        
        recommended = use_case_mapping.get(use_case, 'balanced')
        
        # Adjust based on constraints
        power_constraint = constraints.get('max_power', float('inf'))
        area_constraint = constraints.get('max_area', 1.0)
        
        if power_constraint < 5.0:
            recommended = 'power'
        elif area_constraint < 0.3:
            recommended = 'area'
        
        logger.info(f"Recommended strategy for {use_case}: {recommended}")
        return recommended