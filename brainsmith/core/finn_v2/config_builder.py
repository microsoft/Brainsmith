"""
Configuration Builder

Utility functions for building FINN configurations from Blueprint V2 data.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigBuilder:
    """Utility class for building FINN configurations."""
    
    def __init__(self):
        """Initialize configuration builder."""
        self.default_params = self._initialize_default_params()
        logger.info("ConfigBuilder initialized")
    
    def build_base_config(self, blueprint_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build base FINN configuration from blueprint.
        
        Args:
            blueprint_config: Blueprint V2 configuration
            
        Returns:
            Base configuration dictionary for FINN
        """
        config = self.default_params.copy()
        
        # Extract and merge blueprint-specific parameters
        if 'constraints' in blueprint_config:
            config.update(self._extract_constraints(blueprint_config['constraints']))
        
        if 'configuration_files' in blueprint_config:
            config.update(self._extract_config_files(blueprint_config['configuration_files']))
        
        if 'objectives' in blueprint_config:
            config.update(self._extract_objectives(blueprint_config['objectives']))
        
        logger.debug(f"Built base config: {config}")
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate FINN configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required parameters
        required_params = ['output_dir', 'synth_clk_period_ns']
        for param in required_params:
            if param not in config:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate parameter ranges
        if 'synth_clk_period_ns' in config:
            clk_period = config['synth_clk_period_ns']
            if not isinstance(clk_period, (int, float)) or clk_period <= 0:
                errors.append("synth_clk_period_ns must be positive number")
        
        if 'target_fps' in config and config['target_fps'] is not None:
            fps = config['target_fps']
            if not isinstance(fps, (int, float)) or fps <= 0:
                errors.append("target_fps must be positive number")
        
        # Validate file paths
        file_params = ['folding_config_file', 'verify_input_npy', 'verify_expected_output_npy']
        for param in file_params:
            if param in config and config[param]:
                if not Path(config[param]).exists():
                    errors.append(f"File not found: {param}={config[param]}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _extract_constraints(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Extract FINN parameters from blueprint constraints."""
        params = {}
        
        # Clock frequency constraint
        if 'target_frequency_mhz' in constraints:
            freq_mhz = constraints['target_frequency_mhz']
            params['synth_clk_period_ns'] = 1000.0 / freq_mhz
        
        # Performance constraints
        if 'target_throughput_fps' in constraints:
            params['target_fps'] = constraints['target_throughput_fps']
        
        # Resource constraints (used for validation)
        resource_constraints = ['max_luts', 'max_dsps', 'max_brams', 'max_power']
        for constraint in resource_constraints:
            if constraint in constraints:
                params[constraint] = constraints[constraint]
        
        return params
    
    def _extract_config_files(self, config_files: Dict[str, str]) -> Dict[str, Any]:
        """Extract configuration file paths."""
        params = {}
        
        # Map blueprint config file keys to FINN parameter names
        file_mappings = {
            'folding_override': 'folding_config_file',
            'platform_config': 'platform_config_file',
            'verification_data': 'verify_input_npy'
        }
        
        for blueprint_key, finn_key in file_mappings.items():
            if blueprint_key in config_files:
                params[finn_key] = config_files[blueprint_key]
        
        return params
    
    def _extract_objectives(self, objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract optimization objectives."""
        params = {}
        
        # Look for performance objectives that map to FINN parameters
        for objective in objectives:
            obj_name = objective.get('name', '')
            
            if obj_name == 'throughput' and 'target' in objective:
                params['target_fps'] = objective['target']
            elif obj_name == 'frequency' and 'target' in objective:
                freq_mhz = objective['target']
                params['synth_clk_period_ns'] = 1000.0 / freq_mhz
        
        return params
    
    def _initialize_default_params(self) -> Dict[str, Any]:
        """Initialize default FINN configuration parameters."""
        return {
            'output_dir': './finn_output',
            'synth_clk_period_ns': 5.0,  # 200 MHz
            'target_fps': None,
            'folding_config_file': None,
            'auto_fifo_depths': True,
            'save_intermediate_models': True,
            'verify_steps': [],
            'board': None,
            'generate_outputs': ['STITCHED_IP']
        }
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries with override precedence.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        merged.update(override_config)
        return merged
    
    def get_optimization_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get optimization-specific configuration parameters.
        
        Args:
            strategy_name: DSE strategy name
            
        Returns:
            Strategy-specific configuration
        """
        strategy_configs = {
            'performance_focused': {
                'auto_fifo_depths': True,
                'minimize_bit_width': True,
                'target_fps': 1000
            },
            'resource_constrained': {
                'auto_fifo_depths': False,
                'minimize_bit_width': True,
                'target_fps': 100
            },
            'balanced': {
                'auto_fifo_depths': True,
                'minimize_bit_width': True,
                'target_fps': 500
            }
        }
        
        return strategy_configs.get(strategy_name, {})