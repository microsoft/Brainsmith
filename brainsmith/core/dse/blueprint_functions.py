"""
Blueprint Functions for DSE Engine

Simple functions for loading and processing blueprint configurations.
These functions provide a clean interface between the DSE engine and
blueprint management infrastructure.
"""

import yaml
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


def load_blueprint_yaml(blueprint_path: str) -> Dict[str, Any]:
    """
    Load blueprint YAML configuration.
    
    Args:
        blueprint_path: Path to blueprint YAML file
        
    Returns:
        Blueprint configuration dictionary
        
    Raises:
        FileNotFoundError: If blueprint file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    try:
        with open(blueprint_path, 'r') as f:
            blueprint_config = yaml.safe_load(f)
        
        logger.info(f"Loaded blueprint: {blueprint_path}")
        return blueprint_config
        
    except FileNotFoundError:
        logger.error(f"Blueprint file not found: {blueprint_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in blueprint {blueprint_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading blueprint {blueprint_path}: {e}")
        raise


def get_build_steps(blueprint_data: Dict[str, Any]) -> List[str]:
    """
    Extract build steps from blueprint configuration.
    
    Args:
        blueprint_data: Blueprint configuration dictionary
        
    Returns:
        List of build step names
    """
    if 'build_steps' in blueprint_data:
        return blueprint_data['build_steps']
    elif 'build' in blueprint_data and 'steps' in blueprint_data['build']:
        return blueprint_data['build']['steps']
    else:
        # Default build steps for FPGA acceleration
        return [
            'load_model',
            'transform_model', 
            'quantize_model',
            'optimize_model',
            'generate_hls',
            'synthesize',
            'implement',
            'generate_bitstream'
        ]


def get_objectives(blueprint_data: Dict[str, Any]) -> List[str]:
    """
    Extract optimization objectives from blueprint configuration.
    
    Args:
        blueprint_data: Blueprint configuration dictionary
        
    Returns:
        List of objective names (e.g., 'throughput', 'latency', 'power')
    """
    if 'objectives' in blueprint_data:
        return blueprint_data['objectives']
    elif 'optimization' in blueprint_data and 'objectives' in blueprint_data['optimization']:
        return blueprint_data['optimization']['objectives']
    else:
        # Default objectives for FPGA DSE
        return ['throughput', 'latency', 'resource_utilization']