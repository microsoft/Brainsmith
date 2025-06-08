"""
Transform registry and discovery system.

Manages registration and discovery of transform steps, organizing
existing steps/ functionality into a structured system.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TransformRegistry:
    """
    Registry for managing transform steps and configurations.
    """
    
    def __init__(self):
        """Initialize transform registry."""
        self.transforms = {}
        self.logger = logging.getLogger("brainsmith.libraries.transforms.registry")
    
    def register_transform(self, name: str, transform_info: Dict[str, Any]):
        """
        Register a transform.
        
        Args:
            name: Transform name
            transform_info: Transform information and configuration
        """
        self.transforms[name] = transform_info
        self.logger.debug(f"Registered transform: {name}")
    
    def get_transform(self, name: str) -> Optional[Dict[str, Any]]:
        """Get transform by name."""
        return self.transforms.get(name)
    
    def get_available_transforms(self) -> List[str]:
        """Get list of available transform names."""
        return list(self.transforms.keys())
    
    def clear(self):
        """Clear all registered transforms."""
        self.transforms.clear()
    
    def get_transforms_by_category(self, category: str) -> Dict[str, Any]:
        """Get transforms filtered by category."""
        filtered = {}
        for name, info in self.transforms.items():
            if info.get('category') == category:
                filtered[name] = info
        return filtered


def discover_transforms(search_paths: List[str]) -> Dict[str, Any]:
    """
    Discover transform steps in specified paths.
    
    Args:
        search_paths: Paths to search for transforms
        
    Returns:
        Dictionary of discovered transforms
    """
    discovered = {}
    
    for search_path in search_paths:
        path = Path(search_path)
        if not path.exists():
            continue
        
        logger.info(f"Searching for transforms in: {path}")
        
        # Look for Python files that might contain transforms
        for py_file in path.glob("**/*.py"):
            if py_file.name.startswith('__'):
                continue
            
            # Simple discovery based on filename patterns
            transform_name = py_file.stem
            
            # Categorize based on common patterns
            category = _categorize_transform(transform_name)
            
            transform_info = {
                'name': transform_name.replace('_', ' ').title(),
                'file_path': str(py_file),
                'category': category,
                'description': f"Transform step: {transform_name}",
                'parameters': _infer_parameters(transform_name)
            }
            
            discovered[transform_name] = transform_info
    
    logger.info(f"Discovered {len(discovered)} transforms")
    return discovered


def _categorize_transform(transform_name: str) -> str:
    """Categorize transform based on name patterns."""
    name_lower = transform_name.lower()
    
    if any(keyword in name_lower for keyword in ['fold', 'folding']):
        return 'folding'
    elif any(keyword in name_lower for keyword in ['stream', 'streaming', 'fifo']):
        return 'streaming'
    elif any(keyword in name_lower for keyword in ['pipeline', 'pipe']):
        return 'pipelining'
    elif any(keyword in name_lower for keyword in ['memory', 'mem', 'buffer']):
        return 'memory'
    elif any(keyword in name_lower for keyword in ['optim', 'opt']):
        return 'optimization'
    else:
        return 'general'


def _infer_parameters(transform_name: str) -> List[str]:
    """Infer likely parameters based on transform name."""
    name_lower = transform_name.lower()
    parameters = []
    
    if 'fold' in name_lower:
        parameters.extend(['folding_factor', 'target_layers'])
    
    if 'stream' in name_lower:
        parameters.extend(['fifo_depth', 'buffer_size'])
    
    if 'pipeline' in name_lower:
        parameters.extend(['pipeline_depth', 'register_balancing'])
    
    if 'memory' in name_lower or 'mem' in name_lower:
        parameters.extend(['memory_type', 'access_pattern'])
    
    # Default parameters
    if not parameters:
        parameters = ['configuration', 'enable']
    
    return parameters


def get_mock_transforms() -> Dict[str, Any]:
    """Get mock transforms for testing."""
    return {
        'conv_folding': {
            'name': 'Convolution Folding',
            'description': 'Apply folding to convolution layers for resource reduction',
            'category': 'folding',
            'parameters': ['folding_factor', 'pe_count', 'simd_count'],
            'resource_impact': {
                'luts': 'reduces',
                'throughput': 'reduces',
                'latency': 'increases'
            }
        },
        'streaming_dataflow': {
            'name': 'Streaming Dataflow',
            'description': 'Enable streaming between layers for improved throughput',
            'category': 'streaming',
            'parameters': ['fifo_depth', 'threshold'],
            'resource_impact': {
                'brams': 'increases',
                'throughput': 'increases',
                'latency': 'reduces'
            }
        },
        'pipeline_insertion': {
            'name': 'Pipeline Insertion',
            'description': 'Insert pipeline registers for higher frequency',
            'category': 'pipelining',
            'parameters': ['pipeline_depth', 'auto_balance'],
            'resource_impact': {
                'ffs': 'increases',
                'frequency': 'increases',
                'latency': 'increases'
            }
        },
        'memory_optimization': {
            'name': 'Memory Optimization',
            'description': 'Optimize memory access patterns and allocation',
            'category': 'memory',
            'parameters': ['memory_type', 'access_width', 'banking_factor'],
            'resource_impact': {
                'brams': 'optimizes',
                'bandwidth': 'improves',
                'power': 'reduces'
            }
        }
    }