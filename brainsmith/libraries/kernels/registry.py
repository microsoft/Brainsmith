"""
Kernel registry and discovery system.

Discovers and manages existing custom_op/ functionality.
"""

import os
import glob
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class KernelRegistry:
    """Registry for managing kernel components."""
    
    def __init__(self):
        """Initialize kernel registry."""
        self.kernels: Dict[str, Any] = {}
        self.logger = logging.getLogger("brainsmith.libraries.kernels.registry")
    
    def register_kernel(self, name: str, kernel_component):
        """Register a kernel component."""
        self.kernels[name] = kernel_component
        self.logger.debug(f"Registered kernel: {name}")
    
    def get_kernel(self, name: str):
        """Get a kernel by name."""
        return self.kernels.get(name)
    
    def list_kernels(self) -> List[str]:
        """List all registered kernels."""
        return list(self.kernels.keys())
    
    def clear(self):
        """Clear all registered kernels."""
        self.kernels.clear()


def discover_kernels(search_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Discover kernels from existing custom_op/ functionality.
    
    Args:
        search_paths: Paths to search for kernels
        
    Returns:
        Dictionary of discovered kernels with metadata
    """
    discovered = {}
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            logger.debug(f"Search path does not exist: {search_path}")
            continue
        
        logger.info(f"Searching for kernels in: {search_path}")
        
        # Look for different kernel file patterns
        patterns = [
            "*.cpp", "*.c", "*.hpp", "*.h",  # C++ kernels
            "*.py",                          # Python kernels
            "*.tcl",                         # HLS TCL scripts
            "*.json"                         # Kernel metadata files
        ]
        
        for pattern in patterns:
            search_pattern = os.path.join(search_path, "**", pattern)
            files = glob.glob(search_pattern, recursive=True)
            
            for file_path in files:
                kernel_info = _analyze_kernel_file(file_path)
                if kernel_info:
                    kernel_name = kernel_info['name']
                    if kernel_name not in discovered:
                        discovered[kernel_name] = kernel_info
                    else:
                        # Merge information if kernel found in multiple files
                        discovered[kernel_name].update(kernel_info)
    
    logger.info(f"Discovered {len(discovered)} kernels")
    return discovered


def _analyze_kernel_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Analyze a file to extract kernel information.
    
    Args:
        file_path: Path to kernel file
        
    Returns:
        Kernel information dictionary or None
    """
    try:
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).stem
        
        kernel_info = {
            'name': file_name,
            'path': file_path,
            'type': _determine_kernel_type(file_ext),
            'parameters': {},
            'metadata': {}
        }
        
        # Extract information based on file type
        if file_ext == '.json':
            # JSON metadata file
            try:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    kernel_info.update(json_data)
            except Exception as e:
                logger.warning(f"Failed to parse JSON metadata {file_path}: {e}")
        
        elif file_ext in ['.cpp', '.c', '.hpp', '.h']:
            # C++ kernel file
            _extract_cpp_kernel_info(file_path, kernel_info)
        
        elif file_ext == '.py':
            # Python kernel file
            _extract_python_kernel_info(file_path, kernel_info)
        
        elif file_ext == '.tcl':
            # HLS TCL script
            _extract_tcl_kernel_info(file_path, kernel_info)
        
        # Add default parameters if not specified
        if not kernel_info['parameters']:
            kernel_info['parameters'] = _get_default_kernel_parameters()
        
        return kernel_info
        
    except Exception as e:
        logger.warning(f"Failed to analyze kernel file {file_path}: {e}")
        return None


def _determine_kernel_type(file_ext: str) -> str:
    """Determine kernel type from file extension."""
    type_mapping = {
        '.cpp': 'hls_cpp',
        '.c': 'hls_c', 
        '.hpp': 'hls_header',
        '.h': 'hls_header',
        '.py': 'python',
        '.tcl': 'hls_tcl',
        '.json': 'metadata'
    }
    return type_mapping.get(file_ext, 'unknown')


def _extract_cpp_kernel_info(file_path: str, kernel_info: Dict[str, Any]):
    """Extract information from C++ kernel files."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for pragma directives that indicate parallelism options
        if '#pragma HLS' in content:
            kernel_info['type'] = 'hls_cpp'
            
            # Extract PE/SIMD hints from pragmas
            if 'PIPELINE' in content:
                kernel_info['supports_pipelining'] = True
            if 'UNROLL' in content:
                kernel_info['supports_unrolling'] = True
        
        # Look for template parameters that might indicate parallelism
        if 'template' in content and ('PE' in content or 'SIMD' in content):
            kernel_info['template_parameters'] = True
            
        # Extract any comments with parameter information
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('//') and ('PE:' in line or 'SIMD:' in line):
                # Parse parameter hints from comments
                _parse_parameter_comment(line, kernel_info)
                
    except Exception as e:
        logger.warning(f"Failed to parse C++ file {file_path}: {e}")


def _extract_python_kernel_info(file_path: str, kernel_info: Dict[str, Any]):
    """Extract information from Python kernel files."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        kernel_info['type'] = 'python'
        
        # Look for class definitions or function definitions
        if 'class ' in content:
            kernel_info['implementation'] = 'class'
        elif 'def ' in content:
            kernel_info['implementation'] = 'function'
            
        # Look for parameter docstrings or comments
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#') and ('PE:' in line or 'SIMD:' in line):
                _parse_parameter_comment(line, kernel_info)
                
    except Exception as e:
        logger.warning(f"Failed to parse Python file {file_path}: {e}")


def _extract_tcl_kernel_info(file_path: str, kernel_info: Dict[str, Any]):
    """Extract information from TCL script files."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        kernel_info['type'] = 'hls_tcl'
        
        # Look for HLS directives
        if 'set_directive_pipeline' in content:
            kernel_info['supports_pipelining'] = True
        if 'set_directive_unroll' in content:
            kernel_info['supports_unrolling'] = True
            
    except Exception as e:
        logger.warning(f"Failed to parse TCL file {file_path}: {e}")


def _parse_parameter_comment(comment: str, kernel_info: Dict[str, Any]):
    """Parse parameter information from code comments."""
    comment = comment.replace('//', '').replace('#', '').strip()
    
    if 'PE:' in comment:
        try:
            pe_part = comment.split('PE:')[1].split()[0]
            if '[' in pe_part and ']' in pe_part:
                # Parse range like "PE: [1,2,4,8]"
                pe_values = eval(pe_part)
                kernel_info['pe_parallelism'] = pe_values
        except:
            pass
    
    if 'SIMD:' in comment:
        try:
            simd_part = comment.split('SIMD:')[1].split()[0]
            if '[' in simd_part and ']' in simd_part:
                simd_values = eval(simd_part)
                kernel_info['simd_parallelism'] = simd_values
        except:
            pass


def _get_default_kernel_parameters() -> Dict[str, Any]:
    """Get default parameters for kernels."""
    return {
        'pe_parallelism': [1, 2, 4, 8, 16],
        'simd_parallelism': [1, 2, 4, 8, 16],
        'supported_precisions': ['int8', 'int16', 'int32'],
        'pipelining_supported': True,
        'unrolling_supported': True
    }


# Example kernel discovery for mock purposes
def get_mock_kernels() -> Dict[str, Dict[str, Any]]:
    """Get mock kernels for testing when no custom_op/ found."""
    return {
        'matrix_vector_multiply': {
            'name': 'matrix_vector_multiply',
            'path': 'custom_op/matrix_vector_multiply.cpp',
            'type': 'hls_cpp',
            'description': 'Matrix-vector multiplication kernel',
            'pe_parallelism': [1, 2, 4, 8, 16, 32],
            'simd_parallelism': [1, 2, 4, 8, 16],
            'supported_precisions': ['int8', 'int16'],
            'parameters': {
                'pe_parallelism': [1, 2, 4, 8, 16, 32],
                'simd_parallelism': [1, 2, 4, 8, 16],
                'supported_precisions': ['int8', 'int16']
            }
        },
        'convolution': {
            'name': 'convolution',
            'path': 'custom_op/convolution.cpp',
            'type': 'hls_cpp',
            'description': 'Convolution kernel with sliding window',
            'pe_parallelism': [1, 2, 4, 8, 16],
            'simd_parallelism': [1, 2, 4, 8],
            'supported_precisions': ['int8', 'int16', 'int32'],
            'parameters': {
                'pe_parallelism': [1, 2, 4, 8, 16],
                'simd_parallelism': [1, 2, 4, 8],
                'supported_precisions': ['int8', 'int16', 'int32']
            }
        },
        'lookup_table': {
            'name': 'lookup_table',
            'path': 'custom_op/lookup_table.cpp',
            'type': 'hls_cpp',
            'description': 'Lookup table implementation',
            'pe_parallelism': [1, 2, 4, 8],
            'simd_parallelism': [1, 2, 4],
            'supported_precisions': ['int8'],
            'parameters': {
                'pe_parallelism': [1, 2, 4, 8],
                'simd_parallelism': [1, 2, 4],
                'supported_precisions': ['int8']
            }
        }
    }