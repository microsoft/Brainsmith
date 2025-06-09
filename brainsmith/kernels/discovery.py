"""
FINN Kernel Discovery Engine
Automated discovery and analysis of available FINN kernels.
"""

import os
import re
import ast
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from .database import (
    FINNKernelInfo, OperatorType, BackendType, PerformanceClass,
    ParameterSchema, ResourceRequirements, PerformanceModel
)

logger = logging.getLogger(__name__)

@dataclass
class KernelInfo:
    """Basic kernel information from discovery"""
    name: str
    path: str
    operator_type: str
    backend_type: str
    implementation_files: Dict[str, str]

@dataclass
class KernelMetadata:
    """Metadata extracted from kernel analysis"""
    parameterization: Dict[str, Any]
    resource_estimates: Dict[str, int]
    performance_characteristics: Dict[str, Any]
    dependencies: List[str]
    documentation: str

class FINNKernelDiscovery:
    """
    Automated discovery and analysis of FINN kernels
    
    Scans FINN installation to discover available kernels, analyzes their
    implementation structure, and extracts parameterization information.
    """
    
    def __init__(self):
        self.discovered_kernels: List[KernelInfo] = []
        self.analysis_results: Dict[str, KernelMetadata] = {}
        
        # Known FINN operator patterns
        self.operator_patterns = {
            'MatMul': [r'MatMul', r'MVAU', r'Matrix.*Vector'],
            'Thresholding': [r'Thresholding', r'MultiThreshold', r'Threshold'],
            'LayerNorm': [r'LayerNorm', r'Normalize', r'BatchNorm'],
            'Convolution': [r'Conv', r'SlidingWindow', r'ConvolutionInputGenerator'],
            'Pool': [r'Pool', r'Pooling', r'MaxPool', r'AvgPool'],
            'ElementWise': [r'Add', r'Mul', r'ElementWise'],
            'Reshape': [r'Reshape', r'Transpose', r'Flatten'],
            'Concat': [r'Concat', r'Merge']
        }
        
        # Backend type patterns
        self.backend_patterns = {
            'RTL': [r'\.v$', r'\.sv$', r'\.vhd$'],
            'HLS': [r'\.cpp$', r'\.hpp$', r'\.cc$', r'\.h$'],
            'Python': [r'\.py$']
        }
        
        # Parameter extraction patterns
        self.parameter_patterns = {
            'pe': [r'PE\s*=\s*(\d+)', r'pe\s*=\s*(\d+)', r'PE_COUNT\s*=\s*(\d+)'],
            'simd': [r'SIMD\s*=\s*(\d+)', r'simd\s*=\s*(\d+)', r'SIMD_WIDTH\s*=\s*(\d+)'],
            'folding': [r'FOLD\s*=\s*(\d+)', r'folding\s*=\s*(\d+)']
        }
    
    def scan_finn_installation(self, finn_path: str) -> List[KernelInfo]:
        """
        Scan FINN installation for available kernels
        
        Args:
            finn_path: Path to FINN installation directory
            
        Returns:
            List of discovered kernel information
        """
        logger.info(f"Scanning FINN installation at: {finn_path}")
        
        self.discovered_kernels = []
        
        # Key directories to scan in FINN
        scan_directories = [
            "src/finn/custom_op",
            "custom_op",
            "finn-hlslib",
            "deps/finn-hlslib"
        ]
        
        for scan_dir in scan_directories:
            full_path = os.path.join(finn_path, scan_dir)
            if os.path.exists(full_path):
                self._scan_directory(full_path)
        
        logger.info(f"Discovered {len(self.discovered_kernels)} kernels")
        return self.discovered_kernels
    
    def _scan_directory(self, directory: str) -> None:
        """Recursively scan directory for kernel implementations"""
        
        for root, dirs, files in os.walk(directory):
            # Look for kernel implementation directories
            if self._is_kernel_directory(root, files):
                kernel_info = self._analyze_kernel_directory(root, files)
                if kernel_info:
                    self.discovered_kernels.append(kernel_info)
    
    def _is_kernel_directory(self, directory: str, files: List[str]) -> bool:
        """Determine if directory contains a kernel implementation"""
        
        # Check for implementation files
        has_implementation = any(
            any(re.search(pattern, f) for pattern in patterns)
            for patterns in self.backend_patterns.values()
            for f in files
        )
        
        # Check for kernel naming patterns
        dir_name = os.path.basename(directory)
        has_kernel_name = any(
            any(re.search(pattern, dir_name, re.IGNORECASE) for pattern in patterns)
            for patterns in self.operator_patterns.values()
        )
        
        return has_implementation and (has_kernel_name or len(files) > 2)
    
    def _analyze_kernel_directory(self, directory: str, files: List[str]) -> Optional[KernelInfo]:
        """Analyze kernel directory to extract basic information"""
        
        try:
            # Determine kernel name
            kernel_name = self._extract_kernel_name(directory, files)
            
            # Determine operator type
            operator_type = self._determine_operator_type(directory, files)
            
            # Determine backend type
            backend_type = self._determine_backend_type(files)
            
            # Collect implementation files
            implementation_files = self._collect_implementation_files(directory, files)
            
            return KernelInfo(
                name=kernel_name,
                path=directory,
                operator_type=operator_type,
                backend_type=backend_type,
                implementation_files=implementation_files
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze kernel directory {directory}: {e}")
            return None
    
    def _extract_kernel_name(self, directory: str, files: List[str]) -> str:
        """Extract kernel name from directory or files"""
        
        # Try directory name first
        dir_name = os.path.basename(directory)
        
        # Clean up directory name
        kernel_name = re.sub(r'[_-]+(rtl|hls|impl)$', '', dir_name, flags=re.IGNORECASE)
        kernel_name = re.sub(r'^(finn_|hlslib_)', '', kernel_name)
        
        return kernel_name
    
    def _determine_operator_type(self, directory: str, files: List[str]) -> str:
        """Determine operator type from directory and file analysis"""
        
        # Analyze directory name
        dir_name = os.path.basename(directory).lower()
        
        for op_type, patterns in self.operator_patterns.items():
            if any(re.search(pattern.lower(), dir_name) for pattern in patterns):
                return op_type
        
        # Analyze file contents for operator hints
        for file in files[:5]:  # Check first few files
            file_path = os.path.join(directory, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1000)  # Read first 1000 chars
                    
                    for op_type, patterns in self.operator_patterns.items():
                        if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                            return op_type
            except:
                continue
        
        return "Custom"
    
    def _determine_backend_type(self, files: List[str]) -> str:
        """Determine backend implementation type"""
        
        # Count files by type
        type_counts = {'RTL': 0, 'HLS': 0, 'Python': 0}
        
        for file in files:
            for backend_type, patterns in self.backend_patterns.items():
                if any(re.search(pattern, file) for pattern in patterns):
                    type_counts[backend_type] += 1
        
        # Return dominant type
        if type_counts['RTL'] > 0:
            return 'RTL'
        elif type_counts['HLS'] > 0:
            return 'HLS'
        elif type_counts['Python'] > 0:
            return 'Python'
        else:
            return 'RTL'  # Default assumption
    
    def _collect_implementation_files(self, directory: str, files: List[str]) -> Dict[str, str]:
        """Collect relevant implementation files"""
        
        implementation_files = {}
        
        for file in files:
            file_path = os.path.join(directory, file)
            
            # Categorize files
            if any(re.search(pattern, file) for pattern in self.backend_patterns['RTL']):
                implementation_files['rtl'] = file_path
            elif any(re.search(pattern, file) for pattern in self.backend_patterns['HLS']):
                implementation_files['hls'] = file_path
            elif file.endswith('.py'):
                implementation_files['python'] = file_path
            elif file.endswith('.json'):
                implementation_files['config'] = file_path
            elif file.lower() in ['readme.md', 'readme.txt', 'doc.md']:
                implementation_files['documentation'] = file_path
        
        return implementation_files
    
    def analyze_kernel_structure(self, kernel_path: str) -> KernelMetadata:
        """
        Analyze kernel implementation files and extract detailed metadata
        
        Args:
            kernel_path: Path to kernel implementation directory
            
        Returns:
            Detailed kernel metadata
        """
        logger.info(f"Analyzing kernel structure: {kernel_path}")
        
        metadata = KernelMetadata(
            parameterization={},
            resource_estimates={},
            performance_characteristics={},
            dependencies=[],
            documentation=""
        )
        
        # Analyze all files in kernel directory
        for root, dirs, files in os.walk(kernel_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    if file.endswith(('.cpp', '.hpp', '.h', '.cc')):
                        self._analyze_hls_file(file_path, metadata)
                    elif file.endswith(('.v', '.sv', '.vhd')):
                        self._analyze_rtl_file(file_path, metadata)
                    elif file.endswith('.py'):
                        self._analyze_python_file(file_path, metadata)
                    elif file.endswith('.json'):
                        self._analyze_config_file(file_path, metadata)
                    elif 'readme' in file.lower() or 'doc' in file.lower():
                        self._extract_documentation(file_path, metadata)
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze file {file_path}: {e}")
        
        # Post-process metadata
        self._infer_missing_parameters(metadata)
        
        return metadata
    
    def _analyze_hls_file(self, file_path: str, metadata: KernelMetadata) -> None:
        """Analyze HLS implementation file"""
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract parameters
        for param_name, patterns in self.parameter_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    metadata.parameterization[param_name] = [int(m) for m in matches]
        
        # Extract pragma directives for resource hints
        pragma_matches = re.findall(r'#pragma\s+HLS\s+(\w+).*', content)
        if pragma_matches:
            metadata.performance_characteristics['hls_pragmas'] = pragma_matches
        
        # Extract template parameters
        template_matches = re.findall(r'template\s*<([^>]+)>', content)
        if template_matches:
            metadata.parameterization['template_params'] = template_matches
    
    def _analyze_rtl_file(self, file_path: str, metadata: KernelMetadata) -> None:
        """Analyze RTL implementation file"""
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract parameters from Verilog/VHDL
        param_matches = re.findall(r'parameter\s+(\w+)\s*=\s*(\d+)', content, re.IGNORECASE)
        for param_name, value in param_matches:
            if param_name.lower() in ['pe', 'simd', 'width', 'depth']:
                metadata.parameterization[param_name.lower()] = int(value)
        
        # Extract port information
        port_matches = re.findall(r'(input|output)\s+.*?(\w+)', content, re.IGNORECASE)
        if port_matches:
            metadata.performance_characteristics['ports'] = len(port_matches)
    
    def _analyze_python_file(self, file_path: str, metadata: KernelMetadata) -> None:
        """Analyze Python implementation file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse Python AST for class and function definitions
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    metadata.performance_characteristics['classes'] = \
                        metadata.performance_characteristics.get('classes', []) + [node.name]
                elif isinstance(node, ast.FunctionDef):
                    metadata.performance_characteristics['functions'] = \
                        metadata.performance_characteristics.get('functions', []) + [node.name]
        
        except:
            # If AST parsing fails, fall back to regex
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            class_matches = re.findall(r'class\s+(\w+)', content)
            if class_matches:
                metadata.performance_characteristics['classes'] = class_matches
    
    def _analyze_config_file(self, file_path: str, metadata: KernelMetadata) -> None:
        """Analyze JSON configuration file"""
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Extract configuration parameters
            if isinstance(config, dict):
                for key, value in config.items():
                    if key.lower() in ['pe', 'simd', 'parameters', 'config']:
                        metadata.parameterization[key.lower()] = value
        
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in config file: {file_path}")
    
    def _extract_documentation(self, file_path: str, metadata: KernelMetadata) -> None:
        """Extract documentation content"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2000)  # Read first 2000 characters
                metadata.documentation += content + "\n\n"
        except:
            pass
    
    def _infer_missing_parameters(self, metadata: KernelMetadata) -> None:
        """Infer missing parameters from available information"""
        
        # Set default parameter ranges if not found
        if 'pe' not in metadata.parameterization:
            metadata.parameterization['pe_range'] = (1, 64)
        else:
            pe_values = metadata.parameterization['pe']
            if isinstance(pe_values, list):
                metadata.parameterization['pe_range'] = (min(pe_values), max(pe_values))
            else:
                metadata.parameterization['pe_range'] = (1, pe_values)
        
        if 'simd' not in metadata.parameterization:
            metadata.parameterization['simd_range'] = (1, 32)
        else:
            simd_values = metadata.parameterization['simd']
            if isinstance(simd_values, list):
                metadata.parameterization['simd_range'] = (min(simd_values), max(simd_values))
            else:
                metadata.parameterization['simd_range'] = (1, simd_values)
        
        # Infer supported datatypes
        if 'supported_datatypes' not in metadata.parameterization:
            metadata.parameterization['supported_datatypes'] = ['int8', 'int16', 'float32']
        
        # Infer memory modes
        if 'memory_modes' not in metadata.parameterization:
            metadata.parameterization['memory_modes'] = ['internal', 'external']
    
    def extract_parameterization(self, kernel_impl: str) -> ParameterSchema:
        """
        Extract parameter schema from kernel implementation
        
        Args:
            kernel_impl: Path to kernel implementation file or directory
            
        Returns:
            Parameter schema for the kernel
        """
        if os.path.isdir(kernel_impl):
            metadata = self.analyze_kernel_structure(kernel_impl)
        else:
            # Analyze single file
            metadata = KernelMetadata(
                parameterization={},
                resource_estimates={},
                performance_characteristics={},
                dependencies=[],
                documentation=""
            )
            
            if kernel_impl.endswith(('.cpp', '.hpp')):
                self._analyze_hls_file(kernel_impl, metadata)
            elif kernel_impl.endswith(('.v', '.sv')):
                self._analyze_rtl_file(kernel_impl, metadata)
        
        # Convert to ParameterSchema
        return ParameterSchema(
            pe_range=metadata.parameterization.get('pe_range', (1, 64)),
            simd_range=metadata.parameterization.get('simd_range', (1, 32)),
            supported_datatypes=metadata.parameterization.get('supported_datatypes', ['int8']),
            memory_modes=metadata.parameterization.get('memory_modes', ['internal']),
            folding_factors=metadata.parameterization.get('folding_factors', {}),
            constraints=metadata.parameterization.get('constraints', {})
        )
    
    def create_kernel_info(self, kernel: KernelInfo, metadata: KernelMetadata) -> FINNKernelInfo:
        """Create complete FINNKernelInfo from discovery results"""
        
        # Create parameter schema
        param_schema = ParameterSchema(
            pe_range=metadata.parameterization.get('pe_range', (1, 64)),
            simd_range=metadata.parameterization.get('simd_range', (1, 32)),
            supported_datatypes=metadata.parameterization.get('supported_datatypes', ['int8']),
            memory_modes=metadata.parameterization.get('memory_modes', ['internal']),
            folding_factors=metadata.parameterization.get('folding_factors', {})
        )
        
        # Create performance model
        perf_model = PerformanceModel(
            model_type="analytical",
            throughput_model={'cycles_per_op': 1, 'base_throughput': 1000},
            latency_model={'base_latency': 10},
            power_model={'base_power': 1.0}
        )
        
        # Create resource requirements
        resources = ResourceRequirements(
            lut_count=metadata.resource_estimates.get('lut', 1000),
            ff_count=metadata.resource_estimates.get('ff', 2000),
            dsp_count=metadata.resource_estimates.get('dsp', 0),
            bram_count=metadata.resource_estimates.get('bram', 0)
        )
        
        return FINNKernelInfo(
            name=kernel.name,
            operator_type=OperatorType(kernel.operator_type),
            backend_type=BackendType(kernel.backend_type),
            implementation_files=kernel.implementation_files,
            parameterization=param_schema,
            performance_model=perf_model,
            resource_requirements=resources,
            finn_version_compatibility=['0.8+'],
            description=metadata.documentation[:200] if metadata.documentation else "",
            performance_class=PerformanceClass.BALANCED
        )