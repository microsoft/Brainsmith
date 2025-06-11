"""
FINN Kernel Database Schema and Management
Core data structures for FINN kernel information storage and management.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

class OperatorType(Enum):
    """FINN operator types"""
    MATMUL = "MatMul"
    THRESHOLDING = "Thresholding"
    LAYERNORM = "LayerNorm"
    CONVOLUTION = "Convolution"
    POOL = "Pool"
    ELEMENTWISE = "ElementWise"
    RESHAPE = "Reshape"
    CONCAT = "Concat"
    CUSTOM = "Custom"

class BackendType(Enum):
    """FINN backend implementation types"""
    RTL = "RTL"
    HLS = "HLS"
    PYTHON = "Python"

class PerformanceClass(Enum):
    """Performance classification for kernels"""
    HIGH_THROUGHPUT = "high_throughput"
    LOW_LATENCY = "low_latency"
    LOW_POWER = "low_power"
    BALANCED = "balanced"

class ResourceType(Enum):
    """FPGA resource types"""
    LUT = "LUT"
    FF = "FF"
    DSP = "DSP"
    BRAM = "BRAM"
    URAM = "URAM"

@dataclass
class ParameterSchema:
    """Schema for kernel parameterization"""
    pe_range: tuple[int, int]  # (min_pe, max_pe)
    simd_range: tuple[int, int]  # (min_simd, max_simd)
    supported_datatypes: List[str]
    memory_modes: List[str]
    folding_factors: Dict[str, List[int]]
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameter values against schema"""
        # Validate PE range
        if 'pe' in parameters:
            pe = parameters['pe']
            if not (self.pe_range[0] <= pe <= self.pe_range[1]):
                return False
        
        # Validate SIMD range
        if 'simd' in parameters:
            simd = parameters['simd']
            if not (self.simd_range[0] <= simd <= self.simd_range[1]):
                return False
        
        # Validate datatype
        if 'datatype' in parameters:
            if parameters['datatype'] not in self.supported_datatypes:
                return False
        
        return True

@dataclass
class ResourceRequirements:
    """Resource requirements for kernel implementation"""
    lut_count: int
    ff_count: int
    dsp_count: int
    bram_count: int
    uram_count: int = 0
    
    # Scaling factors for parameterization
    lut_scaling: Dict[str, float] = field(default_factory=dict)
    dsp_scaling: Dict[str, float] = field(default_factory=dict)
    bram_scaling: Dict[str, float] = field(default_factory=dict)
    
    def estimate_resources(self, parameters: Dict[str, Any]) -> "ResourceRequirements":
        """Estimate resource usage for given parameters"""
        pe = parameters.get('pe', 1)
        simd = parameters.get('simd', 1)
        
        # Scale base resources
        estimated = ResourceRequirements(
            lut_count=int(self.lut_count * pe * self.lut_scaling.get('pe', 1.0) * 
                         simd * self.lut_scaling.get('simd', 1.0)),
            ff_count=int(self.ff_count * pe * simd * 1.1),  # FF typically scale linearly
            dsp_count=int(self.dsp_count * pe * self.dsp_scaling.get('pe', 1.0)),
            bram_count=int(self.bram_count * self.bram_scaling.get('pe', 0.5))
        )
        
        return estimated
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format"""
        return {
            'lut': self.lut_count,
            'ff': self.ff_count,
            'dsp': self.dsp_count,
            'bram': self.bram_count,
            'uram': self.uram_count
        }

@dataclass
class PerformanceModel:
    """Performance model for kernel"""
    model_type: str  # "analytical" or "empirical"
    throughput_model: Dict[str, Any]
    latency_model: Dict[str, Any]
    power_model: Dict[str, Any]
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    
    def estimate_throughput(self, parameters: Dict[str, Any], platform: Dict[str, Any]) -> float:
        """Estimate throughput for given parameters and platform"""
        if self.model_type == "analytical":
            return self._analytical_throughput(parameters, platform)
        else:
            return self._empirical_throughput(parameters, platform)
    
    def estimate_latency(self, parameters: Dict[str, Any], platform: Dict[str, Any]) -> int:
        """Estimate latency in clock cycles"""
        if self.model_type == "analytical":
            return self._analytical_latency(parameters, platform)
        else:
            return self._empirical_latency(parameters, platform)
    
    def _analytical_throughput(self, parameters: Dict[str, Any], platform: Dict[str, Any]) -> float:
        """Analytical throughput model"""
        pe = parameters.get('pe', 1)
        simd = parameters.get('simd', 1)
        clock_freq = platform.get('clock_frequency', 100e6)  # 100 MHz default
        
        # Simple analytical model: throughput = PE * SIMD * clock_frequency / cycles_per_op
        cycles_per_op = self.throughput_model.get('cycles_per_op', 1)
        throughput = (pe * simd * clock_freq) / cycles_per_op
        
        return throughput
    
    def _empirical_throughput(self, parameters: Dict[str, Any], platform: Dict[str, Any]) -> float:
        """Empirical throughput model using historical data"""
        # Placeholder for ML-based empirical model
        # In real implementation, this would use trained models
        base_throughput = self.throughput_model.get('base_throughput', 1000)
        pe_factor = parameters.get('pe', 1) * 0.9  # Efficiency factor
        simd_factor = parameters.get('simd', 1) * 0.95
        
        return base_throughput * pe_factor * simd_factor
    
    def _analytical_latency(self, parameters: Dict[str, Any], platform: Dict[str, Any]) -> int:
        """Analytical latency model"""
        base_latency = self.latency_model.get('base_latency', 10)
        pe = parameters.get('pe', 1)
        
        # Latency typically decreases with more PE units
        latency = max(1, base_latency // pe)
        return latency
    
    def _empirical_latency(self, parameters: Dict[str, Any], platform: Dict[str, Any]) -> int:
        """Empirical latency model"""
        # Placeholder for empirical model
        base_latency = self.latency_model.get('base_latency', 10)
        return base_latency

@dataclass
class FINNKernelInfo:
    """Complete information about a FINN kernel"""
    name: str
    operator_type: OperatorType
    backend_type: BackendType
    implementation_files: Dict[str, str]
    parameterization: ParameterSchema
    performance_model: PerformanceModel
    resource_requirements: ResourceRequirements
    finn_version_compatibility: List[str]
    
    # Metadata
    description: str = ""
    author: str = ""
    creation_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    verification_status: str = "unverified"  # "verified", "unverified", "deprecated"
    
    # Performance characteristics
    performance_class: PerformanceClass = PerformanceClass.BALANCED
    optimization_target: str = "balanced"  # "throughput", "latency", "power", "area"
    
    # Quality metrics
    reliability_score: float = 0.0  # 0.0 to 1.0
    test_coverage: float = 0.0      # 0.0 to 1.0
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    
    def is_compatible_with_finn(self, finn_version: str) -> bool:
        """Check compatibility with FINN version"""
        return finn_version in self.finn_version_compatibility
    
    def estimate_performance(self, parameters: Dict[str, Any], platform: Dict[str, Any]) -> Dict[str, float]:
        """Estimate performance for given parameters"""
        return {
            'throughput': self.performance_model.estimate_throughput(parameters, platform),
            'latency': self.performance_model.estimate_latency(parameters, platform),
            'resource_efficiency': self._compute_resource_efficiency(parameters)
        }
    
    def _compute_resource_efficiency(self, parameters: Dict[str, Any]) -> float:
        """Compute resource efficiency metric"""
        resources = self.resource_requirements.estimate_resources(parameters)
        total_resources = sum(resources.to_dict().values())
        
        # Simple efficiency metric based on resource utilization
        efficiency = 1.0 / (1.0 + total_resources / 10000)  # Normalize
        return min(1.0, efficiency)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'operator_type': self.operator_type.value,
            'backend_type': self.backend_type.value,
            'implementation_files': self.implementation_files,
            'parameterization': {
                'pe_range': self.parameterization.pe_range,
                'simd_range': self.parameterization.simd_range,
                'supported_datatypes': self.parameterization.supported_datatypes,
                'memory_modes': self.parameterization.memory_modes,
                'folding_factors': self.parameterization.folding_factors
            },
            'resource_requirements': self.resource_requirements.to_dict(),
            'finn_version_compatibility': self.finn_version_compatibility,
            'description': self.description,
            'performance_class': self.performance_class.value,
            'verification_status': self.verification_status,
            'reliability_score': self.reliability_score,
            'test_coverage': self.test_coverage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FINNKernelInfo":
        """Create from dictionary"""
        # This would include full deserialization logic
        # Simplified version for now
        return cls(
            name=data['name'],
            operator_type=OperatorType(data['operator_type']),
            backend_type=BackendType(data['backend_type']),
            implementation_files=data['implementation_files'],
            parameterization=ParameterSchema(
                pe_range=tuple(data['parameterization']['pe_range']),
                simd_range=tuple(data['parameterization']['simd_range']),
                supported_datatypes=data['parameterization']['supported_datatypes'],
                memory_modes=data['parameterization']['memory_modes'],
                folding_factors=data['parameterization']['folding_factors']
            ),
            performance_model=PerformanceModel(
                model_type="analytical",
                throughput_model={},
                latency_model={},
                power_model={}
            ),
            resource_requirements=ResourceRequirements(
                lut_count=data['resource_requirements'].get('lut', 0),
                ff_count=data['resource_requirements'].get('ff', 0),
                dsp_count=data['resource_requirements'].get('dsp', 0),
                bram_count=data['resource_requirements'].get('bram', 0)
            ),
            finn_version_compatibility=data['finn_version_compatibility']
        )

class FINNKernelDatabase:
    """Database for FINN kernel information"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.kernels: Dict[str, FINNKernelInfo] = {}
        self.indices = {
            'by_operator': {},
            'by_backend': {},
            'by_performance_class': {}
        }
        
        if storage_path:
            self.load_from_file(storage_path)
    
    def add_kernel(self, kernel: FINNKernelInfo) -> bool:
        """Add kernel to database"""
        if kernel.name in self.kernels:
            return False  # Kernel already exists
        
        self.kernels[kernel.name] = kernel
        self._update_indices(kernel)
        return True
    
    def get_kernel(self, name: str) -> Optional[FINNKernelInfo]:
        """Get kernel by name"""
        return self.kernels.get(name)
    
    def search_kernels(self, 
                      operator_type: Optional[OperatorType] = None,
                      backend_type: Optional[BackendType] = None,
                      performance_class: Optional[PerformanceClass] = None,
                      finn_version: Optional[str] = None) -> List[FINNKernelInfo]:
        """Search kernels by criteria"""
        results = list(self.kernels.values())
        
        if operator_type:
            results = [k for k in results if k.operator_type == operator_type]
        
        if backend_type:
            results = [k for k in results if k.backend_type == backend_type]
        
        if performance_class:
            results = [k for k in results if k.performance_class == performance_class]
        
        if finn_version:
            results = [k for k in results if k.is_compatible_with_finn(finn_version)]
        
        return results
    
    def get_all_kernels(self) -> List[FINNKernelInfo]:
        """Get all kernels in database"""
        return list(self.kernels.values())
    
    def _update_indices(self, kernel: FINNKernelInfo) -> None:
        """Update search indices"""
        # Update operator type index
        op_type = kernel.operator_type.value
        if op_type not in self.indices['by_operator']:
            self.indices['by_operator'][op_type] = []
        self.indices['by_operator'][op_type].append(kernel.name)
        
        # Update backend type index
        backend = kernel.backend_type.value
        if backend not in self.indices['by_backend']:
            self.indices['by_backend'][backend] = []
        self.indices['by_backend'][backend].append(kernel.name)
        
        # Update performance class index
        perf_class = kernel.performance_class.value
        if perf_class not in self.indices['by_performance_class']:
            self.indices['by_performance_class'][perf_class] = []
        self.indices['by_performance_class'][perf_class].append(kernel.name)
    
    def save_to_file(self, filepath: str) -> None:
        """Save database to JSON file"""
        data = {
            'kernels': {name: kernel.to_dict() for name, kernel in self.kernels.items()},
            'metadata': {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'kernel_count': len(self.kernels)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load database from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.kernels = {}
            for name, kernel_data in data['kernels'].items():
                kernel = FINNKernelInfo.from_dict(kernel_data)
                self.kernels[name] = kernel
                self._update_indices(kernel)
                
        except FileNotFoundError:
            # File doesn't exist yet, start with empty database
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            'total_kernels': len(self.kernels),
            'by_operator_type': {},
            'by_backend_type': {},
            'by_performance_class': {},
            'verification_status': {}
        }
        
        for kernel in self.kernels.values():
            # Count by operator type
            op_type = kernel.operator_type.value
            stats['by_operator_type'][op_type] = stats['by_operator_type'].get(op_type, 0) + 1
            
            # Count by backend type
            backend = kernel.backend_type.value
            stats['by_backend_type'][backend] = stats['by_backend_type'].get(backend, 0) + 1
            
            # Count by performance class
            perf_class = kernel.performance_class.value
            stats['by_performance_class'][perf_class] = stats['by_performance_class'].get(perf_class, 0) + 1
            
            # Count by verification status
            status = kernel.verification_status
            stats['verification_status'][status] = stats['verification_status'].get(status, 0) + 1
        
        return stats