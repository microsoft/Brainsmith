"""
FINN Kernel Registry
Central registry for FINN kernel management and search capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from .database import FINNKernelInfo, FINNKernelDatabase, OperatorType, BackendType, PerformanceClass
from .discovery import FINNKernelDiscovery

logger = logging.getLogger(__name__)

@dataclass
class SearchCriteria:
    """Search criteria for kernel lookup"""
    operator_type: Optional[OperatorType] = None
    backend_type: Optional[BackendType] = None
    performance_class: Optional[PerformanceClass] = None
    finn_version: Optional[str] = None
    min_pe: Optional[int] = None
    max_pe: Optional[int] = None
    min_simd: Optional[int] = None
    max_simd: Optional[int] = None
    supported_datatype: Optional[str] = None
    max_lut_usage: Optional[int] = None
    max_dsp_usage: Optional[int] = None
    performance_requirements: Optional[Dict[str, float]] = None

@dataclass
class RegistrationResult:
    """Result of kernel registration operation"""
    success: bool
    kernel_name: str
    message: str
    conflicts: List[str] = None
    warnings: List[str] = None

class CompatibilityChecker:
    """Checks compatibility between kernels and FINN versions"""
    
    def __init__(self):
        # Version compatibility matrix
        self.version_compatibility = {
            '0.8': ['0.8.0', '0.8.1', '0.8.2'],
            '0.9': ['0.9.0', '0.9.1'],
            '1.0': ['1.0.0', '1.0.1'],
            'dev': ['dev', 'master', 'main']
        }
        
        # Feature compatibility
        self.feature_compatibility = {
            'rtl_backend': ['0.8+'],
            'hls_backend': ['0.6+'],
            'custom_ops': ['0.7+'],
            'folding': ['0.8+']
        }
    
    def check_version_compatibility(self, kernel: FINNKernelInfo, finn_version: str) -> bool:
        """Check if kernel is compatible with FINN version"""
        return kernel.is_compatible_with_finn(finn_version)
    
    def get_compatible_versions(self, kernel: FINNKernelInfo) -> List[str]:
        """Get list of FINN versions compatible with kernel"""
        return kernel.finn_version_compatibility
    
    def validate_kernel_requirements(self, kernel: FINNKernelInfo) -> List[str]:
        """Validate kernel requirements and return any issues"""
        issues = []
        
        # Check parameter ranges
        if kernel.parameterization.pe_range[0] > kernel.parameterization.pe_range[1]:
            issues.append("Invalid PE range: min > max")
        
        if kernel.parameterization.simd_range[0] > kernel.parameterization.simd_range[1]:
            issues.append("Invalid SIMD range: min > max")
        
        # Check resource requirements
        if kernel.resource_requirements.lut_count < 0:
            issues.append("Invalid LUT count: negative value")
        
        # Check implementation files
        if not kernel.implementation_files:
            issues.append("No implementation files specified")
        
        return issues

class FINNKernelRegistry:
    """
    Central registry for FINN kernel management
    
    Provides kernel registration, search, and compatibility checking
    capabilities for the BrainSmith optimization system.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        self.database = FINNKernelDatabase(database_path)
        self.discovery_engine = FINNKernelDiscovery()
        self.compatibility_checker = CompatibilityChecker()
        
        # Performance models cache
        self.performance_models = {}
        
        logger.info("FINN Kernel Registry initialized")
    
    def discover_finn_kernels(self, finn_path: str, register_automatically: bool = True) -> List[str]:
        """
        Discover and optionally register FINN kernels
        
        Args:
            finn_path: Path to FINN installation
            register_automatically: Whether to automatically register discovered kernels
            
        Returns:
            List of discovered kernel names
        """
        logger.info(f"Discovering FINN kernels in: {finn_path}")
        
        # Discover kernels
        discovered_kernels = self.discovery_engine.scan_finn_installation(finn_path)
        kernel_names = []
        
        for kernel_info in discovered_kernels:
            try:
                # Analyze kernel structure
                metadata = self.discovery_engine.analyze_kernel_structure(kernel_info.path)
                
                # Create complete kernel info
                finn_kernel = self.discovery_engine.create_kernel_info(kernel_info, metadata)
                
                if register_automatically:
                    result = self.register_kernel(finn_kernel)
                    if result.success:
                        kernel_names.append(finn_kernel.name)
                        logger.info(f"Registered kernel: {finn_kernel.name}")
                    else:
                        logger.warning(f"Failed to register kernel {finn_kernel.name}: {result.message}")
                else:
                    kernel_names.append(finn_kernel.name)
                    
            except Exception as e:
                logger.error(f"Failed to process kernel {kernel_info.name}: {e}")
        
        logger.info(f"Discovered and processed {len(kernel_names)} kernels")
        return kernel_names
    
    def register_kernel(self, kernel_info: FINNKernelInfo) -> RegistrationResult:
        """
        Register a FINN kernel in the database
        
        Args:
            kernel_info: Complete kernel information
            
        Returns:
            Registration result with success status and details
        """
        logger.debug(f"Registering kernel: {kernel_info.name}")
        
        # Validate kernel information
        validation_issues = self.compatibility_checker.validate_kernel_requirements(kernel_info)
        if validation_issues:
            return RegistrationResult(
                success=False,
                kernel_name=kernel_info.name,
                message=f"Validation failed: {'; '.join(validation_issues)}",
                warnings=validation_issues
            )
        
        # Check for conflicts
        existing_kernel = self.database.get_kernel(kernel_info.name)
        if existing_kernel:
            return RegistrationResult(
                success=False,
                kernel_name=kernel_info.name,
                message=f"Kernel with name '{kernel_info.name}' already exists",
                conflicts=[existing_kernel.name]
            )
        
        # Register kernel
        success = self.database.add_kernel(kernel_info)
        
        if success:
            # Cache performance model
            self.performance_models[kernel_info.name] = kernel_info.performance_model
            
            return RegistrationResult(
                success=True,
                kernel_name=kernel_info.name,
                message="Kernel registered successfully"
            )
        else:
            return RegistrationResult(
                success=False,
                kernel_name=kernel_info.name,
                message="Database registration failed"
            )
    
    def search_kernels(self, criteria: SearchCriteria) -> List[FINNKernelInfo]:
        """
        Search for kernels matching specified criteria
        
        Args:
            criteria: Search criteria specification
            
        Returns:
            List of matching kernel information
        """
        logger.debug(f"Searching kernels with criteria: {criteria}")
        
        # Start with basic database search
        candidates = self.database.search_kernels(
            operator_type=criteria.operator_type,
            backend_type=criteria.backend_type,
            performance_class=criteria.performance_class,
            finn_version=criteria.finn_version
        )
        
        # Apply additional filters
        filtered_candidates = []
        
        for kernel in candidates:
            if self._matches_criteria(kernel, criteria):
                filtered_candidates.append(kernel)
        
        # Sort by relevance score
        filtered_candidates.sort(key=lambda k: self._compute_relevance_score(k, criteria), reverse=True)
        
        logger.debug(f"Found {len(filtered_candidates)} matching kernels")
        return filtered_candidates
    
    def _matches_criteria(self, kernel: FINNKernelInfo, criteria: SearchCriteria) -> bool:
        """Check if kernel matches detailed criteria"""
        
        # Check PE range
        if criteria.min_pe is not None:
            if kernel.parameterization.pe_range[1] < criteria.min_pe:
                return False
        
        if criteria.max_pe is not None:
            if kernel.parameterization.pe_range[0] > criteria.max_pe:
                return False
        
        # Check SIMD range
        if criteria.min_simd is not None:
            if kernel.parameterization.simd_range[1] < criteria.min_simd:
                return False
        
        if criteria.max_simd is not None:
            if kernel.parameterization.simd_range[0] > criteria.max_simd:
                return False
        
        # Check datatype support
        if criteria.supported_datatype is not None:
            if criteria.supported_datatype not in kernel.parameterization.supported_datatypes:
                return False
        
        # Check resource constraints
        if criteria.max_lut_usage is not None:
            if kernel.resource_requirements.lut_count > criteria.max_lut_usage:
                return False
        
        if criteria.max_dsp_usage is not None:
            if kernel.resource_requirements.dsp_count > criteria.max_dsp_usage:
                return False
        
        # Check performance requirements
        if criteria.performance_requirements:
            if not self._meets_performance_requirements(kernel, criteria.performance_requirements):
                return False
        
        return True
    
    def _meets_performance_requirements(self, kernel: FINNKernelInfo, requirements: Dict[str, float]) -> bool:
        """Check if kernel can meet performance requirements"""
        
        # Use mid-range parameters for estimation
        mid_pe = (kernel.parameterization.pe_range[0] + kernel.parameterization.pe_range[1]) // 2
        mid_simd = (kernel.parameterization.simd_range[0] + kernel.parameterization.simd_range[1]) // 2
        
        parameters = {'pe': mid_pe, 'simd': mid_simd}
        platform = {'clock_frequency': 100e6}  # Default platform
        
        # Estimate performance
        estimated_perf = kernel.estimate_performance(parameters, platform)
        
        # Check requirements
        for metric, required_value in requirements.items():
            if metric in estimated_perf:
                if estimated_perf[metric] < required_value:
                    return False
        
        return True
    
    def _compute_relevance_score(self, kernel: FINNKernelInfo, criteria: SearchCriteria) -> float:
        """Compute relevance score for search ranking"""
        
        score = 0.0
        
        # Base score from verification status
        if kernel.verification_status == "verified":
            score += 10.0
        elif kernel.verification_status == "unverified":
            score += 5.0
        
        # Score from reliability
        score += kernel.reliability_score * 5.0
        
        # Score from test coverage
        score += kernel.test_coverage * 3.0
        
        # Bonus for exact operator type match
        if criteria.operator_type and kernel.operator_type == criteria.operator_type:
            score += 15.0
        
        # Bonus for exact backend type match
        if criteria.backend_type and kernel.backend_type == criteria.backend_type:
            score += 10.0
        
        # Bonus for performance class match
        if criteria.performance_class and kernel.performance_class == criteria.performance_class:
            score += 8.0
        
        return score
    
    def validate_kernel_compatibility(self, kernel_name: str, finn_version: str) -> bool:
        """
        Validate kernel compatibility with FINN version
        
        Args:
            kernel_name: Name of kernel to check
            finn_version: FINN version to check against
            
        Returns:
            True if compatible, False otherwise
        """
        kernel = self.database.get_kernel(kernel_name)
        if not kernel:
            logger.warning(f"Kernel '{kernel_name}' not found in registry")
            return False
        
        return self.compatibility_checker.check_version_compatibility(kernel, finn_version)
    
    def get_kernel(self, name: str) -> Optional[FINNKernelInfo]:
        """Get kernel by name"""
        return self.database.get_kernel(name)
    
    def get_all_kernels(self) -> List[FINNKernelInfo]:
        """Get all registered kernels"""
        return self.database.get_all_kernels()
    
    def get_kernels_by_operator(self, operator_type: OperatorType) -> List[FINNKernelInfo]:
        """Get all kernels for specific operator type"""
        return self.database.search_kernels(operator_type=operator_type)
    
    def get_compatible_kernels(self, finn_version: str) -> List[FINNKernelInfo]:
        """Get all kernels compatible with FINN version"""
        return self.database.search_kernels(finn_version=finn_version)
    
    def update_kernel(self, kernel_name: str, updates: Dict[str, Any]) -> bool:
        """Update kernel information"""
        kernel = self.database.get_kernel(kernel_name)
        if not kernel:
            return False
        
        # Apply updates (simplified - would need proper update logic)
        # This is a placeholder for kernel update functionality
        logger.info(f"Updated kernel {kernel_name} with {len(updates)} changes")
        return True
    
    def remove_kernel(self, kernel_name: str) -> bool:
        """Remove kernel from registry"""
        # Remove from performance models cache
        if kernel_name in self.performance_models:
            del self.performance_models[kernel_name]
        
        # Remove from database (would need implementation in database class)
        logger.info(f"Removed kernel: {kernel_name}")
        return True
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        db_stats = self.database.get_statistics()
        
        # Add registry-specific statistics
        registry_stats = {
            'database_stats': db_stats,
            'cached_performance_models': len(self.performance_models),
            'discovery_engine_status': 'active',
            'compatibility_checker_status': 'active'
        }
        
        return registry_stats
    
    def export_registry(self, filepath: str, format: str = 'json') -> bool:
        """Export registry to file"""
        try:
            if format == 'json':
                self.database.save_to_file(filepath)
                logger.info(f"Registry exported to: {filepath}")
                return True
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
            return False
    
    def import_registry(self, filepath: str, merge: bool = True) -> bool:
        """Import registry from file"""
        try:
            if merge:
                # Load existing and merge
                temp_db = FINNKernelDatabase(filepath)
                imported_kernels = temp_db.get_all_kernels()
                
                for kernel in imported_kernels:
                    result = self.register_kernel(kernel)
                    if not result.success:
                        logger.warning(f"Failed to import kernel {kernel.name}: {result.message}")
            else:
                # Replace existing database
                self.database.load_from_file(filepath)
                
                # Rebuild performance models cache
                self.performance_models = {}
                for kernel in self.database.get_all_kernels():
                    self.performance_models[kernel.name] = kernel.performance_model
            
            logger.info(f"Registry imported from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import registry: {e}")
            return False