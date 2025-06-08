"""
Analysis Integration Layer for Orchestrated Interface and Pragma Analysis.

This module provides the integration layer that coordinates interface analysis
and pragma processing, manages caching, and provides legacy compatibility.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from pathlib import Path
from collections import defaultdict
import threading

# Import dataflow components
try:
    from ...dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
    from ...dataflow.core.dataflow_model import DataflowModel, ParallelismConfiguration
    from ...dataflow.core.validation import ValidationResult, create_validation_result
    DATAFLOW_AVAILABLE = True
except ImportError:
    DATAFLOW_AVAILABLE = False
    # Create placeholder types
    class DataflowInterface: pass
    class DataflowInterfaceType: pass
    class DataflowModel: pass
    class ParallelismConfiguration: pass
    ValidationResult = Dict[str, Any]

# Import Week 1 components
from ..enhanced_data_structures import RTLSignal, RTLInterface, RTLModule, ParsedRTLData
from ..enhanced_config import PipelineConfig

# Import analysis components
from .enhanced_interface_analyzer import (
    InterfaceAnalyzer, InterfaceAnalysisResult, InterfaceClassifier,
    create_interface_analyzer
)
from .enhanced_pragma_processor import (
    PragmaProcessor, PragmaProcessingResult, ParsedPragma,
    create_pragma_processor
)
from .analysis_config import (
    InterfaceAnalysisConfig, PragmaAnalysisConfig, AnalysisMetrics,
    create_analysis_config
)

from ..errors import ValidationError, ConfigurationError


@dataclass
class AnalysisResults:
    """Unified results container for complete analysis."""
    
    # Input information
    rtl_module: RTLModule
    analysis_start_time: float = field(default_factory=time.time)
    
    # Interface analysis results
    interface_results: List[InterfaceAnalysisResult] = field(default_factory=list)
    interface_analysis_time: float = 0.0
    
    # Pragma processing results
    pragma_results: Optional[PragmaProcessingResult] = None
    pragma_processing_time: float = 0.0
    
    # Integrated results
    dataflow_interfaces: List[DataflowInterface] = field(default_factory=list)
    dataflow_model: Optional[DataflowModel] = None
    parallelism_configuration: Optional[ParallelismConfiguration] = None
    
    # Validation results
    overall_validation: Optional[ValidationResult] = None
    validation_time: float = 0.0
    
    # Analysis statistics
    total_analysis_time: float = 0.0
    metrics: AnalysisMetrics = field(default_factory=AnalysisMetrics)
    
    # Status
    success: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def finalize(self) -> None:
        """Finalize analysis results and calculate totals."""
        self.total_analysis_time = time.time() - self.analysis_start_time
        
        # Update success status
        self.success = (
            len(self.interface_results) > 0 and
            len(self.errors) == 0 and
            (self.overall_validation is None or 
             (self.overall_validation.success if DATAFLOW_AVAILABLE 
              else self.overall_validation.get("success", True)))
        )
        
        # Update metrics
        self.metrics.interface_analysis_count = len(self.interface_results)
        self.metrics.interface_analysis_time = self.interface_analysis_time
        
        if self.pragma_results:
            self.metrics.pragma_processing_count = self.pragma_results.pragma_count
            self.metrics.pragma_processing_time = self.pragma_processing_time
            
            if self.pragma_results.pragma_count > 0:
                success_rate = self.pragma_results.valid_pragma_count / self.pragma_results.pragma_count
                self.metrics.pragma_parse_success_rate = success_rate
        
        self.metrics.dataflow_conversion_count = len(self.dataflow_interfaces)
        self.metrics.error_count = len(self.errors)
        self.metrics.warning_count = len(self.warnings)
    
    def add_error(self, error: str) -> None:
        """Add error message."""
        self.errors.append(error)
        self.metrics.add_error()
    
    def add_warning(self, warning: str) -> None:
        """Add warning message."""
        self.warnings.append(warning)
        self.metrics.add_warning()
    
    def get_interface_by_name(self, name: str) -> Optional[InterfaceAnalysisResult]:
        """Get interface analysis result by name."""
        for result in self.interface_results:
            if result.interface_name == name:
                return result
        return None
    
    def get_interfaces_by_type(self, interface_type) -> List[InterfaceAnalysisResult]:
        """Get interface analysis results by type."""
        return [result for result in self.interface_results 
                if result.interface_type == interface_type]
    
    def get_dataflow_interface_by_name(self, name: str) -> Optional[DataflowInterface]:
        """Get dataflow interface by name."""
        for interface in self.dataflow_interfaces:
            if interface.name == name:
                return interface
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary representation."""
        return {
            "module_name": self.rtl_module.name,
            "success": self.success,
            "total_analysis_time": self.total_analysis_time,
            "interface_analysis": {
                "count": len(self.interface_results),
                "time": self.interface_analysis_time,
                "results": [result.to_dict() for result in self.interface_results]
            },
            "pragma_processing": {
                "enabled": self.pragma_results is not None,
                "time": self.pragma_processing_time,
                "results": self.pragma_results.to_dict() if self.pragma_results else None
            },
            "dataflow_integration": {
                "interface_count": len(self.dataflow_interfaces),
                "has_model": self.dataflow_model is not None,
                "has_parallelism_config": self.parallelism_configuration is not None
            },
            "validation": {
                "time": self.validation_time,
                "success": (
                    self.overall_validation.success if DATAFLOW_AVAILABLE and self.overall_validation
                    else self.overall_validation.get("success", True) if self.overall_validation
                    else None
                )
            },
            "metrics": self.metrics.to_dict(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings
        }


class AnalysisCache:
    """
    Cache for analysis results to improve performance.
    
    This cache stores interface analysis and pragma processing results
    to avoid redundant computations.
    """
    
    def __init__(self, max_size: int = 100, ttl: float = 3600.0):
        """Initialize cache with size limit and TTL."""
        self.max_size = max_size
        self.ttl = ttl
        self._interface_cache: Dict[str, Tuple[InterfaceAnalysisResult, float]] = {}
        self._pragma_cache: Dict[str, Tuple[PragmaProcessingResult, float]] = {}
        self._analysis_cache: Dict[str, Tuple[AnalysisResults, float]] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
    
    def get_interface_result(self, cache_key: str) -> Optional[InterfaceAnalysisResult]:
        """Get cached interface analysis result."""
        with self._lock:
            if cache_key in self._interface_cache:
                result, timestamp = self._interface_cache[cache_key]
                if time.time() - timestamp < self.ttl:
                    self._hits += 1
                    return result
                else:
                    # Expired
                    del self._interface_cache[cache_key]
            
            self._misses += 1
            return None
    
    def put_interface_result(self, cache_key: str, result: InterfaceAnalysisResult) -> None:
        """Cache interface analysis result."""
        with self._lock:
            self._cleanup_if_needed(self._interface_cache)
            self._interface_cache[cache_key] = (result, time.time())
    
    def get_pragma_result(self, cache_key: str) -> Optional[PragmaProcessingResult]:
        """Get cached pragma processing result."""
        with self._lock:
            if cache_key in self._pragma_cache:
                result, timestamp = self._pragma_cache[cache_key]
                if time.time() - timestamp < self.ttl:
                    self._hits += 1
                    return result
                else:
                    # Expired
                    del self._pragma_cache[cache_key]
            
            self._misses += 1
            return None
    
    def put_pragma_result(self, cache_key: str, result: PragmaProcessingResult) -> None:
        """Cache pragma processing result."""
        with self._lock:
            self._cleanup_if_needed(self._pragma_cache)
            self._pragma_cache[cache_key] = (result, time.time())
    
    def get_analysis_result(self, cache_key: str) -> Optional[AnalysisResults]:
        """Get cached complete analysis result."""
        with self._lock:
            if cache_key in self._analysis_cache:
                result, timestamp = self._analysis_cache[cache_key]
                if time.time() - timestamp < self.ttl:
                    self._hits += 1
                    return result
                else:
                    # Expired
                    del self._analysis_cache[cache_key]
            
            self._misses += 1
            return None
    
    def put_analysis_result(self, cache_key: str, result: AnalysisResults) -> None:
        """Cache complete analysis result."""
        with self._lock:
            self._cleanup_if_needed(self._analysis_cache)
            self._analysis_cache[cache_key] = (result, time.time())
    
    def _cleanup_if_needed(self, cache_dict: Dict) -> None:
        """Clean up cache if it exceeds max size."""
        if len(cache_dict) >= self.max_size:
            # Remove oldest entries (simple LRU approximation)
            oldest_keys = sorted(cache_dict.keys(), 
                               key=lambda k: cache_dict[k][1])[:len(cache_dict) // 4]
            for key in oldest_keys:
                del cache_dict[key]
    
    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._interface_cache.clear()
            self._pragma_cache.clear()
            self._analysis_cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "hit_rate": hit_rate,
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total_requests,
                "cache_sizes": {
                    "interface": len(self._interface_cache),
                    "pragma": len(self._pragma_cache),
                    "analysis": len(self._analysis_cache)
                },
                "total_size": (len(self._interface_cache) + 
                             len(self._pragma_cache) + 
                             len(self._analysis_cache))
            }


class LegacyAnalysisAdapter:
    """
    Adapter for legacy analysis compatibility.
    
    This adapter provides backward compatibility with existing
    analysis interfaces and data formats.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize adapter with configuration."""
        self.config = config or PipelineConfig()
    
    def adapt_legacy_interface_data(
        self,
        legacy_interfaces: List[Dict[str, Any]]
    ) -> List[RTLInterface]:
        """Convert legacy interface data to RTL interfaces."""
        rtl_interfaces = []
        
        for legacy_interface in legacy_interfaces:
            # Extract interface information
            name = legacy_interface.get("name", "unknown")
            interface_type = legacy_interface.get("type", "unknown")
            
            # Convert signals
            signals = []
            for signal_data in legacy_interface.get("signals", []):
                signal = RTLSignal(
                    name=signal_data.get("name", ""),
                    direction=signal_data.get("direction", "input"),
                    width=signal_data.get("width", 1),
                    interface_role=signal_data.get("role", "unknown")
                )
                signals.append(signal)
            
            # Create RTL interface
            rtl_interface = RTLInterface(
                name=name,
                interface_type=interface_type,
                signals=signals
            )
            rtl_interfaces.append(rtl_interface)
        
        return rtl_interfaces
    
    def adapt_legacy_pragma_data(
        self,
        legacy_pragmas: List[str]
    ) -> List[ParsedPragma]:
        """Convert legacy pragma strings to parsed pragmas."""
        processor = create_pragma_processor(self.config)
        return processor.parser.parse_pragma_list(legacy_pragmas)
    
    def adapt_to_legacy_format(
        self,
        analysis_results: AnalysisResults
    ) -> Dict[str, Any]:
        """Convert analysis results to legacy format."""
        legacy_results = {
            "interfaces": [],
            "pragmas": [],
            "success": analysis_results.success,
            "errors": analysis_results.errors,
            "warnings": analysis_results.warnings
        }
        
        # Convert interface results
        for interface_result in analysis_results.interface_results:
            legacy_interface = {
                "name": interface_result.interface_name,
                "type": interface_result.interface_type.value,
                "confidence": interface_result.confidence,
                "signals": [
                    {
                        "name": signal.name,
                        "direction": signal.direction,
                        "width": signal.width,
                        "role": signal.interface_role
                    }
                    for signal in interface_result.detected_signals
                ],
                "valid": interface_result.is_valid
            }
            legacy_results["interfaces"].append(legacy_interface)
        
        # Convert pragma results
        if analysis_results.pragma_results:
            for pragma in analysis_results.pragma_results.parsed_pragmas:
                legacy_pragma = {
                    "type": pragma.pragma_type.value,
                    "text": pragma.raw_text,
                    "line": pragma.line_number,
                    "directive": pragma.directive,
                    "parameters": pragma.parameters,
                    "valid": pragma.is_valid
                }
                legacy_results["pragmas"].append(legacy_pragma)
        
        return legacy_results


class AnalysisOrchestrator:
    """
    Main orchestrator for integrated interface and pragma analysis.
    
    This orchestrator coordinates all analysis activities, manages caching,
    provides error recovery, and integrates with the dataflow system.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize orchestrator with configuration."""
        self.config = config or PipelineConfig()
        
        # Create analysis configurations
        self.interface_config, self.pragma_config = create_analysis_config(
            self.config,
            profile="comprehensive" if self.config.is_dataflow_enabled() else "legacy_compatible"
        )
        
        # Create analyzers
        self.interface_analyzer = create_interface_analyzer(self.config)
        self.pragma_processor = create_pragma_processor(self.config)
        
        # Create cache and adapter
        self.cache = AnalysisCache(max_size=100, ttl=3600.0)
        self.legacy_adapter = LegacyAnalysisAdapter(self.config)
        
        # Statistics
        self._orchestration_count = 0
        self._total_orchestration_time = 0.0
    
    def analyze_rtl_module(
        self,
        rtl_module: RTLModule,
        pragma_sources: Union[List[str], str, Path] = None,
        enable_caching: bool = True
    ) -> AnalysisResults:
        """
        Perform complete analysis of an RTL module.
        
        Args:
            rtl_module: RTL module to analyze
            pragma_sources: Optional pragma sources
            enable_caching: Whether to use caching
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        
        # Create cache key
        cache_key = self._create_cache_key(rtl_module, pragma_sources)
        
        # Check cache if enabled
        if enable_caching and self.cache:
            cached_result = self.cache.get_analysis_result(cache_key)
            if cached_result:
                return cached_result
        
        # Create results container
        results = AnalysisResults(rtl_module=rtl_module)
        
        try:
            # Phase 1: Interface Analysis
            interface_start = time.time()
            results.interface_results = self.interface_analyzer.analyze_interfaces(rtl_module)
            results.interface_analysis_time = time.time() - interface_start
            
            # Phase 2: Pragma Processing
            if pragma_sources:
                pragma_start = time.time()
                results.pragma_results = self.pragma_processor.process_pragmas(
                    pragma_sources, rtl_module
                )
                results.pragma_processing_time = time.time() - pragma_start
            
            # Phase 3: Dataflow Integration
            if self.config.is_dataflow_enabled():
                self._perform_dataflow_integration(results)
            
            # Phase 4: Validation
            validation_start = time.time()
            results.overall_validation = self._perform_overall_validation(results)
            results.validation_time = time.time() - validation_start
            
        except Exception as e:
            results.add_error(f"Analysis failed: {str(e)}")
        
        # Finalize results
        results.finalize()
        
        # Cache results if enabled
        if enable_caching and self.cache and results.success:
            self.cache.put_analysis_result(cache_key, results)
        
        # Update statistics
        self._orchestration_count += 1
        self._total_orchestration_time += results.total_analysis_time
        
        return results
    
    def analyze_parsed_rtl_data(
        self,
        parsed_rtl: ParsedRTLData,
        pragma_sources: Union[List[str], str, Path] = None,
        enable_caching: bool = True
    ) -> Dict[str, AnalysisResults]:
        """Analyze all modules in parsed RTL data."""
        results = {}
        
        for module in parsed_rtl.modules:
            module_results = self.analyze_rtl_module(module, pragma_sources, enable_caching)
            results[module.name] = module_results
        
        return results
    
    def _perform_dataflow_integration(self, results: AnalysisResults) -> None:
        """Perform dataflow integration for analysis results."""
        if not DATAFLOW_AVAILABLE:
            return
        
        # Collect dataflow interfaces
        for interface_result in results.interface_results:
            if interface_result.dataflow_interface:
                results.dataflow_interfaces.append(interface_result.dataflow_interface)
        
        # Create dataflow model if we have interfaces
        if results.dataflow_interfaces:
            try:
                results.dataflow_model = DataflowModel(
                    results.dataflow_interfaces,
                    results.rtl_module.parameters
                )
            except Exception as e:
                results.add_warning(f"Failed to create dataflow model: {e}")
        
        # Create parallelism configuration from pragmas
        if results.pragma_results and results.pragma_results.dataflow_configuration:
            results.parallelism_configuration = results.pragma_results.dataflow_configuration
    
    def _perform_overall_validation(self, results: AnalysisResults) -> ValidationResult:
        """Perform overall validation of analysis results."""
        if DATAFLOW_AVAILABLE:
            validation_result = create_validation_result()
        else:
            validation_result = {"errors": [], "warnings": [], "success": True}
        
        # Validate interface analysis results
        for interface_result in results.interface_results:
            if not interface_result.is_valid and interface_result.validation_result:
                if DATAFLOW_AVAILABLE:
                    validation_result.merge(interface_result.validation_result)
                else:
                    validation_result["errors"].extend(
                        interface_result.validation_result.get("errors", [])
                    )
                    validation_result["warnings"].extend(
                        interface_result.validation_result.get("warnings", [])
                    )
        
        # Validate pragma results
        if results.pragma_results and not results.pragma_results.is_valid:
            if results.pragma_results.overall_validation:
                if DATAFLOW_AVAILABLE:
                    validation_result.merge(results.pragma_results.overall_validation)
                else:
                    validation_result["errors"].extend(
                        results.pragma_results.overall_validation.get("errors", [])
                    )
                    validation_result["warnings"].extend(
                        results.pragma_results.overall_validation.get("warnings", [])
                    )
        
        # Validate dataflow integration
        if results.dataflow_model:
            try:
                model_validation = results.dataflow_model.validate_mathematical_constraints()
                if DATAFLOW_AVAILABLE:
                    validation_result.merge(model_validation)
                else:
                    validation_result["errors"].extend(model_validation.get("errors", []))
                    validation_result["warnings"].extend(model_validation.get("warnings", []))
            except Exception as e:
                results.add_warning(f"Dataflow model validation failed: {e}")
        
        # Update success status
        if not DATAFLOW_AVAILABLE:
            validation_result["success"] = len(validation_result["errors"]) == 0
        
        return validation_result
    
    def _create_cache_key(
        self,
        rtl_module: RTLModule,
        pragma_sources: Union[List[str], str, Path] = None
    ) -> str:
        """Create cache key for analysis results."""
        import hashlib
        
        # Create content for hashing
        content_parts = [
            rtl_module.name,
            str(len(rtl_module.interfaces)),
            str(len(rtl_module.parameters)),
        ]
        
        if pragma_sources:
            if isinstance(pragma_sources, (str, Path)):
                content_parts.append(str(pragma_sources))
            elif isinstance(pragma_sources, list):
                content_parts.extend(pragma_sources)
        
        content = "|".join(content_parts)
        hash_obj = hashlib.md5(content.encode())
        return f"analysis_{hash_obj.hexdigest()[:16]}"
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get orchestration performance statistics."""
        return {
            "orchestration_count": self._orchestration_count,
            "total_orchestration_time": self._total_orchestration_time,
            "average_orchestration_time": (
                self._total_orchestration_time / self._orchestration_count
                if self._orchestration_count > 0 else 0.0
            ),
            "interface_analyzer_stats": self.interface_analyzer.get_analysis_statistics(),
            "pragma_processor_stats": self.pragma_processor.get_processing_statistics(),
            "cache_stats": self.cache.get_stats() if self.cache else None
        }
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        if self.cache:
            self.cache.clear()


# Factory functions
def create_analysis_orchestrator(config: PipelineConfig = None) -> AnalysisOrchestrator:
    """Create an analysis orchestrator with the given configuration."""
    return AnalysisOrchestrator(config)


def run_complete_analysis(
    rtl_module: RTLModule,
    pragma_sources: Union[List[str], str, Path] = None,
    config: PipelineConfig = None,
    enable_caching: bool = True
) -> AnalysisResults:
    """Convenience function to run complete analysis."""
    orchestrator = create_analysis_orchestrator(config)
    return orchestrator.analyze_rtl_module(rtl_module, pragma_sources, enable_caching)


def create_analysis_cache(max_size: int = 100, ttl: float = 3600.0) -> AnalysisCache:
    """Create an analysis cache with specified parameters."""
    return AnalysisCache(max_size, ttl)


def create_legacy_adapter(config: PipelineConfig = None) -> LegacyAnalysisAdapter:
    """Create a legacy analysis adapter."""
    return LegacyAnalysisAdapter(config)