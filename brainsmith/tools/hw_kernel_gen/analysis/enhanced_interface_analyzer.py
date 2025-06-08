"""
Enhanced Interface Analyzer with Dataflow Integration.

This module provides comprehensive interface analysis capabilities that can
detect, classify, and convert RTL interfaces to dataflow representations.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from pathlib import Path
from collections import defaultdict

# Import dataflow components
try:
    from ...dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
    from ...dataflow.core.dataflow_model import DataflowModel
    from ...dataflow.core.validation import ValidationResult, create_validation_result
    from ...dataflow.integration.rtl_conversion import RTLInterfaceConverter
    DATAFLOW_AVAILABLE = True
except ImportError:
    DATAFLOW_AVAILABLE = False
    # Create placeholder types
    class DataflowInterface: pass
    class DataflowInterfaceType: pass
    class DataflowModel: pass
    ValidationResult = Dict[str, Any]

# Import Week 1 components
from ..enhanced_data_structures import RTLSignal, RTLInterface, RTLModule
from ..enhanced_config import PipelineConfig, AnalysisConfig
from ..errors import InterfaceDetectionError, ValidationError

# Import analysis patterns
from .analysis_patterns import (
    InterfaceType, SignalRole, InterfacePattern, SignalPattern,
    get_interface_patterns, create_custom_interface_pattern
)


@dataclass
class InterfaceAnalysisResult:
    """Result of interface analysis."""
    
    # Basic information
    interface_name: str
    interface_type: InterfaceType
    confidence: float = 0.0
    
    # Signals
    detected_signals: List[RTLSignal] = field(default_factory=list)
    missing_signals: List[SignalRole] = field(default_factory=list)
    unknown_signals: List[RTLSignal] = field(default_factory=list)
    
    # Analysis metadata
    pattern_matches: List[str] = field(default_factory=list)
    analysis_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Dataflow integration
    dataflow_interface: Optional[DataflowInterface] = None
    conversion_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validation_result: Optional[ValidationResult] = None
    is_valid: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "interface_name": self.interface_name,
            "interface_type": self.interface_type.value,
            "confidence": self.confidence,
            "signal_count": len(self.detected_signals),
            "missing_signal_count": len(self.missing_signals),
            "unknown_signal_count": len(self.unknown_signals),
            "pattern_matches": self.pattern_matches,
            "analysis_time": self.analysis_time,
            "metadata": self.metadata,
            "has_dataflow_interface": self.dataflow_interface is not None,
            "conversion_metadata": self.conversion_metadata,
            "is_valid": self.is_valid,
            "validation_success": (
                self.validation_result.success if DATAFLOW_AVAILABLE and self.validation_result
                else self.validation_result.get("success", True) if self.validation_result
                else None
            )
        }


class InterfaceClassifier:
    """
    Classify interfaces based on signal patterns and naming conventions.
    
    This classifier uses pattern matching to identify different types of
    interfaces (AXI-Stream, AXI-Lite, control, etc.) and provides confidence
    scores for the classifications.
    """
    
    def __init__(self, patterns: List[InterfacePattern] = None):
        """Initialize classifier with patterns."""
        self.patterns = patterns or get_interface_patterns()
        self._pattern_cache: Dict[str, List[InterfacePattern]] = {}
        
    def classify_interface(
        self,
        interface_name: str,
        signals: List[RTLSignal]
    ) -> List[Tuple[InterfaceType, float]]:
        """
        Classify interface and return possible types with confidence scores.
        
        Args:
            interface_name: Name of the interface
            signals: List of signals in the interface
            
        Returns:
            List of (InterfaceType, confidence) tuples sorted by confidence
        """
        candidates = []
        
        for pattern in self.patterns:
            confidence = self._calculate_confidence(pattern, interface_name, signals)
            if confidence > 0.0:
                candidates.append((pattern.interface_type, confidence))
        
        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def get_best_classification(
        self,
        interface_name: str,
        signals: List[RTLSignal]
    ) -> Tuple[InterfaceType, float]:
        """Get the best classification for an interface."""
        candidates = self.classify_interface(interface_name, signals)
        
        if candidates:
            return candidates[0]
        else:
            return InterfaceType.UNKNOWN, 0.0
    
    def _calculate_confidence(
        self,
        pattern: InterfacePattern,
        interface_name: str,
        signals: List[RTLSignal]
    ) -> float:
        """Calculate confidence score for a pattern match."""
        confidence = 0.0
        
        # Check naming patterns
        name_score = 0.0
        if pattern.matches_prefix(interface_name):
            name_score += 0.3
        if pattern.matches_suffix(interface_name):
            name_score += 0.2
        
        # Check signal patterns
        signal_scores = []
        detected_signals = set()
        
        for signal_pattern in pattern.signal_patterns:
            matches = self._find_matching_signals(signal_pattern, signals)
            
            if matches:
                detected_signals.add(signal_pattern.role)
                # Weight required signals more heavily
                weight = 1.0 if signal_pattern.required else 0.5
                signal_scores.append(weight)
        
        # Check completeness
        required_signals = {sp.role for sp in pattern.signal_patterns if sp.required}
        missing_required = required_signals - detected_signals
        
        if missing_required:
            # Penalize missing required signals
            completeness_penalty = len(missing_required) / len(required_signals) if required_signals else 0
            signal_score = (sum(signal_scores) / len(pattern.signal_patterns)) * (1.0 - completeness_penalty)
        else:
            signal_score = sum(signal_scores) / len(pattern.signal_patterns) if pattern.signal_patterns else 0.5
        
        # Check minimum signal count
        count_score = 1.0 if len(signals) >= pattern.min_signals else len(signals) / pattern.min_signals
        
        # Combine scores
        confidence = (name_score * 0.3 + signal_score * 0.6 + count_score * 0.1)
        
        return min(confidence, 1.0)
    
    def _find_matching_signals(
        self,
        signal_pattern: SignalPattern,
        signals: List[RTLSignal]
    ) -> List[RTLSignal]:
        """Find signals that match a signal pattern."""
        matches = []
        
        for signal in signals:
            if signal_pattern.matches(signal.name):
                # Check direction if specified
                if signal_pattern.direction and signal.direction != signal_pattern.direction:
                    continue
                
                # Check width range if specified
                if signal_pattern.width_range:
                    min_width, max_width = signal_pattern.width_range
                    if not (min_width <= signal.width <= max_width):
                        continue
                
                matches.append(signal)
        
        return matches


class InterfaceValidator:
    """
    Validate interface completeness and consistency.
    
    This validator checks that interfaces have all required signals,
    proper signal directions, and consistent naming conventions.
    """
    
    def __init__(self, config: AnalysisConfig = None):
        """Initialize validator with configuration."""
        self.config = config or AnalysisConfig()
        
    def validate_interface(
        self,
        interface: RTLInterface,
        interface_type: InterfaceType,
        pattern: InterfacePattern = None
    ) -> ValidationResult:
        """Validate an interface against its type requirements."""
        if DATAFLOW_AVAILABLE:
            result = create_validation_result()
        else:
            result = {"errors": [], "warnings": [], "success": True}
        
        if not pattern:
            # Find matching pattern
            patterns = get_interface_patterns()
            pattern = next((p for p in patterns if p.interface_type == interface_type), None)
            
            if not pattern:
                error_msg = f"No validation pattern found for interface type: {interface_type}"
                self._add_error(result, "validation", "missing_pattern", error_msg)
                return result
        
        # Validate required signals
        self._validate_required_signals(result, interface, pattern)
        
        # Validate signal properties
        self._validate_signal_properties(result, interface, pattern)
        
        # Validate naming consistency
        self._validate_naming_consistency(result, interface, pattern)
        
        return result
    
    def _validate_required_signals(
        self,
        result: ValidationResult,
        interface: RTLInterface,
        pattern: InterfacePattern
    ) -> None:
        """Validate that all required signals are present."""
        classifier = InterfaceClassifier([pattern])
        signal_roles = set()
        
        for signal_pattern in pattern.signal_patterns:
            matches = classifier._find_matching_signals(signal_pattern, interface.signals)
            if matches:
                signal_roles.add(signal_pattern.role)
        
        missing_required = pattern.required_signals - signal_roles
        
        if missing_required:
            error_msg = f"Interface '{interface.name}' missing required signals: {[role.value for role in missing_required]}"
            self._add_error(result, "validation", "missing_signals", error_msg, {
                "interface_type": pattern.interface_type.value,
                "missing_signals": [role.value for role in missing_required]
            })
    
    def _validate_signal_properties(
        self,
        result: ValidationResult,
        interface: RTLInterface,
        pattern: InterfacePattern
    ) -> None:
        """Validate signal properties like width and direction."""
        classifier = InterfaceClassifier([pattern])
        
        for signal_pattern in pattern.signal_patterns:
            matches = classifier._find_matching_signals(signal_pattern, interface.signals)
            
            for signal in matches:
                # Validate direction
                if signal_pattern.direction and signal.direction != signal_pattern.direction:
                    warning_msg = f"Signal '{signal.name}' has direction '{signal.direction}' but expected '{signal_pattern.direction}'"
                    self._add_warning(result, "validation", "direction_mismatch", warning_msg)
                
                # Validate width
                if signal_pattern.width_range:
                    min_width, max_width = signal_pattern.width_range
                    if not (min_width <= signal.width <= max_width):
                        warning_msg = f"Signal '{signal.name}' width {signal.width} not in expected range [{min_width}, {max_width}]"
                        self._add_warning(result, "validation", "width_mismatch", warning_msg)
    
    def _validate_naming_consistency(
        self,
        result: ValidationResult,
        interface: RTLInterface,
        pattern: InterfacePattern
    ) -> None:
        """Validate naming consistency within the interface."""
        # Check prefix consistency
        if pattern.prefix_patterns:
            matching_prefixes = [prefix for prefix in pattern.prefix_patterns 
                               if interface.name.startswith(prefix)]
            
            if not matching_prefixes:
                warning_msg = f"Interface '{interface.name}' does not follow expected naming convention"
                self._add_warning(result, "validation", "naming_convention", warning_msg)
    
    def _add_error(
        self,
        result: ValidationResult,
        component: str,
        error_type: str,
        message: str,
        context: Dict[str, Any] = None
    ) -> None:
        """Add error to validation result."""
        if DATAFLOW_AVAILABLE:
            from ...dataflow.core.validation import ValidationError, ValidationSeverity
            error = ValidationError(
                component=component,
                error_type=error_type,
                message=message,
                severity=ValidationSeverity.ERROR,
                context=context or {}
            )
            result.add_error(error)
        else:
            error = {
                "component": component,
                "error_type": error_type,
                "message": message,
                "context": context or {}
            }
            result["errors"].append(error)
            result["success"] = False
    
    def _add_warning(
        self,
        result: ValidationResult,
        component: str,
        warning_type: str,
        message: str,
        context: Dict[str, Any] = None
    ) -> None:
        """Add warning to validation result."""
        if DATAFLOW_AVAILABLE:
            from ...dataflow.core.validation import ValidationError, ValidationSeverity
            warning = ValidationError(
                component=component,
                error_type=warning_type,
                message=message,
                severity=ValidationSeverity.WARNING,
                context=context or {}
            )
            result.add_warning(warning)
        else:
            warning = {
                "component": component,
                "warning_type": warning_type,
                "message": message,
                "context": context or {}
            }
            result["warnings"].append(warning)


class DataflowInterfaceConverter:
    """
    Convert RTL interfaces to dataflow interfaces.
    
    This converter bridges the gap between RTL signal-level descriptions
    and dataflow tensor-level descriptions.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize converter with configuration."""
        self.config = config
        self._converter: Optional[RTLInterfaceConverter] = None
        
        if DATAFLOW_AVAILABLE and config and config.is_dataflow_enabled():
            self._converter = RTLInterfaceConverter(
                onnx_metadata=config.dataflow.onnx_metadata
            )
    
    def convert_interface(
        self,
        rtl_interface: RTLInterface,
        interface_type: InterfaceType,
        metadata: Dict[str, Any] = None
    ) -> Optional[DataflowInterface]:
        """Convert RTL interface to dataflow interface."""
        if not DATAFLOW_AVAILABLE or not self._converter:
            return None
        
        try:
            # Convert using the dataflow converter
            dataflow_interfaces = self._converter.convert_interfaces(
                {rtl_interface.name: rtl_interface},
                parameters=metadata or {}
            )
            
            if dataflow_interfaces:
                return dataflow_interfaces[0]
            
        except Exception as e:
            # Log conversion error but don't fail analysis
            pass
        
        return None
    
    def infer_dataflow_type(
        self,
        interface_type: InterfaceType,
        rtl_interface: RTLInterface
    ) -> Optional[DataflowInterfaceType]:
        """Infer dataflow interface type from RTL interface type."""
        if not DATAFLOW_AVAILABLE:
            return None
        
        # Map RTL interface types to dataflow types
        type_mapping = {
            InterfaceType.AXI_STREAM: DataflowInterfaceType.INPUT,  # Default, can be INPUT or OUTPUT
            InterfaceType.CONTROL: DataflowInterfaceType.CONTROL,
        }
        
        base_type = type_mapping.get(interface_type)
        
        if base_type == DataflowInterfaceType.INPUT and interface_type == InterfaceType.AXI_STREAM:
            # Determine if it's input or output based on signal directions
            tdata_signals = [s for s in rtl_interface.signals if "tdata" in s.name.lower()]
            if tdata_signals:
                signal = tdata_signals[0]
                if signal.direction == "output":
                    return DataflowInterfaceType.OUTPUT
                elif signal.direction == "input":
                    return DataflowInterfaceType.INPUT
        
        return base_type


class InterfaceAnalyzer:
    """
    Main interface analysis engine.
    
    This analyzer orchestrates the interface detection, classification,
    validation, and dataflow conversion processes.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize analyzer with configuration."""
        self.config = config or PipelineConfig()
        self.classifier = InterfaceClassifier()
        self.validator = InterfaceValidator(self.config.analysis)
        self.converter = DataflowInterfaceConverter(self.config)
        
        # Analysis state
        self._analysis_count = 0
        self._total_analysis_time = 0.0
    
    def analyze_interfaces(
        self,
        rtl_module: RTLModule,
        custom_patterns: List[InterfacePattern] = None
    ) -> List[InterfaceAnalysisResult]:
        """
        Analyze all interfaces in an RTL module.
        
        Args:
            rtl_module: RTL module to analyze
            custom_patterns: Optional custom interface patterns
            
        Returns:
            List of interface analysis results
        """
        start_time = time.time()
        results = []
        
        # Update classifier with custom patterns if provided
        if custom_patterns:
            self.classifier.patterns.extend(custom_patterns)
        
        # Group signals by interface
        interface_groups = self._group_signals_by_interface(rtl_module.interfaces)
        
        # Analyze each interface
        for interface in interface_groups:
            result = self._analyze_single_interface(interface)
            results.append(result)
        
        # Update statistics
        analysis_time = time.time() - start_time
        self._analysis_count += 1
        self._total_analysis_time += analysis_time
        
        return results
    
    def analyze_single_interface(
        self,
        rtl_interface: RTLInterface
    ) -> InterfaceAnalysisResult:
        """Analyze a single RTL interface."""
        return self._analyze_single_interface(rtl_interface)
    
    def _analyze_single_interface(
        self,
        rtl_interface: RTLInterface
    ) -> InterfaceAnalysisResult:
        """Internal method to analyze a single interface."""
        start_time = time.time()
        
        # Classify interface
        interface_type, confidence = self.classifier.get_best_classification(
            rtl_interface.name,
            rtl_interface.signals
        )
        
        # Create analysis result
        result = InterfaceAnalysisResult(
            interface_name=rtl_interface.name,
            interface_type=interface_type,
            confidence=confidence,
            detected_signals=rtl_interface.signals.copy(),
            analysis_time=time.time() - start_time
        )
        
        # Validate interface
        if self.config.validation.validate_interface_constraints:
            validation_result = self.validator.validate_interface(
                rtl_interface,
                interface_type
            )
            result.validation_result = validation_result
            result.is_valid = (validation_result.success if DATAFLOW_AVAILABLE 
                             else validation_result.get("success", True))
        
        # Convert to dataflow interface if enabled
        if self.config.is_dataflow_enabled():
            dataflow_interface = self.converter.convert_interface(
                rtl_interface,
                interface_type,
                result.metadata
            )
            result.dataflow_interface = dataflow_interface
            
            if dataflow_interface:
                result.conversion_metadata = {
                    "dataflow_type": dataflow_interface.interface_type.value,
                    "qDim": dataflow_interface.qDim,
                    "tDim": dataflow_interface.tDim,
                    "sDim": dataflow_interface.sDim
                }
        
        return result
    
    def _group_signals_by_interface(
        self,
        interfaces: List[RTLInterface]
    ) -> List[RTLInterface]:
        """Group signals by interface (already grouped in RTLInterface objects)."""
        return interfaces
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis performance statistics."""
        return {
            "analysis_count": self._analysis_count,
            "total_analysis_time": self._total_analysis_time,
            "average_analysis_time": (
                self._total_analysis_time / self._analysis_count 
                if self._analysis_count > 0 else 0.0
            )
        }


# Factory functions
def create_interface_analyzer(config: PipelineConfig = None) -> InterfaceAnalyzer:
    """Create an interface analyzer with the given configuration."""
    return InterfaceAnalyzer(config)


def analyze_interfaces(
    rtl_module: RTLModule,
    config: PipelineConfig = None,
    custom_patterns: List[InterfacePattern] = None
) -> List[InterfaceAnalysisResult]:
    """Convenience function to analyze interfaces in an RTL module."""
    analyzer = create_interface_analyzer(config)
    return analyzer.analyze_interfaces(rtl_module, custom_patterns)


def create_interface_classifier(patterns: List[InterfacePattern] = None) -> InterfaceClassifier:
    """Create an interface classifier with custom patterns."""
    return InterfaceClassifier(patterns)


def create_interface_validator(config: AnalysisConfig = None) -> InterfaceValidator:
    """Create an interface validator with configuration."""
    return InterfaceValidator(config)