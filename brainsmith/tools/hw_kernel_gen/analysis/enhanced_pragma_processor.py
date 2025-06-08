"""
Enhanced Pragma Processor with Dataflow Integration.

This module provides comprehensive pragma processing capabilities that can
parse, validate, and convert pragmas to dataflow constraints.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from pathlib import Path
from collections import defaultdict

# Import dataflow components
try:
    from ...dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
    from ...dataflow.core.dataflow_model import DataflowModel, ParallelismConfiguration
    from ...dataflow.core.validation import ValidationResult, create_validation_result
    from ...dataflow.core.tensor_chunking import ChunkingStrategy
    DATAFLOW_AVAILABLE = True
except ImportError:
    DATAFLOW_AVAILABLE = False
    # Create placeholder types
    class DataflowInterface: pass
    class DataflowInterfaceType: pass
    class DataflowModel: pass
    class ParallelismConfiguration: pass
    class ChunkingStrategy: pass
    ValidationResult = Dict[str, Any]

# Import Week 1 components
from ..enhanced_data_structures import RTLSignal, RTLInterface, RTLModule
from ..enhanced_config import PipelineConfig, AnalysisConfig
from ..errors import PragmaProcessingError, ValidationError

# Import analysis patterns
from .analysis_patterns import (
    PragmaType, PragmaPattern, get_pragma_patterns,
    create_custom_pragma_pattern
)


@dataclass
class ParsedPragma:
    """Result of pragma parsing."""
    
    # Basic information
    pragma_type: PragmaType
    raw_text: str
    line_number: int = 0
    
    # Parsed content
    directive: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)  # Referenced signals/parameters
    
    # Metadata
    parsing_time: float = 0.0
    pattern_match: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validation_result: Optional[ValidationResult] = None
    is_valid: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pragma to dictionary representation."""
        return {
            "pragma_type": self.pragma_type.value,
            "raw_text": self.raw_text,
            "line_number": self.line_number,
            "directive": self.directive,
            "parameters": self.parameters,
            "references": self.references,
            "parsing_time": self.parsing_time,
            "pattern_match": self.pattern_match,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
            "validation_success": (
                self.validation_result.success if DATAFLOW_AVAILABLE and self.validation_result
                else self.validation_result.get("success", True) if self.validation_result
                else None
            )
        }


@dataclass
class PragmaProcessingResult:
    """Result of complete pragma processing."""
    
    # Processed pragmas
    parsed_pragmas: List[ParsedPragma] = field(default_factory=list)
    ignored_pragmas: List[str] = field(default_factory=list)
    
    # Analysis results
    interface_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parallelism_constraints: Dict[str, Any] = field(default_factory=dict)
    dataflow_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    processing_time: float = 0.0
    pragma_count: int = 0
    valid_pragma_count: int = 0
    
    # Dataflow integration
    dataflow_configuration: Optional[ParallelismConfiguration] = None
    chunking_strategy: Optional[ChunkingStrategy] = None
    
    # Validation
    overall_validation: Optional[ValidationResult] = None
    is_valid: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "pragma_count": self.pragma_count,
            "valid_pragma_count": self.valid_pragma_count,
            "ignored_pragma_count": len(self.ignored_pragmas),
            "processing_time": self.processing_time,
            "interface_constraint_count": len(self.interface_constraints),
            "has_parallelism_constraints": bool(self.parallelism_constraints),
            "has_dataflow_constraints": bool(self.dataflow_constraints),
            "has_dataflow_configuration": self.dataflow_configuration is not None,
            "has_chunking_strategy": self.chunking_strategy is not None,
            "is_valid": self.is_valid,
            "parsed_pragmas": [pragma.to_dict() for pragma in self.parsed_pragmas]
        }


class PragmaParser:
    """
    Parse pragmas from various sources and formats.
    
    This parser can handle multiple pragma formats including Brainsmith-specific
    pragmas, HLS pragmas, and custom pragma formats.
    """
    
    def __init__(self, patterns: List[PragmaPattern] = None):
        """Initialize parser with patterns."""
        self.patterns = patterns or get_pragma_patterns()
        self._compiled_patterns: Dict[PragmaType, List[re.Pattern]] = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        for pattern in self.patterns:
            if pattern.pragma_type not in self._compiled_patterns:
                self._compiled_patterns[pattern.pragma_type] = []
            self._compiled_patterns[pattern.pragma_type].extend(pattern.compiled_patterns)
    
    def parse_pragma_text(self, pragma_text: str, line_number: int = 0) -> Optional[ParsedPragma]:
        """Parse a single pragma text line."""
        start_time = time.time()
        
        # Try each pattern type
        for pattern in self.patterns:
            for compiled_pattern in pattern.compiled_patterns:
                match = compiled_pattern.search(pragma_text)
                if match:
                    pragma = self._create_parsed_pragma(
                        pattern, pragma_text, match, line_number, start_time
                    )
                    return pragma
        
        return None
    
    def parse_pragma_file(self, file_path: Union[str, Path]) -> List[ParsedPragma]:
        """Parse pragmas from a file."""
        pragmas = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            return pragmas
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('//'):
                        continue
                    
                    pragma = self.parse_pragma_text(line, line_number)
                    if pragma:
                        pragmas.append(pragma)
        
        except Exception as e:
            # Log error but don't fail completely
            pass
        
        return pragmas
    
    def parse_pragma_list(self, pragma_texts: List[str]) -> List[ParsedPragma]:
        """Parse a list of pragma texts."""
        pragmas = []
        
        for i, pragma_text in enumerate(pragma_texts):
            pragma = self.parse_pragma_text(pragma_text, i + 1)
            if pragma:
                pragmas.append(pragma)
        
        return pragmas
    
    def _create_parsed_pragma(
        self,
        pattern: PragmaPattern,
        pragma_text: str,
        match: re.Match,
        line_number: int,
        start_time: float
    ) -> ParsedPragma:
        """Create a ParsedPragma from a pattern match."""
        pragma = ParsedPragma(
            pragma_type=pattern.pragma_type,
            raw_text=pragma_text,
            line_number=line_number,
            parsing_time=time.time() - start_time,
            pattern_match=match.group(0) if match.groups() else pragma_text
        )
        
        # Extract directive
        if match.groups():
            pragma.directive = match.group(1) if len(match.groups()) >= 1 else ""
        
        # Parse parameters using parameter patterns
        if len(match.groups()) >= 2:
            param_text = match.group(2)
            pragma.parameters = self._parse_parameters(pattern, param_text)
        
        # Extract references
        pragma.references = self._extract_references(pragma.parameters)
        
        return pragma
    
    def _parse_parameters(self, pattern: PragmaPattern, param_text: str) -> Dict[str, Any]:
        """Parse parameters from parameter text."""
        parameters = {}
        
        for param_name, param_pattern in pattern.parameter_patterns.items():
            compiled_param_pattern = re.compile(param_pattern, re.IGNORECASE)
            matches = compiled_param_pattern.findall(param_text)
            
            if matches:
                if len(matches) == 1 and isinstance(matches[0], str):
                    parameters[param_name] = matches[0]
                elif len(matches) == 1 and isinstance(matches[0], tuple):
                    # Multiple capture groups
                    parameters[param_name] = list(matches[0])
                else:
                    # Multiple matches
                    parameters[param_name] = matches
        
        return parameters
    
    def _extract_references(self, parameters: Dict[str, Any]) -> List[str]:
        """Extract signal/parameter references from parameters."""
        references = []
        
        for key, value in parameters.items():
            if isinstance(value, str):
                # Look for identifiers (potential signal/parameter names)
                identifier_pattern = re.compile(r'\b[a-zA-Z_]\w*\b')
                identifiers = identifier_pattern.findall(value)
                references.extend(identifiers)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        identifier_pattern = re.compile(r'\b[a-zA-Z_]\w*\b')
                        identifiers = identifier_pattern.findall(item)
                        references.extend(identifiers)
        
        # Remove duplicates and common keywords
        keywords = {'input', 'output', 'wire', 'reg', 'int', 'float', 'true', 'false'}
        references = list(set(ref for ref in references if ref.lower() not in keywords))
        
        return references


class PragmaValidator:
    """
    Validate pragma constraints and references.
    
    This validator checks that pragmas have correct syntax, valid parameter
    references, and consistent constraints.
    """
    
    def __init__(self, config: AnalysisConfig = None):
        """Initialize validator with configuration."""
        self.config = config or AnalysisConfig()
    
    def validate_pragma(
        self,
        pragma: ParsedPragma,
        rtl_module: RTLModule = None,
        pattern: PragmaPattern = None
    ) -> ValidationResult:
        """Validate a single pragma."""
        if DATAFLOW_AVAILABLE:
            result = create_validation_result()
        else:
            result = {"errors": [], "warnings": [], "success": True}
        
        # Find the pattern if not provided
        if not pattern:
            patterns = get_pragma_patterns()
            pattern = next((p for p in patterns if p.pragma_type == pragma.pragma_type), None)
        
        if pattern:
            # Validate required parameters
            self._validate_required_parameters(result, pragma, pattern)
            
            # Validate parameter types and values
            self._validate_parameter_values(result, pragma, pattern)
            
            # Validate references
            if rtl_module:
                self._validate_references(result, pragma, rtl_module)
        
        return result
    
    def validate_pragma_consistency(
        self,
        pragmas: List[ParsedPragma],
        rtl_module: RTLModule = None
    ) -> ValidationResult:
        """Validate consistency across multiple pragmas."""
        if DATAFLOW_AVAILABLE:
            result = create_validation_result()
        else:
            result = {"errors": [], "warnings": [], "success": True}
        
        # Check for conflicting constraints
        self._check_conflicting_constraints(result, pragmas)
        
        # Check for duplicate pragma definitions
        self._check_duplicate_pragmas(result, pragmas)
        
        # Check interface coverage
        if rtl_module:
            self._check_interface_coverage(result, pragmas, rtl_module)
        
        return result
    
    def _validate_required_parameters(
        self,
        result: ValidationResult,
        pragma: ParsedPragma,
        pattern: PragmaPattern
    ) -> None:
        """Validate that all required parameters are present."""
        missing_params = pattern.required_parameters - set(pragma.parameters.keys())
        
        if missing_params:
            error_msg = f"Pragma '{pragma.directive}' missing required parameters: {list(missing_params)}"
            self._add_error(result, "validation", "missing_parameters", error_msg, {
                "pragma_type": pragma.pragma_type.value,
                "missing_parameters": list(missing_params)
            })
    
    def _validate_parameter_values(
        self,
        result: ValidationResult,
        pragma: ParsedPragma,
        pattern: PragmaPattern
    ) -> None:
        """Validate parameter values and types."""
        for param_name, param_value in pragma.parameters.items():
            # Check if parameter is recognized
            if (param_name not in pattern.required_parameters and 
                param_name not in pattern.optional_parameters):
                warning_msg = f"Unknown parameter '{param_name}' in pragma '{pragma.directive}'"
                self._add_warning(result, "validation", "unknown_parameter", warning_msg)
    
    def _validate_references(
        self,
        result: ValidationResult,
        pragma: ParsedPragma,
        rtl_module: RTLModule
    ) -> None:
        """Validate that referenced signals/parameters exist."""
        # Get all available names
        available_names = set()
        available_names.update(rtl_module.parameters.keys())
        
        for interface in rtl_module.interfaces:
            available_names.add(interface.name)
            for signal in interface.signals:
                available_names.add(signal.name)
        
        # Check references
        for ref in pragma.references:
            if ref not in available_names:
                warning_msg = f"Pragma '{pragma.directive}' references unknown identifier '{ref}'"
                self._add_warning(result, "validation", "unknown_reference", warning_msg)
    
    def _check_conflicting_constraints(
        self,
        result: ValidationResult,
        pragmas: List[ParsedPragma]
    ) -> None:
        """Check for conflicting pragma constraints."""
        # Group pragmas by directive
        pragma_groups = defaultdict(list)
        for pragma in pragmas:
            pragma_groups[pragma.directive].append(pragma)
        
        # Check for conflicts within groups
        for directive, pragma_list in pragma_groups.items():
            if len(pragma_list) > 1:
                # Check if multiple definitions are conflicting
                self._check_directive_conflicts(result, directive, pragma_list)
    
    def _check_directive_conflicts(
        self,
        result: ValidationResult,
        directive: str,
        pragmas: List[ParsedPragma]
    ) -> None:
        """Check for conflicts within a directive group."""
        # For interface pragmas, check for conflicting interface definitions
        if directive == "interface":
            interface_defs = {}
            for pragma in pragmas:
                if "interface" in pragma.parameters:
                    interface_name = pragma.parameters["interface"]
                    if isinstance(interface_name, list) and len(interface_name) >= 2:
                        name = interface_name[1]
                        if name in interface_defs:
                            error_msg = f"Conflicting interface definitions for '{name}'"
                            self._add_error(result, "validation", "conflicting_definitions", error_msg)
                        interface_defs[name] = pragma
    
    def _check_duplicate_pragmas(
        self,
        result: ValidationResult,
        pragmas: List[ParsedPragma]
    ) -> None:
        """Check for duplicate pragma definitions."""
        seen_pragmas = set()
        
        for pragma in pragmas:
            pragma_key = (pragma.pragma_type, pragma.directive, tuple(sorted(pragma.parameters.items())))
            
            if pragma_key in seen_pragmas:
                warning_msg = f"Duplicate pragma definition: {pragma.directive}"
                self._add_warning(result, "validation", "duplicate_pragma", warning_msg)
            
            seen_pragmas.add(pragma_key)
    
    def _check_interface_coverage(
        self,
        result: ValidationResult,
        pragmas: List[ParsedPragma],
        rtl_module: RTLModule
    ) -> None:
        """Check if all interfaces have appropriate pragma coverage."""
        # Get interfaces defined in pragmas
        pragma_interfaces = set()
        for pragma in pragmas:
            if pragma.directive == "interface" and "interface" in pragma.parameters:
                interface_param = pragma.parameters["interface"]
                if isinstance(interface_param, list) and len(interface_param) >= 2:
                    pragma_interfaces.add(interface_param[1])
        
        # Get actual interfaces
        actual_interfaces = {interface.name for interface in rtl_module.interfaces}
        
        # Check for missing pragma definitions
        missing_pragmas = actual_interfaces - pragma_interfaces
        if missing_pragmas and self.config.validate_pragma_compatibility:
            warning_msg = f"Interfaces without pragma definitions: {list(missing_pragmas)}"
            self._add_warning(result, "validation", "missing_pragma_coverage", warning_msg)
    
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


class DataflowPragmaConverter:
    """
    Convert pragmas to dataflow constraints and configurations.
    
    This converter transforms pragma-based constraints into dataflow
    modeling constraints that can be used for optimization and validation.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize converter with configuration."""
        self.config = config
    
    def convert_pragmas_to_dataflow(
        self,
        pragmas: List[ParsedPragma],
        rtl_module: RTLModule = None
    ) -> Dict[str, Any]:
        """Convert pragmas to dataflow constraints."""
        constraints = {
            "interface_constraints": {},
            "parallelism_constraints": {},
            "chunking_constraints": {},
            "optimization_constraints": {}
        }
        
        for pragma in pragmas:
            if pragma.pragma_type == PragmaType.INTERFACE:
                self._convert_interface_pragma(pragma, constraints["interface_constraints"])
            elif pragma.pragma_type == PragmaType.PARALLELISM:
                self._convert_parallelism_pragma(pragma, constraints["parallelism_constraints"])
            elif pragma.pragma_type == PragmaType.DATAFLOW:
                self._convert_dataflow_pragma(pragma, constraints)
        
        return constraints
    
    def create_parallelism_configuration(
        self,
        pragmas: List[ParsedPragma]
    ) -> Optional[ParallelismConfiguration]:
        """Create dataflow parallelism configuration from pragmas."""
        if not DATAFLOW_AVAILABLE:
            return None
        
        ipar = {}
        wpar = {}
        opar = {}
        
        for pragma in pragmas:
            if pragma.pragma_type == PragmaType.PARALLELISM:
                self._extract_parallelism_params(pragma, ipar, wpar, opar)
        
        if ipar or wpar or opar:
            return ParallelismConfiguration(ipar, wpar, opar)
        
        return None
    
    def _convert_interface_pragma(
        self,
        pragma: ParsedPragma,
        interface_constraints: Dict[str, Any]
    ) -> None:
        """Convert interface pragma to constraints."""
        if "interface" in pragma.parameters:
            interface_param = pragma.parameters["interface"]
            if isinstance(interface_param, list) and len(interface_param) >= 2:
                interface_type = interface_param[0]
                interface_name = interface_param[1]
                
                interface_constraints[interface_name] = {
                    "type": interface_type,
                    "pragma_line": pragma.line_number,
                    "parameters": pragma.parameters
                }
    
    def _convert_parallelism_pragma(
        self,
        pragma: ParsedPragma,
        parallelism_constraints: Dict[str, Any]
    ) -> None:
        """Convert parallelism pragma to constraints."""
        if "parallelism" in pragma.parameters:
            param_value = pragma.parameters["parallelism"]
            if isinstance(param_value, list) and len(param_value) >= 2:
                param_name = param_value[0]
                param_val = param_value[1]
                
                try:
                    parallelism_constraints[param_name] = int(param_val)
                except ValueError:
                    parallelism_constraints[param_name] = param_val
    
    def _convert_dataflow_pragma(
        self,
        pragma: ParsedPragma,
        constraints: Dict[str, Any]
    ) -> None:
        """Convert dataflow pragma to constraints."""
        if "dataflow" in pragma.parameters:
            param_value = pragma.parameters["dataflow"]
            if isinstance(param_value, list) and len(param_value) >= 2:
                constraint_type = param_value[0]
                constraint_value = param_value[1]
                
                if constraint_type == "chunking":
                    constraints["chunking_constraints"][constraint_type] = constraint_value
                else:
                    constraints["optimization_constraints"][constraint_type] = constraint_value
    
    def _extract_parallelism_params(
        self,
        pragma: ParsedPragma,
        ipar: Dict[str, int],
        wpar: Dict[str, int],
        opar: Dict[str, int]
    ) -> None:
        """Extract parallelism parameters for dataflow configuration."""
        for param_name, param_value in pragma.parameters.items():
            if isinstance(param_value, str) and param_value.isdigit():
                value = int(param_value)
                
                if param_name.startswith("input") or param_name.endswith("_ipar"):
                    ipar[param_name] = value
                elif param_name.startswith("weight") or param_name.endswith("_wpar"):
                    wpar[param_name] = value
                elif param_name.startswith("output") or param_name.endswith("_opar"):
                    opar[param_name] = value


class PragmaProcessor:
    """
    Main pragma processing engine.
    
    This processor orchestrates the pragma parsing, validation,
    and dataflow conversion processes.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize processor with configuration."""
        self.config = config or PipelineConfig()
        self.parser = PragmaParser()
        self.validator = PragmaValidator(self.config.analysis)
        self.converter = DataflowPragmaConverter(self.config)
        
        # Processing state
        self._processing_count = 0
        self._total_processing_time = 0.0
    
    def process_pragmas(
        self,
        pragma_sources: Union[List[str], str, Path],
        rtl_module: RTLModule = None
    ) -> PragmaProcessingResult:
        """
        Process pragmas from various sources.
        
        Args:
            pragma_sources: Pragma texts, file path, or list of pragma texts
            rtl_module: RTL module for reference validation
            
        Returns:
            Complete pragma processing result
        """
        start_time = time.time()
        
        # Parse pragmas
        if isinstance(pragma_sources, (str, Path)):
            if Path(pragma_sources).exists():
                parsed_pragmas = self.parser.parse_pragma_file(pragma_sources)
            else:
                # Treat as single pragma text
                pragma = self.parser.parse_pragma_text(str(pragma_sources))
                parsed_pragmas = [pragma] if pragma else []
        elif isinstance(pragma_sources, list):
            parsed_pragmas = self.parser.parse_pragma_list(pragma_sources)
        else:
            parsed_pragmas = []
        
        # Create result
        result = PragmaProcessingResult(
            parsed_pragmas=parsed_pragmas,
            pragma_count=len(parsed_pragmas),
            processing_time=time.time() - start_time
        )
        
        # Validate pragmas
        valid_pragmas = []
        for pragma in parsed_pragmas:
            validation_result = self.validator.validate_pragma(pragma, rtl_module)
            pragma.validation_result = validation_result
            pragma.is_valid = (validation_result.success if DATAFLOW_AVAILABLE 
                             else validation_result.get("success", True))
            
            if pragma.is_valid:
                valid_pragmas.append(pragma)
        
        result.valid_pragma_count = len(valid_pragmas)
        
        # Validate consistency
        if valid_pragmas:
            overall_validation = self.validator.validate_pragma_consistency(valid_pragmas, rtl_module)
            result.overall_validation = overall_validation
            result.is_valid = (overall_validation.success if DATAFLOW_AVAILABLE 
                             else overall_validation.get("success", True))
        
        # Convert to dataflow constraints
        if self.config.is_dataflow_enabled() and valid_pragmas:
            constraints = self.converter.convert_pragmas_to_dataflow(valid_pragmas, rtl_module)
            result.interface_constraints = constraints.get("interface_constraints", {})
            result.parallelism_constraints = constraints.get("parallelism_constraints", {})
            result.dataflow_constraints = constraints.get("optimization_constraints", {})
            
            # Create dataflow configuration
            result.dataflow_configuration = self.converter.create_parallelism_configuration(valid_pragmas)
        
        # Update statistics
        self._processing_count += 1
        self._total_processing_time += result.processing_time
        
        return result
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing performance statistics."""
        return {
            "processing_count": self._processing_count,
            "total_processing_time": self._total_processing_time,
            "average_processing_time": (
                self._total_processing_time / self._processing_count 
                if self._processing_count > 0 else 0.0
            )
        }


# Factory functions
def create_pragma_processor(config: PipelineConfig = None) -> PragmaProcessor:
    """Create a pragma processor with the given configuration."""
    return PragmaProcessor(config)


def process_pragmas(
    pragma_sources: Union[List[str], str, Path],
    rtl_module: RTLModule = None,
    config: PipelineConfig = None
) -> PragmaProcessingResult:
    """Convenience function to process pragmas."""
    processor = create_pragma_processor(config)
    return processor.process_pragmas(pragma_sources, rtl_module)


def create_pragma_parser(patterns: List[PragmaPattern] = None) -> PragmaParser:
    """Create a pragma parser with custom patterns."""
    return PragmaParser(patterns)


def create_pragma_validator(config: AnalysisConfig = None) -> PragmaValidator:
    """Create a pragma validator with configuration."""
    return PragmaValidator(config)