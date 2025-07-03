"""
Blueprint parser for DSE V3 - Refactored version.

This module handles parsing Blueprint YAML files into structured DesignSpace objects.
It supports various kernel and transform formats including simple strings, tuples,
mutually exclusive groups, and optional elements.
"""

import yaml
from typing import Dict, List, Any, Union, Optional, Type, Callable
from dataclasses import dataclass
from functools import wraps
import logging

from .data_structures import (
    DesignSpace,
    HWCompilerSpace,
    ProcessingSpace,
    ProcessingStep,
    SearchConfig,
    SearchConstraint,
    SearchStrategy,
    GlobalConfig,
    OutputStage,
)
from .exceptions import BlueprintParseError
from ..config import get_config
from ..plugins import get_registry


logger = logging.getLogger(__name__)


# Removed FieldSpec - not used in this refactoring


def wrap_parse_errors(section_name: str):
    """Decorator to wrap parsing errors with consistent context."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BlueprintParseError:
                raise
            except Exception as e:
                raise BlueprintParseError(f"Error parsing {section_name}: {str(e)}")
        return wrapper
    return decorator


class BlueprintParser:
    """
    Parse Blueprint YAML files into DesignSpace objects.
    
    This parser handles the V3 blueprint format with support for:
    - Simple kernel names that auto-import all backends
    - Explicit kernel backend lists
    - Mutually exclusive kernel/transform groups
    - Optional elements marked with ~
    - Both flat and phase-based transform organization
    """
    
    SUPPORTED_VERSION = "3.0"
    
    def __init__(self):
        """Initialize parser with plugin registry for validation."""
        self.plugin_registry = get_registry()
    
    def parse(self, blueprint_data: Dict[str, Any], model_path: str) -> DesignSpace:
        """
        Parse blueprint data into a DesignSpace object.
        
        Args:
            blueprint_data: Parsed YAML data from blueprint file
            model_path: Path to the ONNX model
            
        Returns:
            Fully constructed DesignSpace object
            
        Raises:
            BlueprintParseError: If parsing fails
        """
        try:
            # Validate version
            self._validate_version(blueprint_data)
            
            # Parse each section
            hw_compiler_space = self._parse_hw_compiler(
                blueprint_data.get("hw_compiler", {})
            )
            processing_space = self._parse_processing(
                blueprint_data.get("processing", {})
            )
            search_config = self._parse_search(
                blueprint_data.get("search", {})
            )
            global_config = self._parse_global(
                blueprint_data.get("global", {})
            )
            
            # Resolve effective timeout with proper priority
            # Priority: search.timeout_minutes > global.timeout_minutes > global library config
            if search_config.timeout_minutes is None:
                if global_config.timeout_minutes is not None:
                    # Use blueprint's global timeout
                    search_config.timeout_minutes = global_config.timeout_minutes
                else:
                    # Use library global config default
                    library_config = get_config()
                    search_config.timeout_minutes = library_config.timeout_minutes
            
            # Create and return DesignSpace
            design_space = DesignSpace(
                model_path=model_path,
                hw_compiler_space=hw_compiler_space,
                processing_space=processing_space,
                search_config=search_config,
                global_config=global_config
            )
            
            logger.info(f"Successfully parsed blueprint: {blueprint_data.get('name', 'unnamed')}")
            return design_space
            
        except Exception as e:
            if isinstance(e, BlueprintParseError):
                raise
            raise BlueprintParseError(f"Failed to parse blueprint: {str(e)}")
    
    def _validate_version(self, blueprint_data: Dict[str, Any]):
        """Validate blueprint version compatibility."""
        version = blueprint_data.get("version")
        if not version:
            raise BlueprintParseError("Blueprint version not specified")
        
        if version != self.SUPPORTED_VERSION:
            raise BlueprintParseError(
                f"Unsupported blueprint version: {version}. "
                f"This parser supports version {self.SUPPORTED_VERSION}"
            )
    
    # ========== Validation Helpers ==========
    
    def _validate_field_type(self, value: Any, expected_type: Type, field_path: str):
        """Validate that a field has the expected type."""
        if not isinstance(value, expected_type):
            type_name = expected_type.__name__
            raise BlueprintParseError(
                f"{field_path} must be a {type_name}, got {type(value).__name__}"
            )
    
    def _validate_required_field(self, data: Dict[str, Any], field: str, field_path: str):
        """Validate that a required field exists."""
        if field not in data or data[field] is None:
            raise BlueprintParseError(f"{field_path} is required")
    
    def _parse_enum_value(self, value: str, enum_class: Type, field_path: str):
        """Parse a string into an enum value with consistent error handling."""
        try:
            return enum_class(value)
        except ValueError:
            valid_values = [e.value for e in enum_class]
            raise BlueprintParseError(
                f"Unknown {field_path}: {value}. "
                f"Supported: {valid_values}"
            )
    
    def _get_field_with_validation(
        self, 
        data: Dict[str, Any], 
        field: str, 
        expected_type: Type,
        default: Any = None,
        required: bool = False,
        parent_path: str = ""
    ) -> Any:
        """Get a field from data with type validation."""
        field_path = f"{parent_path}.{field}" if parent_path else field
        
        if required:
            self._validate_required_field(data, field, field_path)
        
        value = data.get(field, default)
        if value is not None and not isinstance(value, expected_type):
            self._validate_field_type(value, expected_type, field_path)
        
        return value
    
    # ========== Main Parsing Methods ==========
    
    @wrap_parse_errors("hw_compiler")
    def _parse_hw_compiler(self, hw_data: Dict[str, Any]) -> HWCompilerSpace:
        """Parse hardware compiler configuration space."""
        # Parse kernels with plugin validation
        kernels_data = self._get_field_with_validation(
            hw_data, "kernels", list, [], parent_path="hw_compiler"
        )
        validated_kernels = self._validate_and_enrich_kernels(kernels_data)
        
        # Parse transforms with validation
        transforms_data = self._get_field_with_validation(
            hw_data, "transforms", (list, dict), [], parent_path="hw_compiler"
        )
        self._validate_transforms(transforms_data)
        
        return HWCompilerSpace(
            kernels=validated_kernels,
            transforms=transforms_data,
            build_steps=self._get_field_with_validation(
                hw_data, "build_steps", list, [], parent_path="hw_compiler"
            ),
            config_flags=self._get_field_with_validation(
                hw_data, "config_flags", dict, {}, parent_path="hw_compiler"
            )
        )
    
    @wrap_parse_errors("processing")
    def _parse_processing(self, proc_data: Dict[str, Any]) -> ProcessingSpace:
        """Parse processing configuration space."""
        preprocessing = self._parse_processing_steps(
            proc_data.get("preprocessing", []), 
            "preprocessing"
        )
        postprocessing = self._parse_processing_steps(
            proc_data.get("postprocessing", []), 
            "postprocessing"
        )
        
        return ProcessingSpace(
            preprocessing=preprocessing,
            postprocessing=postprocessing
        )
    
    def _parse_processing_steps(
        self, 
        steps_data: List[Dict[str, Any]], 
        step_type: str
    ) -> List[List[ProcessingStep]]:
        """Parse a list of processing steps with alternatives."""
        self._validate_field_type(steps_data, list, f"processing.{step_type}")
        
        all_steps = []
        
        for i, step_config in enumerate(steps_data):
            self._validate_field_type(
                step_config, dict, f"processing.{step_type}[{i}]"
            )
            
            step_name = self._get_field_with_validation(
                step_config, "name", str, required=True,
                parent_path=f"processing.{step_type}[{i}]"
            )
            
            # Parse options for this step
            options = self._get_field_with_validation(
                step_config, "options", list, [], 
                parent_path=f"processing.{step_type}[{i}]"
            )
            
            step_alternatives = []
            for j, option in enumerate(options):
                self._validate_field_type(
                    option, dict, 
                    f"processing.{step_type}[{i}].options[{j}]"
                )
                
                step = ProcessingStep(
                    name=step_name,
                    type=step_type,
                    parameters=option,
                    enabled=option.get("enabled", True)
                )
                step_alternatives.append(step)
            
            if step_alternatives:
                all_steps.append(step_alternatives)
        
        return all_steps
    
    @wrap_parse_errors("search")
    def _parse_search(self, search_data: Dict[str, Any]) -> SearchConfig:
        """Parse search configuration."""
        # Parse strategy
        strategy_str = search_data.get("strategy", "exhaustive")
        strategy = self._parse_enum_value(strategy_str, SearchStrategy, "search.strategy")
        
        # Parse constraints
        constraints_data = self._get_field_with_validation(
            search_data, "constraints", list, [], parent_path="search"
        )
        constraints = [
            self._parse_constraint(c, i) 
            for i, c in enumerate(constraints_data)
        ]
        
        # Parse timeout_minutes - will be None if not specified here
        # The effective timeout will be resolved later considering global config
        timeout_minutes = search_data.get("timeout_minutes")
        if timeout_minutes is not None:
            try:
                timeout_minutes = int(timeout_minutes)
                if timeout_minutes <= 0:
                    raise BlueprintParseError(
                        "search.timeout_minutes must be a positive integer"
                    )
            except (TypeError, ValueError):
                raise BlueprintParseError(
                    f"search.timeout_minutes must be an integer, got {type(timeout_minutes).__name__}"
                )
        
        return SearchConfig(
            strategy=strategy,
            constraints=constraints,
            max_evaluations=search_data.get("max_evaluations"),
            timeout_minutes=timeout_minutes,
            parallel_builds=search_data.get("parallel_builds", 1)
        )
    
    def _parse_constraint(self, constraint_data: Dict[str, Any], index: int) -> SearchConstraint:
        """Parse a single constraint."""
        path = f"search.constraints[{index}]"
        self._validate_field_type(constraint_data, dict, path)
        
        metric = self._get_field_with_validation(
            constraint_data, "metric", str, required=True, parent_path=path
        )
        operator = self._get_field_with_validation(
            constraint_data, "operator", str, required=True, parent_path=path
        )
        value = constraint_data.get("value")
        
        if value is None:
            raise BlueprintParseError(f"{path}.value is required")
        
        return SearchConstraint(
            metric=metric,
            operator=operator,
            value=value
        )
    
    @wrap_parse_errors("global")
    def _parse_global(self, global_data: Dict[str, Any]) -> GlobalConfig:
        """Parse global configuration."""
        # Parse output stage
        output_stage_str = global_data.get("output_stage", "rtl")
        output_stage = self._parse_enum_value(
            output_stage_str, OutputStage, "global.output_stage"
        )
        
        # Parse optional max_combinations
        max_combinations = global_data.get("max_combinations")
        if max_combinations is not None:
            try:
                max_combinations = int(max_combinations)
                if max_combinations <= 0:
                    raise BlueprintParseError(
                        "global.max_combinations must be a positive integer"
                    )
            except (TypeError, ValueError):
                raise BlueprintParseError(
                    f"global.max_combinations must be an integer, got {type(max_combinations).__name__}"
                )
        
        # Parse optional timeout_minutes
        timeout_minutes = global_data.get("timeout_minutes")
        if timeout_minutes is not None:
            try:
                timeout_minutes = int(timeout_minutes)
                if timeout_minutes <= 0:
                    raise BlueprintParseError(
                        "global.timeout_minutes must be a positive integer"
                    )
            except (TypeError, ValueError):
                raise BlueprintParseError(
                    f"global.timeout_minutes must be an integer, got {type(timeout_minutes).__name__}"
                )
        
        # Parse step configuration fields
        start_step = global_data.get("start_step")
        stop_step = global_data.get("stop_step")
        input_type = global_data.get("input_type")
        output_type = global_data.get("output_type")
        
        # Validate step types if provided
        if start_step is not None and not isinstance(start_step, (str, int)):
            raise BlueprintParseError(
                f"global.start_step must be a string or integer, got {type(start_step).__name__}"
            )
        
        if stop_step is not None and not isinstance(stop_step, (str, int)):
            raise BlueprintParseError(
                f"global.stop_step must be a string or integer, got {type(stop_step).__name__}"
            )
        
        if input_type is not None and not isinstance(input_type, str):
            raise BlueprintParseError(
                f"global.input_type must be a string, got {type(input_type).__name__}"
            )
        
        if output_type is not None and not isinstance(output_type, str):
            raise BlueprintParseError(
                f"global.output_type must be a string, got {type(output_type).__name__}"
            )

        return GlobalConfig(
            output_stage=output_stage,
            working_directory=global_data.get("working_directory", "./builds"),
            cache_results=global_data.get("cache_results", True),
            save_artifacts=global_data.get("save_artifacts", True),
            log_level=global_data.get("log_level", "INFO"),
            max_combinations=max_combinations,
            timeout_minutes=timeout_minutes,
            start_step=start_step,
            stop_step=stop_step,
            input_type=input_type,
            output_type=output_type
        )
    
    # ========== Plugin Validation Methods ==========
    
    def _validate_and_enrich_kernels(self, kernels_data: List) -> List:
        """
        Validate kernel specifications and auto-discover backends when needed.
        
        Args:
            kernels_data: Raw kernel specifications from blueprint
            
        Returns:
            Enriched kernel specifications with validated backends
        """
        validated_kernels = []
        
        for i, kernel_spec in enumerate(kernels_data):
            if isinstance(kernel_spec, str):
                # Simple kernel name - auto-discover backends
                kernel_name = kernel_spec.strip("~")
                if kernel_name not in self.plugin_registry.kernels:
                    raise BlueprintParseError(
                        f"Kernel '{kernel_name}' not found in plugin registry. "
                        f"Available kernels: {self.plugin_registry.list_available_kernels()[:5]}..."
                    )
                
                # Get available backends
                available_backends = self.plugin_registry.list_backends_by_kernel(kernel_name)
                if not available_backends:
                    raise BlueprintParseError(
                        f"No backends found for kernel '{kernel_name}'"
                    )
                
                # Create tuple with auto-discovered backends
                validated_kernels.append((kernel_spec, available_backends))
                logger.debug(f"Auto-discovered backends for '{kernel_name}': {available_backends}")
                
            elif isinstance(kernel_spec, (tuple, list)) and len(kernel_spec) == 2:
                # Check if this is a kernel with explicit backends (not a mutually exclusive group)
                # In YAML, both are lists, but kernel+backends has string as first element
                if isinstance(kernel_spec[0], str) and isinstance(kernel_spec[1], list):
                    # Explicit backend specification - validate they exist
                    kernel_name, specified_backends = kernel_spec
                    kernel_name_clean = kernel_name.strip("~") if isinstance(kernel_name, str) else kernel_name
                    
                    # Validate kernel exists
                    if kernel_name_clean not in self.plugin_registry.kernels:
                        raise BlueprintParseError(
                            f"Kernel '{kernel_name_clean}' not found in plugin registry"
                        )
                    
                    # Validate backends
                    invalid_backends = self.plugin_registry.validate_kernel_backends(
                        kernel_name_clean, specified_backends
                    )
                    if invalid_backends:
                        available = self.plugin_registry.list_backends_by_kernel(kernel_name_clean)
                        raise BlueprintParseError(
                            f"Invalid backends {invalid_backends} for kernel '{kernel_name_clean}'. "
                            f"Available: {available}"
                        )
                    
                    # Convert to tuple to distinguish from mutually exclusive groups
                    validated_kernels.append((kernel_name, specified_backends))
                else:
                    # This is a mutually exclusive group
                    validated_group = []
                    for item in kernel_spec:
                        if item is not None:
                            validated_items = self._validate_and_enrich_kernels([item])
                            validated_group.extend(validated_items)
                        else:
                            validated_group.append(None)
                    validated_kernels.append(validated_group)
                
            elif isinstance(kernel_spec, list):
                # Mutually exclusive group - validate each item
                validated_group = []
                for item in kernel_spec:
                    validated_items = self._validate_and_enrich_kernels([item])
                    validated_group.extend(validated_items)
                validated_kernels.append(validated_group)
                
            else:
                validated_kernels.append(kernel_spec)
        
        return validated_kernels
    
    def _validate_transforms(self, transforms_data: Union[List, Dict]):
        """
        Validate transform specifications exist in registry.
        
        Args:
            transforms_data: Transform specifications (flat list or phase dict)
        """
        if isinstance(transforms_data, list):
            # Flat list of transforms
            for transform_spec in transforms_data:
                if isinstance(transform_spec, str):
                    transform_name = transform_spec.strip("~")
                    if transform_name not in self.plugin_registry.transforms:
                        available = self.plugin_registry.list_available_transforms()[:5]
                        raise BlueprintParseError(
                            f"Transform '{transform_name}' not found. "
                            f"Available: {available}..."
                        )
                        
        elif isinstance(transforms_data, dict):
            # Phase-based transforms
            for phase, phase_transforms in transforms_data.items():
                if isinstance(phase_transforms, list):
                    for transform_spec in phase_transforms:
                        if isinstance(transform_spec, str):
                            transform_name = transform_spec.strip("~")
                            if transform_name not in self.plugin_registry.transforms:
                                available = self.plugin_registry.list_available_transforms()[:5]
                                raise BlueprintParseError(
                                    f"Transform '{transform_name}' in phase '{phase}' not found. "
                                    f"Available: {available}..."
                                )


def load_blueprint(blueprint_path: str) -> Dict[str, Any]:
    """
    Load and parse a blueprint YAML file.
    
    Args:
        blueprint_path: Path to the blueprint YAML file
        
    Returns:
        Parsed YAML data as a dictionary
        
    Raises:
        BlueprintParseError: If file cannot be loaded or parsed
    """
    try:
        with open(blueprint_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise BlueprintParseError(
                "Blueprint file must contain a YAML dictionary at the top level"
            )
        
        return data
        
    except FileNotFoundError:
        raise BlueprintParseError(f"Blueprint file not found: {blueprint_path}")
    except yaml.YAMLError as e:
        raise BlueprintParseError(f"Invalid YAML syntax: {str(e)}")
    except Exception as e:
        raise BlueprintParseError(f"Failed to load blueprint: {str(e)}")