"""
Blueprint parser for DSE V3 - Refactored version.

This module handles parsing Blueprint YAML files into structured DesignSpace objects.
It supports various kernel and transform formats including simple strings, tuples,
mutually exclusive groups, and optional elements.
"""

import yaml
from typing import Dict, List, Any, Union

from .data_structures import (
    DesignSpace,
    HWCompilerSpace,
    SearchConfig,
    SearchConstraint,
    SearchStrategy,
    GlobalConfig,
    OutputStage,
)
from .exceptions import BlueprintParseError
from ..config import get_config
from ..plugins import get_registry


# Removed FieldSpec - not used in this refactoring


# Removed wrap_parse_errors decorator - let errors bubble up naturally


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
    
    def __init__(self):
        """Initialize parser."""
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
            # Parse each section
            hw_compiler_space = self._parse_hw_compiler(
                blueprint_data.get("hw_compiler", {})
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
                search_config=search_config,
                global_config=global_config
            )
            
            # Successfully parsed
            return design_space
            
        except Exception as e:
            if isinstance(e, BlueprintParseError):
                raise
            raise BlueprintParseError(f"Failed to parse blueprint: {str(e)}")
    
    # ========== Removed validation helpers - let Python handle errors ==========
    
    # ========== Main Parsing Methods ==========
    
    def _parse_hw_compiler(self, hw_data: Dict[str, Any]) -> HWCompilerSpace:
        """Parse hardware compiler configuration space."""
        # Get kernels and transforms
        kernels_data = hw_data.get("kernels", [])
        kernels = self._process_kernels(kernels_data)
        transforms_data = hw_data.get("transforms", [])
        
        return HWCompilerSpace(
            kernels=kernels,
            transforms=transforms_data,
            build_steps=hw_data.get("build_steps", []),
            config_flags=hw_data.get("config_flags", {})
        )
    
    def _parse_search(self, search_data: Dict[str, Any]) -> SearchConfig:
        """Parse search configuration."""
        # Parse strategy and constraints
        strategy = SearchStrategy(search_data.get("strategy", "exhaustive"))
        constraints_data = search_data.get("constraints", [])
        constraints = [self._parse_constraint(c, i) for i, c in enumerate(constraints_data)]
        
        timeout_minutes = search_data.get("timeout_minutes")
        
        return SearchConfig(
            strategy=strategy,
            constraints=constraints,
            max_evaluations=search_data.get("max_evaluations"),
            timeout_minutes=timeout_minutes,
            parallel_builds=search_data.get("parallel_builds", 1)
        )
    
    def _parse_constraint(self, constraint_data: Dict[str, Any], index: int) -> SearchConstraint:
        """Parse a single constraint."""
        return SearchConstraint(
            metric=constraint_data["metric"],
            operator=constraint_data["operator"],
            value=constraint_data["value"]
        )
    
    def _parse_global(self, global_data: Dict[str, Any]) -> GlobalConfig:
        """Parse global configuration."""
        # Parse output stage
        output_stage = OutputStage(global_data.get("output_stage", "rtl"))
        
        # Get optional fields
        max_combinations = global_data.get("max_combinations")
        timeout_minutes = global_data.get("timeout_minutes")
        
        return GlobalConfig(
            output_stage=output_stage,
            working_directory=global_data.get("working_directory", "./builds"),
            cache_results=global_data.get("cache_results", True),
            save_artifacts=global_data.get("save_artifacts", True),
            log_level=global_data.get("log_level", "INFO"),
            max_combinations=max_combinations,
            timeout_minutes=timeout_minutes
        )
    
    # ========== Simplified Processing Methods ==========
    
    def _process_kernels(self, kernels_data: List) -> List:
        """Process kernel specifications and auto-discover backends when needed."""
        processed_kernels = []
        
        for kernel_spec in kernels_data:
            if isinstance(kernel_spec, str):
                # Simple kernel name - auto-discover backends
                kernel_name = kernel_spec.strip("~")
                # Get backends from registry
                backends = list(self.plugin_registry.backends_by_kernel.get(kernel_name, []))
                processed_kernels.append((kernel_spec, backends))
                
            elif isinstance(kernel_spec, (tuple, list)) and len(kernel_spec) == 2:
                # Check if this is a kernel with explicit backends
                if isinstance(kernel_spec[0], str) and isinstance(kernel_spec[1], list):
                    # Explicit backend specification
                    processed_kernels.append(tuple(kernel_spec))
                else:
                    # Mutually exclusive group
                    group = []
                    for item in kernel_spec:
                        if item is not None:
                            group.extend(self._process_kernels([item]))
                        else:
                            group.append(None)
                    processed_kernels.append(group)
                
            elif isinstance(kernel_spec, list):
                # Mutually exclusive group
                group = []
                for item in kernel_spec:
                    group.extend(self._process_kernels([item]))
                processed_kernels.append(group)
                
            else:
                processed_kernels.append(kernel_spec)
        
        return processed_kernels


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