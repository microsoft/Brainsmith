"""
Blueprint V2 System - Design Space Definition and Parser

This module implements the V2 blueprint system that defines design spaces
for 6-entrypoint FINN architecture exploration rather than fixed configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import yaml
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationDirection(Enum):
    """Optimization direction for objectives."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class ExplorationRules:
    """Rules for exploring component combinations in design space."""
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    mutually_exclusive: List[List[str]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate exploration rules for consistency."""
        errors = []
        
        # Check for overlap between required and optional
        required_set = set(self.required)
        optional_set = set(self.optional)
        overlap = required_set.intersection(optional_set)
        if overlap:
            errors.append(f"Components cannot be both required and optional: {overlap}")
        
        # Check mutually exclusive groups for validity
        for i, group in enumerate(self.mutually_exclusive):
            if len(group) < 2:
                errors.append(f"Mutually exclusive group {i} must have at least 2 components")
            
            # Check if any required components are in mutually exclusive groups
            group_set = set(group)
            required_in_group = required_set.intersection(group_set)
            if required_in_group:
                errors.append(f"Required components cannot be in mutually exclusive groups: {required_in_group}")
        
        # Check dependencies for validity
        all_components = required_set.union(optional_set)
        for component, deps in self.dependencies.items():
            if component not in all_components:
                errors.append(f"Dependency component '{component}' not in available components")
            for dep in deps:
                if dep not in all_components:
                    errors.append(f"Dependency '{dep}' for '{component}' not in available components")
        
        # Check for circular dependencies
        def has_circular_dependency(component: str, visited: set, path: set) -> bool:
            if component in path:
                return True
            if component in visited:
                return False
            
            visited.add(component)
            path.add(component)
            
            for dep in self.dependencies.get(component, []):
                if has_circular_dependency(dep, visited, path):
                    return True
            
            path.remove(component)
            return False
        
        visited = set()
        for component in self.dependencies:
            if has_circular_dependency(component, visited, set()):
                errors.append(f"Circular dependency detected involving '{component}'")
                break
        
        return len(errors) == 0, errors


@dataclass
class ComponentSpace:
    """Design space for a component category with exploration rules."""
    available: List[Union[str, Dict[str, List[str]]]] = field(default_factory=list)
    exploration: Optional[ExplorationRules] = None
    
    def __post_init__(self):
        """Initialize exploration rules if not provided."""
        if self.exploration is None:
            self.exploration = ExplorationRules()
    
    def get_component_names(self) -> List[str]:
        """Extract all component names from available list."""
        names = []
        for item in self.available:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.extend(item.keys())
        return names
    
    def get_component_options(self, component_name: str) -> List[str]:
        """Get available options for a specific component."""
        for item in self.available:
            if isinstance(item, str) and item == component_name:
                return [component_name]  # Single option
            elif isinstance(item, dict) and component_name in item:
                return item[component_name]
        return []
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate component space configuration."""
        errors = []
        
        # Allow empty component spaces - FINN will use defaults
        # This enables all 6 entrypoints to be optional
        if not self.available:
            # Empty space is valid - FINN will use default components/transforms
            # No exploration rules should be specified for empty spaces
            if self.exploration and (self.exploration.required or self.exploration.optional or 
                                   self.exploration.mutually_exclusive or self.exploration.dependencies):
                errors.append("Empty component space cannot have exploration rules")
            return len(errors) == 0, errors
        
        for item in self.available:
            if not isinstance(item, (str, dict)):
                errors.append(f"Available component must be string or dict, got {type(item)}")
            elif isinstance(item, dict):
                for key, value in item.items():
                    if not isinstance(key, str):
                        errors.append(f"Component name must be string, got {type(key)}")
                    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                        errors.append(f"Component options must be list of strings for '{key}'")
        
        # Validate exploration rules
        if self.exploration:
            rules_valid, rules_errors = self.exploration.validate()
            if not rules_valid:
                errors.extend(rules_errors)
            
            # Check that exploration rules reference valid components
            all_components = set(self.get_component_names())
            all_exploration_components = set(self.exploration.required + self.exploration.optional)
            
            # Add mutually exclusive components
            if self.exploration.mutually_exclusive:
                for group in self.exploration.mutually_exclusive:
                    all_exploration_components.update(group)
            
            # Add dependency components
            if self.exploration.dependencies:
                all_exploration_components.update(self.exploration.dependencies.keys())
                for deps in self.exploration.dependencies.values():
                    all_exploration_components.update(deps)
            
            invalid_components = all_exploration_components - all_components
            if invalid_components:
                errors.append(f"Exploration rules reference non-existent components: {invalid_components}")
        
        return len(errors) == 0, errors


@dataclass
class NodeDesignSpace:
    """Design space definition for node components (canonical ops and hw kernels)."""
    canonical_ops: ComponentSpace = field(default_factory=ComponentSpace)
    hw_kernels: ComponentSpace = field(default_factory=ComponentSpace)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate node design space."""
        errors = []
        
        # Validate canonical ops
        canonical_valid, canonical_errors = self.canonical_ops.validate()
        if not canonical_valid:
            errors.extend([f"canonical_ops: {err}" for err in canonical_errors])
        
        # Validate hw kernels
        hw_valid, hw_errors = self.hw_kernels.validate()
        if not hw_valid:
            errors.extend([f"hw_kernels: {err}" for err in hw_errors])
        
        return len(errors) == 0, errors


@dataclass
class TransformDesignSpace:
    """Design space definition for transform components."""
    model_topology: ComponentSpace = field(default_factory=ComponentSpace)
    hw_kernel: ComponentSpace = field(default_factory=ComponentSpace)
    hw_graph: ComponentSpace = field(default_factory=ComponentSpace)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate transform design space."""
        errors = []
        
        # Validate each transform category
        for name, space in [
            ("model_topology", self.model_topology),
            ("hw_kernel", self.hw_kernel),
            ("hw_graph", self.hw_graph)
        ]:
            space_valid, space_errors = space.validate()
            if not space_valid:
                errors.extend([f"{name}: {err}" for err in space_errors])
        
        return len(errors) == 0, errors


@dataclass
class Objective:
    """Optimization objective definition."""
    name: str
    direction: OptimizationDirection
    weight: float = 1.0
    target_value: Optional[float] = None
    
    def __post_init__(self):
        """Validate objective parameters."""
        if self.weight <= 0:
            raise ValueError(f"Objective weight must be positive, got {self.weight}")
        if not isinstance(self.direction, OptimizationDirection):
            # Convert string to enum if needed
            if isinstance(self.direction, str):
                self.direction = OptimizationDirection(self.direction)
            else:
                raise ValueError(f"Invalid direction type: {type(self.direction)}")


@dataclass
class Constraint:
    """Design constraint definition."""
    name: str
    operator: str  # '<', '<=', '>', '>=', '==', '!='
    value: Union[float, int, str]
    description: str = ""
    
    def __post_init__(self):
        """Validate constraint parameters."""
        valid_operators = {'<', '<=', '>', '>=', '==', '!='}
        if self.operator not in valid_operators:
            raise ValueError(f"Invalid constraint operator '{self.operator}', must be one of {valid_operators}")


@dataclass
class DSEStrategy:
    """DSE strategy configuration."""
    name: str
    description: str = ""
    max_evaluations: int = 50
    sampling: str = "random"  # "random", "grid", "latin_hypercube", "adaptive"
    focus_areas: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate strategy parameters."""
        if self.max_evaluations <= 0:
            raise ValueError(f"max_evaluations must be positive, got {self.max_evaluations}")
        
        valid_sampling = {"random", "grid", "latin_hypercube", "adaptive", "pareto_guided"}
        if self.sampling not in valid_sampling:
            raise ValueError(f"Invalid sampling strategy '{self.sampling}', must be one of {valid_sampling}")


@dataclass
class DSEStrategies:
    """Collection of DSE strategies with primary strategy selection."""
    primary_strategy: str
    strategies: Dict[str, DSEStrategy] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate strategy collection."""
        if self.primary_strategy not in self.strategies:
            raise ValueError(f"Primary strategy '{self.primary_strategy}' not found in strategies")
    
    def get_primary_strategy(self) -> DSEStrategy:
        """Get the primary strategy configuration."""
        return self.strategies[self.primary_strategy]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate strategy configurations."""
        errors = []
        
        if not self.strategies:
            errors.append("At least one strategy must be defined")
        
        if self.primary_strategy not in self.strategies:
            errors.append(f"Primary strategy '{self.primary_strategy}' not found in strategies")
        
        return len(errors) == 0, errors


@dataclass
class DesignSpaceDefinition:
    """Complete design space definition for Blueprint V2."""
    name: str
    version: str = "2.0"
    base_blueprint: Optional[str] = None
    nodes: NodeDesignSpace = field(default_factory=NodeDesignSpace)
    transforms: TransformDesignSpace = field(default_factory=TransformDesignSpace)
    configuration_files: Dict[str, str] = field(default_factory=dict)
    dse_strategies: Optional[DSEStrategies] = None
    objectives: List[Objective] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate complete design space definition."""
        errors = []
        
        # Validate name and version
        if not self.name:
            errors.append("Blueprint name is required")
        
        if not self.version:
            errors.append("Blueprint version is required")
        
        # Validate nodes design space
        nodes_valid, nodes_errors = self.nodes.validate()
        if not nodes_valid:
            errors.extend([f"nodes: {err}" for err in nodes_errors])
        
        # Validate transforms design space
        transforms_valid, transforms_errors = self.transforms.validate()
        if not transforms_valid:
            errors.extend([f"transforms: {err}" for err in transforms_errors])
        
        # Validate DSE strategies
        if self.dse_strategies:
            strategies_valid, strategies_errors = self.dse_strategies.validate()
            if not strategies_valid:
                errors.extend([f"dse_strategies: {err}" for err in strategies_errors])
        
        # Validate objectives
        objective_names = set()
        for obj in self.objectives:
            if obj.name in objective_names:
                errors.append(f"Duplicate objective name: {obj.name}")
            objective_names.add(obj.name)
        
        # Validate constraints
        constraint_names = set()
        for constraint in self.constraints:
            if constraint.name in constraint_names:
                errors.append(f"Duplicate constraint name: {constraint.name}")
            constraint_names.add(constraint.name)
        
        # Validate configuration files exist (if paths provided)
        for file_type, file_path in self.configuration_files.items():
            if file_path and not Path(file_path).exists():
                # Only warn, don't error - files might be relative to blueprint location
                logger.warning(f"Configuration file not found: {file_path} (type: {file_type})")
        
        return len(errors) == 0, errors
    
    def get_all_components(self) -> Dict[str, List[str]]:
        """Get all available components across all categories."""
        components = {}
        
        # Add node components
        components['canonical_ops'] = self.nodes.canonical_ops.get_component_names()
        components['hw_kernels'] = self.nodes.hw_kernels.get_component_names()
        
        # Add transform components
        components['model_topology'] = self.transforms.model_topology.get_component_names()
        components['hw_kernel'] = self.transforms.hw_kernel.get_component_names()
        components['hw_graph'] = self.transforms.hw_graph.get_component_names()
        
        return components


def _is_blueprint_v2(blueprint_path: str) -> bool:
    """Detect if blueprint file is V2 format."""
    try:
        with open(blueprint_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # V2 blueprints have 'nodes' and 'transforms' sections
        return isinstance(data, dict) and 'nodes' in data and 'transforms' in data
    except Exception:
        return False


def load_blueprint_v2(blueprint_path: str) -> DesignSpaceDefinition:
    """Load Blueprint V2 from YAML file with enhanced inheritance support."""
    blueprint_path = Path(blueprint_path)
    
    if not blueprint_path.exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    with open(blueprint_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Blueprint must be a YAML dictionary")
    
    # Handle inheritance with enhanced system
    if 'base_blueprint' in data and data['base_blueprint']:
        try:
            from .blueprint_inheritance import (
                merge_blueprints, resolve_blueprint_path, 
                validate_inheritance_chain, BlueprintInheritanceError
            )
            
            # Validate inheritance chain for circular dependencies
            validate_inheritance_chain(blueprint_path)
            
            # Resolve and load base blueprint
            base_path = resolve_blueprint_path(blueprint_path, data['base_blueprint'])
            base_blueprint = load_blueprint_v2(str(base_path))
            
            # Merge base with current using intelligent merging
            data = merge_blueprints(base_blueprint, data)
            
        except BlueprintInheritanceError as e:
            logger.error(f"Blueprint inheritance error: {e}")
            raise ValueError(f"Blueprint inheritance failed: {e}")
        except ImportError:
            logger.warning("Blueprint inheritance system not available, skipping inheritance")
    
    # Parse blueprint data into data structures
    blueprint = _parse_blueprint_data(data, str(blueprint_path))
    
    # Validate the blueprint
    is_valid, errors = blueprint.validate()
    if not is_valid:
        raise ValueError(f"Invalid blueprint: {'; '.join(errors)}")
    
    return blueprint


def _parse_blueprint_data(data: Dict[str, Any], blueprint_path: str) -> DesignSpaceDefinition:
    """Parse blueprint YAML data into data structures."""
    
    # Parse nodes section
    nodes_data = data.get('nodes', {})
    nodes = NodeDesignSpace(
        canonical_ops=_parse_component_space(nodes_data.get('canonical_ops', {})),
        hw_kernels=_parse_component_space(nodes_data.get('hw_kernels', {}))
    )
    
    # Parse transforms section
    transforms_data = data.get('transforms', {})
    transforms = TransformDesignSpace(
        model_topology=_parse_component_space(transforms_data.get('model_topology', {})),
        hw_kernel=_parse_component_space(transforms_data.get('hw_kernel', {})),
        hw_graph=_parse_component_space(transforms_data.get('hw_graph', {}))
    )
    
    # Parse DSE strategies
    dse_strategies = None
    if 'dse_strategies' in data:
        dse_strategies = _parse_dse_strategies(data['dse_strategies'])
    
    # Parse objectives
    objectives = []
    for obj_data in data.get('objectives', []):
        if isinstance(obj_data, dict):
            objectives.append(Objective(**obj_data))
    
    # Parse constraints
    constraints = []
    for constraint_data in data.get('constraints', []):
        if isinstance(constraint_data, dict):
            constraints.append(Constraint(**constraint_data))
    
    return DesignSpaceDefinition(
        name=data.get('name', 'unnamed_blueprint'),
        version=data.get('version', '2.0'),
        base_blueprint=data.get('base_blueprint'),
        nodes=nodes,
        transforms=transforms,
        configuration_files=data.get('configuration_files', {}),
        dse_strategies=dse_strategies,
        objectives=objectives,
        constraints=constraints
    )


def _parse_component_space(space_data: Union[Dict, List]) -> ComponentSpace:
    """Parse component space from YAML data."""
    if isinstance(space_data, list):
        # Simple list format
        return ComponentSpace(available=space_data)
    elif isinstance(space_data, dict):
        # Structured format with exploration rules
        available = space_data.get('available', [])
        exploration_data = space_data.get('exploration', {})
        
        exploration = ExplorationRules(
            required=exploration_data.get('required', []),
            optional=exploration_data.get('optional', []),
            mutually_exclusive=exploration_data.get('mutually_exclusive', []),
            dependencies=exploration_data.get('dependencies', {})
        )
        
        return ComponentSpace(available=available, exploration=exploration)
    else:
        return ComponentSpace()


def _parse_dse_strategies(strategies_data: Dict[str, Any]) -> DSEStrategies:
    """Parse DSE strategies from YAML data."""
    primary_strategy = strategies_data.get('primary_strategy')
    strategies = {}
    
    for name, strategy_data in strategies_data.get('strategies', {}).items():
        strategies[name] = DSEStrategy(name=name, **strategy_data)
    
    return DSEStrategies(primary_strategy=primary_strategy, strategies=strategies)