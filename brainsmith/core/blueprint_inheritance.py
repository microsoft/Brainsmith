"""
Blueprint V2 Inheritance System

Handles hierarchical blueprint inheritance with proper merging of design spaces,
exploration rules, and configuration sections.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from copy import deepcopy

from .blueprint import (
    DesignSpaceDefinition, NodeDesignSpace, TransformDesignSpace,
    ComponentSpace, ExplorationRules, DSEStrategies, Objective, Constraint
)

logger = logging.getLogger(__name__)


class BlueprintInheritanceError(Exception):
    """Raised when blueprint inheritance fails."""
    pass


def merge_blueprints(base: DesignSpaceDefinition, derived: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge base blueprint with derived blueprint data using intelligent merging rules.
    
    Args:
        base: Base blueprint (already parsed)
        derived: Derived blueprint data (from YAML)
        
    Returns:
        Merged blueprint data dictionary
    """
    # Start with base blueprint converted back to dict format
    merged = _blueprint_to_dict(base)
    
    # Apply derived blueprint changes
    merged = _deep_merge_blueprint_sections(merged, derived)
    
    return merged


def _blueprint_to_dict(blueprint: DesignSpaceDefinition) -> Dict[str, Any]:
    """Convert blueprint object back to dictionary format for merging."""
    result = {
        'name': blueprint.name,
        'version': blueprint.version,
        'base_blueprint': blueprint.base_blueprint,
        'nodes': _node_design_space_to_dict(blueprint.nodes),
        'transforms': _transform_design_space_to_dict(blueprint.transforms),
        'configuration_files': dict(blueprint.configuration_files),
        'objectives': [_objective_to_dict(obj) for obj in blueprint.objectives],
        'constraints': [_constraint_to_dict(constraint) for constraint in blueprint.constraints]
    }
    
    if blueprint.dse_strategies:
        result['dse_strategies'] = _dse_strategies_to_dict(blueprint.dse_strategies)
    
    return result


def _node_design_space_to_dict(nodes: NodeDesignSpace) -> Dict[str, Any]:
    """Convert NodeDesignSpace to dictionary."""
    return {
        'canonical_ops': _component_space_to_dict(nodes.canonical_ops),
        'hw_kernels': _component_space_to_dict(nodes.hw_kernels)
    }


def _transform_design_space_to_dict(transforms: TransformDesignSpace) -> Dict[str, Any]:
    """Convert TransformDesignSpace to dictionary."""
    return {
        'model_topology': _component_space_to_dict(transforms.model_topology),
        'hw_kernel': _component_space_to_dict(transforms.hw_kernel),
        'hw_graph': _component_space_to_dict(transforms.hw_graph)
    }


def _component_space_to_dict(space: ComponentSpace) -> Dict[str, Any]:
    """Convert ComponentSpace to dictionary."""
    result = {'available': space.available}
    
    if space.exploration and (
        space.exploration.required or 
        space.exploration.optional or 
        space.exploration.mutually_exclusive or 
        space.exploration.dependencies
    ):
        exploration = {}
        if space.exploration.required:
            exploration['required'] = space.exploration.required
        if space.exploration.optional:
            exploration['optional'] = space.exploration.optional
        if space.exploration.mutually_exclusive:
            exploration['mutually_exclusive'] = space.exploration.mutually_exclusive
        if space.exploration.dependencies:
            exploration['dependencies'] = space.exploration.dependencies
        
        result['exploration'] = exploration
    
    return result


def _objective_to_dict(obj: Objective) -> Dict[str, Any]:
    """Convert Objective to dictionary."""
    result = {
        'name': obj.name,
        'direction': obj.direction.value,
        'weight': obj.weight
    }
    if obj.target_value is not None:
        result['target_value'] = obj.target_value
    return result


def _constraint_to_dict(constraint: Constraint) -> Dict[str, Any]:
    """Convert Constraint to dictionary."""
    result = {
        'name': constraint.name,
        'operator': constraint.operator,
        'value': constraint.value
    }
    if constraint.description:
        result['description'] = constraint.description
    return result


def _dse_strategies_to_dict(strategies: DSEStrategies) -> Dict[str, Any]:
    """Convert DSEStrategies to dictionary."""
    return {
        'primary_strategy': strategies.primary_strategy,
        'strategies': {
            name: _dse_strategy_to_dict(strategy)
            for name, strategy in strategies.strategies.items()
        }
    }


def _dse_strategy_to_dict(strategy) -> Dict[str, Any]:
    """Convert DSEStrategy to dictionary."""
    result = {
        'max_evaluations': strategy.max_evaluations,
        'sampling': strategy.sampling
    }
    if strategy.description:
        result['description'] = strategy.description
    if strategy.focus_areas:
        result['focus_areas'] = strategy.focus_areas
    if strategy.objectives:
        result['objectives'] = strategy.objectives
    if strategy.constraints:
        result['constraints'] = strategy.constraints
    return result


def _deep_merge_blueprint_sections(base: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge blueprint sections with intelligent merging rules.
    
    Merging rules:
    - Simple values: derived overrides base
    - Lists: derived extends base (with deduplication)
    - Dicts: recursive merge
    - Special sections: use section-specific merge logic
    """
    merged = deepcopy(base)
    
    for key, value in derived.items():
        if key not in merged:
            # New key in derived - add it
            merged[key] = deepcopy(value)
        elif key in ['name', 'version', 'base_blueprint']:
            # Simple override for metadata
            merged[key] = value
        elif key == 'nodes':
            merged[key] = _merge_node_sections(merged[key], value)
        elif key == 'transforms':
            merged[key] = _merge_transform_sections(merged[key], value)
        elif key == 'configuration_files':
            merged[key] = _merge_configuration_files(merged[key], value)
        elif key == 'dse_strategies':
            merged[key] = _merge_dse_strategies(merged[key], value)
        elif key == 'objectives':
            merged[key] = _merge_objectives(merged[key], value)
        elif key == 'constraints':
            merged[key] = _merge_constraints(merged[key], value)
        else:
            # Default: derived overrides base
            merged[key] = deepcopy(value)
    
    return merged


def _merge_node_sections(base: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    """Merge node sections (canonical_ops, hw_kernels)."""
    merged = deepcopy(base)
    
    for section_name, section_data in derived.items():
        if section_name not in merged:
            merged[section_name] = deepcopy(section_data)
        else:
            merged[section_name] = _merge_component_space(merged[section_name], section_data)
    
    return merged


def _merge_transform_sections(base: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    """Merge transform sections (model_topology, hw_kernel, hw_graph)."""
    merged = deepcopy(base)
    
    for section_name, section_data in derived.items():
        if section_name not in merged:
            merged[section_name] = deepcopy(section_data)
        else:
            merged[section_name] = _merge_component_space(merged[section_name], section_data)
    
    return merged


def _merge_component_space(base: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge component space with intelligent rules:
    - available: extend with deduplication
    - exploration: merge exploration rules intelligently
    """
    merged = deepcopy(base)
    
    # Merge available components
    if 'available' in derived:
        base_available = merged.get('available', [])
        derived_available = derived['available']
        
        # Extend base with derived, handling both strings and dicts
        merged_available = _merge_available_components(base_available, derived_available)
        merged['available'] = merged_available
    
    # Merge exploration rules
    if 'exploration' in derived:
        base_exploration = merged.get('exploration', {})
        derived_exploration = derived['exploration']
        
        merged['exploration'] = _merge_exploration_rules(base_exploration, derived_exploration)
    
    return merged


def _merge_available_components(base: List[Any], derived: List[Any]) -> List[Any]:
    """
    Merge available component lists, handling both strings and dicts.
    For dicts with same key, derived options override base options.
    """
    merged = []
    base_dict_keys = set()
    
    # First pass: add all base components, track dict keys
    for item in base:
        merged.append(item)
        if isinstance(item, dict):
            base_dict_keys.update(item.keys())
    
    # Second pass: add derived components, handling conflicts
    for item in derived:
        if isinstance(item, str):
            # String component - add if not already present
            if item not in merged:
                merged.append(item)
        elif isinstance(item, dict):
            # Dict component - merge or override
            for key, value in item.items():
                if key in base_dict_keys:
                    # Override existing dict entry
                    for i, base_item in enumerate(merged):
                        if isinstance(base_item, dict) and key in base_item:
                            merged[i] = {key: value}  # Override
                            break
                else:
                    # New dict entry
                    merged.append({key: value})
                    base_dict_keys.add(key)
    
    return merged


def _merge_exploration_rules(base: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge exploration rules with intelligent behavior:
    - required: extend (derived can add more requirements)
    - optional: extend (derived can add more options)
    - mutually_exclusive: extend (derived can add more groups)
    - dependencies: merge (derived can add/override dependencies)
    """
    merged = deepcopy(base)
    
    # Merge required (extend)
    if 'required' in derived:
        base_required = set(merged.get('required', []))
        derived_required = set(derived['required'])
        merged['required'] = list(base_required | derived_required)
    
    # Merge optional (extend)
    if 'optional' in derived:
        base_optional = set(merged.get('optional', []))
        derived_optional = set(derived['optional'])
        merged['optional'] = list(base_optional | derived_optional)
    
    # Merge mutually_exclusive (extend groups)
    if 'mutually_exclusive' in derived:
        base_groups = merged.get('mutually_exclusive', [])
        derived_groups = derived['mutually_exclusive']
        merged['mutually_exclusive'] = base_groups + derived_groups
    
    # Merge dependencies (derived overrides base for same keys)
    if 'dependencies' in derived:
        base_deps = merged.get('dependencies', {})
        derived_deps = derived['dependencies']
        merged_deps = deepcopy(base_deps)
        merged_deps.update(derived_deps)
        merged['dependencies'] = merged_deps
    
    return merged


def _merge_configuration_files(base: Dict[str, str], derived: Dict[str, str]) -> Dict[str, str]:
    """Merge configuration files (derived overrides base)."""
    merged = deepcopy(base)
    merged.update(derived)
    return merged


def _merge_dse_strategies(base: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    """Merge DSE strategies (derived strategies extend base, same names override)."""
    merged = deepcopy(base)
    
    # Update primary strategy if specified
    if 'primary_strategy' in derived:
        merged['primary_strategy'] = derived['primary_strategy']
    
    # Merge strategies dict
    if 'strategies' in derived:
        base_strategies = merged.get('strategies', {})
        derived_strategies = derived['strategies']
        
        merged_strategies = deepcopy(base_strategies)
        merged_strategies.update(derived_strategies)
        merged['strategies'] = merged_strategies
    
    return merged


def _merge_objectives(base: List[Dict[str, Any]], derived: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge objectives (derived objectives with same name override base, others extend).
    """
    merged = deepcopy(base)
    
    # Create lookup by name for base objectives
    base_by_name = {obj['name']: i for i, obj in enumerate(merged)}
    
    for derived_obj in derived:
        obj_name = derived_obj['name']
        if obj_name in base_by_name:
            # Override existing objective
            merged[base_by_name[obj_name]] = deepcopy(derived_obj)
        else:
            # Add new objective
            merged.append(deepcopy(derived_obj))
    
    return merged


def _merge_constraints(base: List[Dict[str, Any]], derived: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge constraints (derived constraints with same name override base, others extend).
    """
    merged = deepcopy(base)
    
    # Create lookup by name for base constraints
    base_by_name = {constraint['name']: i for i, constraint in enumerate(merged)}
    
    for derived_constraint in derived:
        constraint_name = derived_constraint['name']
        if constraint_name in base_by_name:
            # Override existing constraint
            merged[base_by_name[constraint_name]] = deepcopy(derived_constraint)
        else:
            # Add new constraint
            merged.append(deepcopy(derived_constraint))
    
    return merged


def resolve_blueprint_path(blueprint_path: Path, base_blueprint_name: str) -> Path:
    """
    Resolve base blueprint path relative to current blueprint.
    
    Args:
        blueprint_path: Path to current blueprint file
        base_blueprint_name: Name of base blueprint (without .yaml extension)
        
    Returns:
        Resolved path to base blueprint file
    """
    blueprint_dir = blueprint_path.parent
    
    # Try several resolution strategies
    candidates = [
        blueprint_dir / f"{base_blueprint_name}.yaml",
        blueprint_dir / f"{base_blueprint_name}.yml",
        blueprint_dir / "base" / f"{base_blueprint_name}.yaml",
        blueprint_dir / "base" / f"{base_blueprint_name}.yml",
        blueprint_dir.parent / f"{base_blueprint_name}.yaml",
        blueprint_dir.parent / f"{base_blueprint_name}.yml",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # If not found, raise informative error
    raise BlueprintInheritanceError(
        f"Base blueprint '{base_blueprint_name}' not found. Searched:\n" +
        "\n".join(f"  - {candidate}" for candidate in candidates)
    )


def validate_inheritance_chain(blueprint_path: Path, visited: Optional[set] = None) -> List[str]:
    """
    Validate inheritance chain for circular dependencies.
    
    Args:
        blueprint_path: Path to blueprint to validate
        visited: Set of already visited blueprint paths (for recursion)
        
    Returns:
        List of blueprint names in inheritance chain
        
    Raises:
        BlueprintInheritanceError: If circular dependency detected
    """
    if visited is None:
        visited = set()
    
    # Convert to absolute path for comparison
    abs_path = blueprint_path.resolve()
    
    if abs_path in visited:
        raise BlueprintInheritanceError(
            f"Circular inheritance detected: {abs_path} already in inheritance chain"
        )
    
    visited.add(abs_path)
    
    # Load blueprint to check for base
    try:
        import yaml
        with open(blueprint_path, 'r') as f:
            data = yaml.safe_load(f)
        
        base_blueprint = data.get('base_blueprint')
        if base_blueprint:
            base_path = resolve_blueprint_path(blueprint_path, base_blueprint)
            chain = validate_inheritance_chain(base_path, visited.copy())
            chain.append(data.get('name', str(blueprint_path.stem)))
            return chain
        else:
            return [data.get('name', str(blueprint_path.stem))]
    
    except Exception as e:
        raise BlueprintInheritanceError(f"Failed to validate inheritance for {blueprint_path}: {e}")