"""
Component Combination Generator for DSE V2

Generates valid component combinations from Blueprint V2 design spaces,
respecting exploration rules (required, optional, mutually_exclusive, dependencies).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Optional, Tuple, Iterator
from itertools import product, combinations
import logging
from copy import deepcopy

from ..blueprint import DesignSpaceDefinition, ComponentSpace, ExplorationRules

logger = logging.getLogger(__name__)


@dataclass
class ComponentCombination:
    """Represents a specific combination of components for all entrypoints."""
    
    # Node components (Entrypoints 1, 3, 4)
    canonical_ops: List[str] = field(default_factory=list)
    hw_kernels: Dict[str, str] = field(default_factory=dict)  # component -> chosen option
    
    # Transform components (Entrypoints 2, 5, 6)
    model_topology: List[str] = field(default_factory=list)
    hw_kernel_transforms: List[str] = field(default_factory=list)
    hw_graph_transforms: List[str] = field(default_factory=list)
    
    # Metadata
    combination_id: Optional[str] = None
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate combination ID if not provided."""
        if self.combination_id is None:
            self.combination_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for this combination."""
        components = []
        
        # Add canonical ops
        if self.canonical_ops:
            components.append(f"ops_{'-'.join(sorted(self.canonical_ops))}")
        
        # Add hw kernels with options
        if self.hw_kernels:
            kernel_strs = [f"{k}:{v}" for k, v in sorted(self.hw_kernels.items())]
            components.append(f"kernels_{'-'.join(kernel_strs)}")
        
        # Add transforms
        for transform_type, transforms in [
            ("topo", self.model_topology),
            ("hw", self.hw_kernel_transforms),
            ("graph", self.hw_graph_transforms)
        ]:
            if transforms:
                components.append(f"{transform_type}_{'-'.join(sorted(transforms))}")
        
        return "_".join(components) if components else "empty_combination"
    
    def get_all_components(self) -> Dict[str, List[str]]:
        """Get all components organized by category."""
        return {
            'canonical_ops': self.canonical_ops,
            'hw_kernels': list(self.hw_kernels.keys()),
            'hw_kernel_options': list(self.hw_kernels.values()),
            'model_topology': self.model_topology,
            'hw_kernel_transforms': self.hw_kernel_transforms,
            'hw_graph_transforms': self.hw_graph_transforms
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'combination_id': self.combination_id,
            'canonical_ops': self.canonical_ops,
            'hw_kernels': self.hw_kernels,
            'model_topology': self.model_topology,
            'hw_kernel_transforms': self.hw_kernel_transforms,
            'hw_graph_transforms': self.hw_graph_transforms,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors
        }
    
    def __hash__(self) -> int:
        """Make hashable for deduplication."""
        return hash(self.combination_id)
    
    def __eq__(self, other) -> bool:
        """Equality comparison for deduplication."""
        if not isinstance(other, ComponentCombination):
            return False
        return self.combination_id == other.combination_id


class CombinationGenerator:
    """Generates valid component combinations from design space definitions."""
    
    def __init__(self, design_space: DesignSpaceDefinition):
        """Initialize with design space definition."""
        self.design_space = design_space
        self.generated_combinations: Set[str] = set()
        
        logger.info(f"Initialized combination generator for blueprint: {design_space.name}")
    
    def generate_all_combinations(self, max_combinations: Optional[int] = None) -> List[ComponentCombination]:
        """
        Generate all valid component combinations from the design space.
        
        Args:
            max_combinations: Maximum number of combinations to generate (None for no limit)
            
        Returns:
            List of valid component combinations
        """
        logger.info("Starting generation of all valid component combinations")
        
        combinations = []
        
        # Generate node combinations
        node_combinations = self._generate_node_combinations()
        logger.info(f"Generated {len(node_combinations)} node combinations")
        
        # Generate transform combinations
        transform_combinations = self._generate_transform_combinations()
        logger.info(f"Generated {len(transform_combinations)} transform combinations")
        
        # Cross-product of node and transform combinations
        total_possible = len(node_combinations) * len(transform_combinations)
        logger.info(f"Total possible combinations: {total_possible}")
        
        if max_combinations and total_possible > max_combinations:
            logger.warning(f"Limiting to {max_combinations} combinations out of {total_possible}")
        
        count = 0
        for node_combo, transform_combo in product(node_combinations, transform_combinations):
            if max_combinations and count >= max_combinations:
                break
            
            # Merge node and transform combinations
            combination = ComponentCombination(
                canonical_ops=node_combo['canonical_ops'],
                hw_kernels=node_combo['hw_kernels'],
                model_topology=transform_combo['model_topology'],
                hw_kernel_transforms=transform_combo['hw_kernel'],
                hw_graph_transforms=transform_combo['hw_graph']
            )
            
            # Validate the complete combination
            if self._validate_combination(combination):
                combinations.append(combination)
                count += 1
        
        # Deduplicate combinations
        unique_combinations = self._deduplicate_combinations(combinations)
        
        logger.info(f"Generated {len(unique_combinations)} unique valid combinations")
        return unique_combinations
    
    def _generate_node_combinations(self) -> List[Dict[str, Any]]:
        """Generate all valid node component combinations."""
        combinations = []
        
        # Generate canonical ops combinations
        canonical_ops_combos = self._generate_component_combinations(
            self.design_space.nodes.canonical_ops
        )
        
        # Generate hw kernels combinations
        hw_kernels_combos = self._generate_hw_kernel_combinations(
            self.design_space.nodes.hw_kernels
        )
        
        # Cross-product of canonical ops and hw kernels
        for canonical_combo in canonical_ops_combos:
            for hw_kernel_combo in hw_kernels_combos:
                combinations.append({
                    'canonical_ops': canonical_combo,
                    'hw_kernels': hw_kernel_combo
                })
        
        return combinations
    
    def _generate_transform_combinations(self) -> List[Dict[str, Any]]:
        """Generate all valid transform component combinations."""
        combinations = []
        
        # Generate combinations for each transform type
        model_topology_combos = self._generate_component_combinations(
            self.design_space.transforms.model_topology
        )
        
        hw_kernel_combos = self._generate_component_combinations(
            self.design_space.transforms.hw_kernel
        )
        
        hw_graph_combos = self._generate_component_combinations(
            self.design_space.transforms.hw_graph
        )
        
        # Cross-product of all transform types
        for topo_combo in model_topology_combos:
            for hw_combo in hw_kernel_combos:
                for graph_combo in hw_graph_combos:
                    combinations.append({
                        'model_topology': topo_combo,
                        'hw_kernel': hw_combo,
                        'hw_graph': graph_combo
                    })
        
        return combinations
    
    def _generate_component_combinations(self, component_space: ComponentSpace) -> List[List[str]]:
        """
        Generate all valid combinations for a component space.
        
        Args:
            component_space: Component space with available components and exploration rules
            
        Returns:
            List of component combinations (each combination is a list of component names)
        """
        if not component_space.available:
            return [[]]  # Empty combination for empty spaces
        
        rules = component_space.exploration
        if not rules:
            # No exploration rules - return all individual components
            return [[comp] for comp in component_space.get_component_names()]
        
        # Get all component names
        all_components = set(component_space.get_component_names())
        
        # Start with required components
        base_components = set(rules.required)
        
        # Generate combinations with optional components
        optional_components = set(rules.optional)
        optional_powerset = self._powerset(optional_components)
        
        combinations = []
        
        for optional_subset in optional_powerset:
            candidate_components = base_components | set(optional_subset)
            
            # Apply mutually exclusive constraints
            valid_exclusive_combinations = self._apply_mutually_exclusive(
                candidate_components, rules.mutually_exclusive
            )
            
            for valid_combo in valid_exclusive_combinations:
                # Check dependencies
                if self._check_dependencies(valid_combo, rules.dependencies):
                    combinations.append(list(valid_combo))
        
        return combinations if combinations else [[]]
    
    def _generate_hw_kernel_combinations(self, hw_kernels_space: ComponentSpace) -> List[Dict[str, str]]:
        """
        Generate hardware kernel combinations with option selection.
        
        Returns:
            List of dictionaries mapping component -> chosen option
        """
        if not hw_kernels_space.available:
            return [{}]
        
        # Get component lists first
        component_combinations = self._generate_component_combinations(hw_kernels_space)
        
        kernel_combinations = []
        
        for component_list in component_combinations:
            # For each component in the combination, get its options
            option_choices = []
            components = []
            
            for component in component_list:
                options = hw_kernels_space.get_component_options(component)
                if options:
                    option_choices.append(options)
                    components.append(component)
            
            if not components:
                kernel_combinations.append({})
                continue
            
            # Generate all option combinations
            for option_combo in product(*option_choices):
                kernel_dict = dict(zip(components, option_combo))
                kernel_combinations.append(kernel_dict)
        
        return kernel_combinations if kernel_combinations else [{}]
    
    def _powerset(self, iterable) -> Iterator[Tuple]:
        """Generate powerset of an iterable (all possible subsets)."""
        items = list(iterable)
        for i in range(len(items) + 1):
            yield from combinations(items, i)
    
    def _apply_mutually_exclusive(self, components: Set[str], 
                                 exclusive_groups: List[List[str]]) -> List[Set[str]]:
        """
        Apply mutually exclusive constraints to component set.
        
        Args:
            components: Set of candidate components
            exclusive_groups: List of mutually exclusive groups
            
        Returns:
            List of valid component sets after applying constraints
        """
        if not exclusive_groups:
            return [components]
        
        valid_combinations = [components]
        
        for group in exclusive_groups:
            new_combinations = []
            
            for combo in valid_combinations:
                # Find which components from this group are in the combination
                group_components = [comp for comp in group if comp in combo]
                
                if len(group_components) <= 1:
                    # No conflict or only one component from group
                    new_combinations.append(combo)
                else:
                    # Multiple components from exclusive group - generate alternatives
                    base_combo = combo - set(group_components)
                    
                    # Add combinations with each individual group component
                    for group_comp in group_components:
                        new_combo = base_combo | {group_comp}
                        new_combinations.append(new_combo)
                    
                    # Also add combination with no components from this group
                    new_combinations.append(base_combo)
            
            valid_combinations = new_combinations
        
        # Remove duplicates
        unique_combinations = []
        seen = set()
        for combo in valid_combinations:
            combo_tuple = tuple(sorted(combo))
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                unique_combinations.append(combo)
        
        return unique_combinations
    
    def _check_dependencies(self, components: Set[str], 
                           dependencies: Dict[str, List[str]]) -> bool:
        """
        Check if component set satisfies all dependencies.
        
        Args:
            components: Set of components to check
            dependencies: Dictionary of component -> list of required components
            
        Returns:
            True if all dependencies are satisfied
        """
        for component, required_deps in dependencies.items():
            if component in components:
                # Component is included, check its dependencies
                for dep in required_deps:
                    if dep not in components:
                        return False
        
        return True
    
    def _validate_combination(self, combination: ComponentCombination) -> bool:
        """
        Validate a complete component combination.
        
        Args:
            combination: Component combination to validate
            
        Returns:
            True if combination is valid
        """
        errors = []
        
        # Check that combination is not empty
        all_components = combination.get_all_components()
        total_components = sum(len(comp_list) for comp_list in all_components.values() 
                             if isinstance(comp_list, list))
        
        if total_components == 0:
            errors.append("Combination cannot be completely empty")
        
        # Validate against design space
        # (Additional validation rules can be added here)
        
        # Update combination with validation results
        combination.is_valid = len(errors) == 0
        combination.validation_errors = errors
        
        return combination.is_valid
    
    def _deduplicate_combinations(self, combinations: List[ComponentCombination]) -> List[ComponentCombination]:
        """Remove duplicate combinations."""
        seen_ids = set()
        unique_combinations = []
        
        for combination in combinations:
            if combination.combination_id not in seen_ids:
                seen_ids.add(combination.combination_id)
                unique_combinations.append(combination)
        
        logger.info(f"Removed {len(combinations) - len(unique_combinations)} duplicate combinations")
        return unique_combinations
    
    def generate_sample_combinations(self, n_samples: int, 
                                   strategy: str = "random") -> List[ComponentCombination]:
        """
        Generate a sample of combinations using specified strategy.
        
        Args:
            n_samples: Number of combinations to generate
            strategy: Sampling strategy ("random", "diverse", "balanced")
            
        Returns:
            List of sampled combinations
        """
        if strategy == "random":
            return self._generate_random_sample(n_samples)
        elif strategy == "diverse":
            return self._generate_diverse_sample(n_samples)
        elif strategy == "balanced":
            return self._generate_balanced_sample(n_samples)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _generate_random_sample(self, n_samples: int) -> List[ComponentCombination]:
        """Generate random sample of combinations."""
        all_combinations = self.generate_all_combinations(max_combinations=n_samples * 3)
        
        if len(all_combinations) <= n_samples:
            return all_combinations
        
        import random
        return random.sample(all_combinations, n_samples)
    
    def _generate_diverse_sample(self, n_samples: int) -> List[ComponentCombination]:
        """Generate diverse sample trying to cover different component types."""
        # Implementation would use diversity metrics
        # For now, fall back to random sampling
        return self._generate_random_sample(n_samples)
    
    def _generate_balanced_sample(self, n_samples: int) -> List[ComponentCombination]:
        """Generate balanced sample across different combination sizes."""
        # Implementation would balance small and large combinations
        # For now, fall back to random sampling
        return self._generate_random_sample(n_samples)


def generate_component_combinations(design_space: DesignSpaceDefinition,
                                  max_combinations: Optional[int] = None) -> List[ComponentCombination]:
    """
    Convenience function to generate combinations from a design space.
    
    Args:
        design_space: Blueprint V2 design space definition
        max_combinations: Maximum number of combinations to generate
        
    Returns:
        List of valid component combinations
    """
    generator = CombinationGenerator(design_space)
    return generator.generate_all_combinations(max_combinations)