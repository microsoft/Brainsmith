"""
Comprehensive tests for Component Combination Generator.
"""

import pytest
from typing import List, Dict, Set

from brainsmith.core.dse_v2.combination_generator import (
    ComponentCombination, CombinationGenerator, generate_component_combinations
)
from brainsmith.core.blueprint_v2 import (
    DesignSpaceDefinition, NodeDesignSpace, TransformDesignSpace,
    ComponentSpace, ExplorationRules
)


class TestComponentCombination:
    """Test ComponentCombination dataclass."""
    
    def test_empty_combination(self):
        """Test creation of empty combination."""
        combo = ComponentCombination()
        
        assert combo.canonical_ops == []
        assert combo.hw_kernels == {}
        assert combo.model_topology == []
        assert combo.is_valid == True
        assert combo.combination_id == "empty_combination"
    
    def test_combination_id_generation(self):
        """Test automatic ID generation."""
        combo = ComponentCombination(
            canonical_ops=["LayerNorm", "Softmax"],
            hw_kernels={"MatMul": "matmul_hls", "Conv2D": "conv2d_hls"},
            model_topology=["cleanup", "streamlining"]
        )
        
        # ID should include all components
        assert "ops_LayerNorm-Softmax" in combo.combination_id
        assert "kernels_Conv2D:conv2d_hls-MatMul:matmul_hls" in combo.combination_id
        assert "topo_cleanup-streamlining" in combo.combination_id
    
    def test_combination_equality_and_hashing(self):
        """Test combination equality and hashing for deduplication."""
        combo1 = ComponentCombination(
            canonical_ops=["LayerNorm"],
            hw_kernels={"MatMul": "matmul_hls"}
        )
        
        combo2 = ComponentCombination(
            canonical_ops=["LayerNorm"],
            hw_kernels={"MatMul": "matmul_hls"}
        )
        
        # Should be equal and have same hash
        assert combo1 == combo2
        assert hash(combo1) == hash(combo2)
        
        # Should be usable in sets for deduplication
        combo_set = {combo1, combo2}
        assert len(combo_set) == 1
    
    def test_get_all_components(self):
        """Test getting all components organized by category."""
        combo = ComponentCombination(
            canonical_ops=["LayerNorm", "Softmax"],
            hw_kernels={"MatMul": "matmul_hls", "Conv2D": "conv2d_hls"},
            model_topology=["cleanup"],
            hw_kernel_transforms=["apply_folding"],
            hw_graph_transforms=["set_fifo_depths"]
        )
        
        all_components = combo.get_all_components()
        
        assert all_components['canonical_ops'] == ["LayerNorm", "Softmax"]
        assert set(all_components['hw_kernels']) == {"MatMul", "Conv2D"}
        assert set(all_components['hw_kernel_options']) == {"matmul_hls", "conv2d_hls"}
        assert all_components['model_topology'] == ["cleanup"]
        assert all_components['hw_kernel_transforms'] == ["apply_folding"]
        assert all_components['hw_graph_transforms'] == ["set_fifo_depths"]
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        combo = ComponentCombination(
            canonical_ops=["LayerNorm"],
            hw_kernels={"MatMul": "matmul_hls"},
            validation_errors=["test error"]
        )
        
        combo_dict = combo.to_dict()
        
        assert combo_dict['canonical_ops'] == ["LayerNorm"]
        assert combo_dict['hw_kernels'] == {"MatMul": "matmul_hls"}
        assert combo_dict['is_valid'] == True
        assert combo_dict['validation_errors'] == ["test error"]
        assert 'combination_id' in combo_dict


class TestCombinationGenerator:
    """Test CombinationGenerator class."""
    
    def create_simple_design_space(self) -> DesignSpaceDefinition:
        """Create simple design space for testing."""
        return DesignSpaceDefinition(
            name="test_space",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["LayerNorm", "Softmax"],
                    exploration=ExplorationRules(
                        required=["LayerNorm"],
                        optional=["Softmax"]
                    )
                ),
                hw_kernels=ComponentSpace(
                    available=[{"MatMul": ["matmul_hls", "matmul_rtl"]}],
                    exploration=ExplorationRules(required=["MatMul"])
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["cleanup", "streamlining"],
                    exploration=ExplorationRules(required=["cleanup"])
                )
            )
        )
    
    def test_simple_combination_generation(self):
        """Test basic combination generation."""
        design_space = self.create_simple_design_space()
        generator = CombinationGenerator(design_space)
        
        combinations = generator.generate_all_combinations()
        
        # Should have combinations covering required/optional choices and kernel options
        assert len(combinations) > 0
        
        # All combinations should have required components
        for combo in combinations:
            assert "LayerNorm" in combo.canonical_ops
            assert "MatMul" in combo.hw_kernels
            assert "cleanup" in combo.model_topology
    
    def test_optional_component_handling(self):
        """Test that optional components generate with/without variants."""
        design_space = self.create_simple_design_space()
        generator = CombinationGenerator(design_space)
        
        combinations = generator.generate_all_combinations()
        
        # Should have combinations with and without optional Softmax
        has_softmax = any("Softmax" in combo.canonical_ops for combo in combinations)
        no_softmax = any("Softmax" not in combo.canonical_ops for combo in combinations)
        
        assert has_softmax, "Should have combinations with optional Softmax"
        assert no_softmax, "Should have combinations without optional Softmax"
    
    def test_hw_kernel_option_selection(self):
        """Test hardware kernel option selection."""
        design_space = self.create_simple_design_space()
        generator = CombinationGenerator(design_space)
        
        combinations = generator.generate_all_combinations()
        
        # Should have combinations with different MatMul options
        hls_variants = [combo for combo in combinations if combo.hw_kernels.get("MatMul") == "matmul_hls"]
        rtl_variants = [combo for combo in combinations if combo.hw_kernels.get("MatMul") == "matmul_rtl"]
        
        assert len(hls_variants) > 0, "Should have HLS MatMul variants"
        assert len(rtl_variants) > 0, "Should have RTL MatMul variants"
    
    def test_mutually_exclusive_constraints(self):
        """Test mutually exclusive constraint handling."""
        design_space = DesignSpaceDefinition(
            name="exclusive_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["option_a", "option_b", "option_c"],
                    exploration=ExplorationRules(
                        optional=["option_a", "option_b", "option_c"],
                        mutually_exclusive=[["option_a", "option_b"]]
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(available=["cleanup"])
            )
        )
        
        generator = CombinationGenerator(design_space)
        combinations = generator.generate_all_combinations()
        
        # Check that no combination has both option_a and option_b
        for combo in combinations:
            ops = combo.canonical_ops
            assert not ("option_a" in ops and "option_b" in ops), \
                f"Found mutually exclusive components together: {ops}"
        
        # Should still have combinations with option_a OR option_b
        has_option_a = any("option_a" in combo.canonical_ops for combo in combinations)
        has_option_b = any("option_b" in combo.canonical_ops for combo in combinations)
        
        assert has_option_a, "Should have combinations with option_a"
        assert has_option_b, "Should have combinations with option_b"
    
    def test_dependency_handling(self):
        """Test dependency constraint handling."""
        design_space = DesignSpaceDefinition(
            name="dependency_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["base", "dependent", "independent"],
                    exploration=ExplorationRules(
                        optional=["base", "dependent", "independent"],
                        dependencies={"dependent": ["base"]}
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(available=["cleanup"])
            )
        )
        
        generator = CombinationGenerator(design_space)
        combinations = generator.generate_all_combinations()
        
        # Check that dependent is never included without base
        for combo in combinations:
            ops = combo.canonical_ops
            if "dependent" in ops:
                assert "base" in ops, f"Found dependent without base: {ops}"
        
        # Should have combinations with just base (no dependent)
        has_base_only = any(
            "base" in combo.canonical_ops and "dependent" not in combo.canonical_ops
            for combo in combinations
        )
        assert has_base_only, "Should have combinations with base but not dependent"
    
    def test_complex_exploration_rules(self):
        """Test complex combination of exploration rules."""
        design_space = DesignSpaceDefinition(
            name="complex_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["required_op", "opt_a", "opt_b", "opt_c", "dependent_op"],
                    exploration=ExplorationRules(
                        required=["required_op"],
                        optional=["opt_a", "opt_b", "opt_c", "dependent_op"],
                        mutually_exclusive=[["opt_a", "opt_b"]],
                        dependencies={"dependent_op": ["opt_c"]}
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(available=["cleanup"])
            )
        )
        
        generator = CombinationGenerator(design_space)
        combinations = generator.generate_all_combinations()
        
        # Verify all combinations are valid
        for combo in combinations:
            ops = set(combo.canonical_ops)
            
            # Must have required
            assert "required_op" in ops
            
            # Must not have both mutually exclusive
            assert not ("opt_a" in ops and "opt_b" in ops)
            
            # If has dependent, must have dependency
            if "dependent_op" in ops:
                assert "opt_c" in ops
    
    def test_empty_component_spaces(self):
        """Test handling of empty component spaces."""
        design_space = DesignSpaceDefinition(
            name="empty_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(available=[]),
                hw_kernels=ComponentSpace(available=[])
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(available=["cleanup"])
            )
        )
        
        generator = CombinationGenerator(design_space)
        combinations = generator.generate_all_combinations()
        
        # Should still generate combinations (with empty node components)
        assert len(combinations) > 0
        
        for combo in combinations:
            assert combo.canonical_ops == []
            assert combo.hw_kernels == {}
            assert combo.model_topology == ["cleanup"]
    
    def test_max_combinations_limit(self):
        """Test maximum combinations limit."""
        design_space = DesignSpaceDefinition(
            name="large_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["op1", "op2", "op3", "op4"],
                    exploration=ExplorationRules(
                        optional=["op1", "op2", "op3", "op4"]
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["t1", "t2", "t3"],
                    exploration=ExplorationRules(
                        optional=["t1", "t2", "t3"]
                    )
                )
            )
        )
        
        generator = CombinationGenerator(design_space)
        
        # Without limit
        all_combinations = generator.generate_all_combinations()
        
        # With limit
        limited_combinations = generator.generate_all_combinations(max_combinations=5)
        
        assert len(all_combinations) > 5
        assert len(limited_combinations) == 5
    
    def test_combination_deduplication(self):
        """Test that duplicate combinations are removed."""
        design_space = DesignSpaceDefinition(
            name="dedup_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["LayerNorm"],
                    exploration=ExplorationRules(required=["LayerNorm"])
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["cleanup"],
                    exploration=ExplorationRules(required=["cleanup"])
                )
            )
        )
        
        generator = CombinationGenerator(design_space)
        combinations = generator.generate_all_combinations()
        
        # Should have exactly one combination (no duplicates)
        assert len(combinations) == 1
        
        # Verify combination content
        combo = combinations[0]
        assert combo.canonical_ops == ["LayerNorm"]
        assert combo.model_topology == ["cleanup"]
    
    def test_sample_generation(self):
        """Test sample generation with different strategies."""
        design_space = self.create_simple_design_space()
        generator = CombinationGenerator(design_space)
        
        # Test random sampling
        random_sample = generator.generate_sample_combinations(3, "random")
        assert len(random_sample) <= 3
        
        # Test diverse sampling (falls back to random for now)
        diverse_sample = generator.generate_sample_combinations(2, "diverse")
        assert len(diverse_sample) <= 2
        
        # Test balanced sampling (falls back to random for now)
        balanced_sample = generator.generate_sample_combinations(2, "balanced")
        assert len(balanced_sample) <= 2
        
        # Test invalid strategy
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            generator.generate_sample_combinations(1, "invalid_strategy")


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_bert_like_design_space(self):
        """Test BERT-like design space with realistic complexity."""
        design_space = DesignSpaceDefinition(
            name="bert_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["LayerNorm", "Softmax", "GELU", "MultiHeadAttention"],
                    exploration=ExplorationRules(
                        required=["LayerNorm", "MultiHeadAttention"],
                        optional=["Softmax", "GELU"]
                    )
                ),
                hw_kernels=ComponentSpace(
                    available=[
                        {"MatMul": ["matmul_hls", "matmul_rtl", "matmul_mixed"]},
                        {"LayerNorm": ["layernorm_custom", "layernorm_builtin"]},
                        {"Softmax": ["softmax_hls", "softmax_lookup"]}
                    ],
                    exploration=ExplorationRules(
                        required=["MatMul", "LayerNorm"],
                        optional=["Softmax"],
                        mutually_exclusive=[
                            ["matmul_hls", "matmul_rtl", "matmul_mixed"],
                            ["layernorm_custom", "layernorm_builtin"],
                            ["softmax_hls", "softmax_lookup"]
                        ]
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["cleanup", "streamlining", "bert_fusion"],
                    exploration=ExplorationRules(
                        required=["cleanup"],
                        optional=["streamlining", "bert_fusion"],
                        dependencies={"bert_fusion": ["streamlining"]}
                    )
                ),
                hw_kernel=ComponentSpace(
                    available=["apply_folding", "target_fps", "bert_optimization"],
                    exploration=ExplorationRules(
                        required=["apply_folding", "target_fps"],
                        optional=["bert_optimization"]
                    )
                ),
                hw_graph=ComponentSpace(
                    available=["set_fifo_depths", "create_stitched_ip"],
                    exploration=ExplorationRules(
                        required=["set_fifo_depths", "create_stitched_ip"]
                    )
                )
            )
        )
        
        generator = CombinationGenerator(design_space)
        combinations = generator.generate_all_combinations(max_combinations=50)
        
        # Should generate valid combinations
        assert len(combinations) > 0
        
        # Verify all have required components
        for combo in combinations:
            # Required canonical ops
            assert "LayerNorm" in combo.canonical_ops
            assert "MultiHeadAttention" in combo.canonical_ops
            
            # Required hw kernels
            assert "MatMul" in combo.hw_kernels
            assert "LayerNorm" in combo.hw_kernels
            
            # Required transforms
            assert "cleanup" in combo.model_topology
            assert "apply_folding" in combo.hw_kernel_transforms
            assert "target_fps" in combo.hw_kernel_transforms
            assert "set_fifo_depths" in combo.hw_graph_transforms
            assert "create_stitched_ip" in combo.hw_graph_transforms
            
            # Check mutual exclusivity
            matmul_options = ["matmul_hls", "matmul_rtl", "matmul_mixed"]
            matmul_chosen = combo.hw_kernels.get("MatMul")
            assert matmul_chosen in matmul_options
            
            # Check dependencies
            if "bert_fusion" in combo.model_topology:
                assert "streamlining" in combo.model_topology


class TestPerformance:
    """Test performance with large design spaces."""
    
    def test_large_design_space_performance(self):
        """Test performance with larger design space."""
        # Create design space with many optional components
        large_space = DesignSpaceDefinition(
            name="large_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=[f"op_{i}" for i in range(10)],
                    exploration=ExplorationRules(
                        required=["op_0"],
                        optional=[f"op_{i}" for i in range(1, 10)]
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=[f"transform_{i}" for i in range(5)],
                    exploration=ExplorationRules(
                        required=["transform_0"],
                        optional=[f"transform_{i}" for i in range(1, 5)]
                    )
                )
            )
        )
        
        generator = CombinationGenerator(large_space)
        
        # Should be able to generate limited number efficiently
        import time
        start_time = time.time()
        combinations = generator.generate_all_combinations(max_combinations=20)
        end_time = time.time()
        
        assert len(combinations) == 20
        assert end_time - start_time < 5.0  # Should complete in reasonable time


def test_convenience_function():
    """Test convenience function for combination generation."""
    design_space = DesignSpaceDefinition(
        name="convenience_test",
        nodes=NodeDesignSpace(
            canonical_ops=ComponentSpace(
                available=["LayerNorm"],
                exploration=ExplorationRules(required=["LayerNorm"])
            )
        ),
        transforms=TransformDesignSpace(
            model_topology=ComponentSpace(
                available=["cleanup"],
                exploration=ExplorationRules(required=["cleanup"])
            )
        )
    )
    
    combinations = generate_component_combinations(design_space, max_combinations=5)
    
    assert len(combinations) > 0
    assert all(isinstance(combo, ComponentCombination) for combo in combinations)


if __name__ == "__main__":
    pytest.main([__file__])