"""
Unit tests for kernel and transform choice enumeration logic.
"""

import pytest
from brainsmith.core.dse.kernel_transform_selection import (
    KernelSelection, TransformSelection,
    enumerate_kernel_combinations, enumerate_transform_pipelines
)


class TestKernelEnumeration:
    """Test kernel choice enumeration."""
    
    def test_kernel_choice_enumeration_no_exclusivity(self):
        """Test kernel enumeration with no mutual exclusivity."""
        kernel_selection = KernelSelection(
            available_kernels=["conv2d_hls", "matmul_rtl"],
            mutually_exclusive_groups=[],
            operation_mappings={}
        )
        
        choices = enumerate_kernel_combinations(kernel_selection)
        
        assert len(choices) == 1
        assert sorted(choices[0]) == ["conv2d_hls", "matmul_rtl"]
    
    def test_kernel_choice_enumeration_with_exclusivity(self):
        """Test kernel enumeration with mutual exclusivity groups."""
        kernel_selection = KernelSelection(
            available_kernels=["conv2d_hls", "conv2d_rtl", "matmul_rtl"],
            mutually_exclusive_groups=[["conv2d_hls", "conv2d_rtl"]],
            operation_mappings={}
        )
        
        choices = enumerate_kernel_combinations(kernel_selection)
        
        assert len(choices) == 2
        # Should have one choice with conv2d_hls and one with conv2d_rtl
        choice_sets = [set(choice) for choice in choices]
        expected_sets = [
            {"conv2d_hls", "matmul_rtl"},
            {"conv2d_rtl", "matmul_rtl"}
        ]
        
        for expected_set in expected_sets:
            assert expected_set in choice_sets
    
    def test_kernel_choice_enumeration_multiple_exclusivity_groups(self):
        """Test kernel enumeration with multiple exclusivity groups."""
        kernel_selection = KernelSelection(
            available_kernels=["conv2d_hls", "conv2d_rtl", "matmul_hls", "matmul_rtl"],
            mutually_exclusive_groups=[
                ["conv2d_hls", "conv2d_rtl"],
                ["matmul_hls", "matmul_rtl"]
            ],
            operation_mappings={}
        )
        
        choices = enumerate_kernel_combinations(kernel_selection)
        
        assert len(choices) == 4  # 2 * 2 combinations
        choice_sets = [set(choice) for choice in choices]
        expected_sets = [
            {"conv2d_hls", "matmul_hls"},
            {"conv2d_hls", "matmul_rtl"},
            {"conv2d_rtl", "matmul_hls"},
            {"conv2d_rtl", "matmul_rtl"}
        ]
        
        for expected_set in expected_sets:
            assert expected_set in choice_sets
    
    def test_kernel_choice_enumeration_empty_available(self):
        """Test kernel enumeration with no available kernels."""
        kernel_selection = KernelSelection(
            available_kernels=[],
            mutually_exclusive_groups=[],
            operation_mappings={}
        )
        
        choices = enumerate_kernel_combinations(kernel_selection)
        
        assert choices == []
    
    def test_kernel_choice_enumeration_invalid_exclusivity_group(self):
        """Test kernel enumeration with exclusivity group containing unavailable kernels."""
        kernel_selection = KernelSelection(
            available_kernels=["conv2d_hls", "matmul_rtl"],
            mutually_exclusive_groups=[["conv2d_hls", "conv2d_rtl"]],  # conv2d_rtl not available
            operation_mappings={}
        )
        
        choices = enumerate_kernel_combinations(kernel_selection)
        
        # Should still work, but only include available kernels
        assert len(choices) == 1
        assert sorted(choices[0]) == ["conv2d_hls", "matmul_rtl"]


class TestTransformEnumeration:
    """Test transform pipeline enumeration."""
    
    def test_transform_pipeline_enumeration_core_only(self):
        """Test transform enumeration with only core pipeline."""
        transform_selection = TransformSelection(
            core_pipeline=["cleanup", "streamlining"],
            optional_transforms=[],
            mutually_exclusive_groups=[],
            hooks={}
        )
        
        variants = enumerate_transform_pipelines(transform_selection)
        
        assert len(variants) == 1
        assert variants[0] == ["cleanup", "streamlining"]
    
    def test_transform_pipeline_enumeration_with_optional(self):
        """Test transform enumeration with optional transforms."""
        transform_selection = TransformSelection(
            core_pipeline=["cleanup", "streamlining"],
            optional_transforms=["remove_head", "remove_tail"],
            mutually_exclusive_groups=[],
            hooks={}
        )
        
        variants = enumerate_transform_pipelines(transform_selection)
        
        # Should have 2^2 = 4 combinations (all subsets of optional transforms)
        assert len(variants) == 4
        
        expected_variants = [
            ["cleanup", "streamlining"],
            ["cleanup", "streamlining", "remove_head"],
            ["cleanup", "streamlining", "remove_tail"],
            ["cleanup", "streamlining", "remove_head", "remove_tail"]
        ]
        
        for expected in expected_variants:
            assert expected in variants
    
    def test_transform_pipeline_enumeration_with_mutual_exclusivity(self):
        """Test transform enumeration with mutual exclusivity."""
        transform_selection = TransformSelection(
            core_pipeline=["cleanup", "streamlining"],
            optional_transforms=[],
            mutually_exclusive_groups=[["infer_hardware", "constrain_folding"]],
            hooks={}
        )
        
        variants = enumerate_transform_pipelines(transform_selection)
        
        assert len(variants) == 2
        expected_variants = [
            ["cleanup", "streamlining", "infer_hardware"],
            ["cleanup", "streamlining", "constrain_folding"]
        ]
        
        for expected in expected_variants:
            assert expected in variants
    
    def test_transform_pipeline_enumeration_complex(self):
        """Test transform enumeration with optional and exclusive transforms."""
        transform_selection = TransformSelection(
            core_pipeline=["cleanup", "streamlining"],
            optional_transforms=["remove_head"],
            mutually_exclusive_groups=[["infer_hardware", "constrain_folding"]],
            hooks={}
        )
        
        variants = enumerate_transform_pipelines(transform_selection)
        
        # Should have 2 optional * 2 exclusive = 4 combinations
        assert len(variants) == 4
        
        expected_variants = [
            ["cleanup", "streamlining", "infer_hardware"],
            ["cleanup", "streamlining", "constrain_folding"],
            ["cleanup", "streamlining", "remove_head", "infer_hardware"],
            ["cleanup", "streamlining", "remove_head", "constrain_folding"]
        ]
        
        for expected in expected_variants:
            assert expected in variants
    
    def test_transform_pipeline_enumeration_empty_core(self):
        """Test transform enumeration with empty core pipeline."""
        transform_selection = TransformSelection(
            core_pipeline=[],
            optional_transforms=["remove_head"],
            mutually_exclusive_groups=[],
            hooks={}
        )
        
        variants = enumerate_transform_pipelines(transform_selection)
        
        assert len(variants) == 2
        expected_variants = [
            [],
            ["remove_head"]
        ]
        
        for expected in expected_variants:
            assert expected in variants
    
    def test_transform_pipeline_enumeration_duplicate_removal(self):
        """Test that duplicate variants are removed."""
        transform_selection = TransformSelection(
            core_pipeline=["cleanup"],
            optional_transforms=["cleanup"],  # Duplicate with core
            mutually_exclusive_groups=[],
            hooks={}
        )
        
        variants = enumerate_transform_pipelines(transform_selection)
        
        # Should remove duplicates
        assert len(variants) == 2
        assert ["cleanup"] in variants
        assert ["cleanup", "cleanup"] in variants


if __name__ == "__main__":
    pytest.main([__file__])