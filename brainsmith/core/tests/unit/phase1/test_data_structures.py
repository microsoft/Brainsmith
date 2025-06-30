"""
Unit tests for Phase 1 data structures.
"""

import pytest
from brainsmith.core.phase1.data_structures import (
    ProcessingStep,
    HWCompilerSpace,
    ProcessingSpace,
    SearchConstraint,
    SearchConfig,
    GlobalConfig,
    DesignSpace,
    SearchStrategy,
    OutputStage,
)


# KernelOption and TransformOption classes have been removed
# Kernels are now represented as tuples: (name, backends)
# Transforms are now represented as strings


class TestProcessingStep:
    """Test ProcessingStep dataclass."""
    
    def test_create_processing_step(self):
        step = ProcessingStep(
            name="normalization",
            type="preprocessing",
            parameters={"method": "standard"},
            enabled=True
        )
        assert step.name == "normalization"
        assert step.type == "preprocessing"
        assert step.parameters == {"method": "standard"}
        assert step.enabled == True
    
    def test_processing_step_defaults(self):
        step = ProcessingStep(name="analysis", type="postprocessing")
        assert step.parameters == {}
        assert step.enabled == True


class TestHWCompilerSpace:
    """Test HWCompilerSpace dataclass."""
    
    def test_create_hw_compiler_space(self):
        space = HWCompilerSpace(
            kernels=["MatMul", ("Softmax", ["hls", "rtl"])],
            transforms=["quantization", "folding"],
            build_steps=["ConvertToHW", "PrepareIP"],
            config_flags={"device": "U250"}
        )
        assert len(space.kernels) == 2
        assert len(space.transforms) == 2
        assert len(space.build_steps) == 2
    
    def test_parse_simple_kernel(self):
        space = HWCompilerSpace(
            kernels=["MatMul"],
            transforms=[],
            build_steps=[]
        )
        kernel_combos = space.get_kernel_combinations()
        assert len(kernel_combos) == 1
        assert len(kernel_combos[0]) == 1
        assert kernel_combos[0][0] == ("MatMul", ["*"])
    
    def test_parse_kernel_with_backends(self):
        space = HWCompilerSpace(
            kernels=[("MatMul", ["rtl", "hls"])],
            transforms=[],
            build_steps=[]
        )
        kernel_combos = space.get_kernel_combinations()
        assert len(kernel_combos) == 2  # One for each backend
        assert kernel_combos[0][0] == ("MatMul", ["rtl"])
        assert kernel_combos[1][0] == ("MatMul", ["hls"])
    
    def test_parse_mutually_exclusive_kernels(self):
        space = HWCompilerSpace(
            kernels=[["LayerNorm", "RMSNorm"]],
            transforms=[],
            build_steps=[]
        )
        kernel_combos = space.get_kernel_combinations()
        assert len(kernel_combos) == 2
    
    def test_parse_optional_kernel(self):
        space = HWCompilerSpace(
            kernels=[[None, "Transpose"]],  # Optional kernel in mutually exclusive list
            transforms=[],
            build_steps=[]
        )
        kernel_combos = space.get_kernel_combinations()
        assert len(kernel_combos) == 2  # Either None or Transpose
    
    def test_parse_transforms(self):
        space = HWCompilerSpace(
            kernels=[],
            transforms=["quantization", "~folding"],
            build_steps=[]
        )
        transform_combos = space.get_transform_combinations()
        assert len(transform_combos) == 2  # quantization and (folding or skipped)
    
    def test_parse_phase_based_transforms(self):
        space = HWCompilerSpace(
            kernels=[],
            transforms={
                "pre_hw": ["quantization"],
                "post_hw": ["optimization"]
            },
            build_steps=[]
        )
        transform_combos = space.get_transform_combinations()
        assert len(transform_combos) == 1
        # Should have both transforms
        assert transform_combos[0][0] == "quantization"
        assert transform_combos[0][1] == "optimization"


class TestSearchConstraint:
    """Test SearchConstraint dataclass."""
    
    def test_create_constraint(self):
        constraint = SearchConstraint(
            metric="lut_utilization",
            operator="<=",
            value=0.85
        )
        assert constraint.metric == "lut_utilization"
        assert constraint.operator == "<="
        assert constraint.value == 0.85
    
    def test_evaluate_constraint(self):
        constraint = SearchConstraint(
            metric="throughput",
            operator=">=",
            value=1000
        )
        assert constraint.evaluate(1500) == True
        assert constraint.evaluate(500) == False
        assert constraint.evaluate(1000) == True
    
    def test_all_operators(self):
        test_cases = [
            ("<=", 10, 5, True),
            (">=", 10, 15, True),
            ("==", 10, 10, True),
            ("<", 10, 5, True),
            (">", 10, 15, True),
        ]
        
        for op, target, test_val, expected in test_cases:
            constraint = SearchConstraint("metric", op, target)
            assert constraint.evaluate(test_val) == expected


class TestDesignSpace:
    """Test DesignSpace dataclass."""
    
    def test_create_design_space(self):
        hw_space = HWCompilerSpace(
            kernels=["MatMul"],
            transforms=["quantization"],
            build_steps=["ConvertToHW"],
            config_flags={}
        )
        
        proc_space = ProcessingSpace()
        
        search_config = SearchConfig(
            strategy=SearchStrategy.EXHAUSTIVE,
            constraints=[],
            parallel_builds=1
        )
        
        global_config = GlobalConfig(
            output_stage=OutputStage.RTL,
            working_directory="./builds"
        )
        
        design_space = DesignSpace(
            model_path="model.onnx",
            hw_compiler_space=hw_space,
            processing_space=proc_space,
            search_config=search_config,
            global_config=global_config
        )
        
        assert design_space.model_path == "model.onnx"
        assert design_space.get_total_combinations() == 1
    
    def test_combination_counting(self):
        # Create a design space with multiple options
        hw_space = HWCompilerSpace(
            kernels=["MatMul", ("Softmax", ["hls", "rtl"])],  # 1 * 1 = 1 combo
            transforms=["quantization", ["fold1", "fold2"]],   # 1 * 2 = 2 combos
            build_steps=["ConvertToHW"],
            config_flags={}
        )
        
        proc_space = ProcessingSpace(
            preprocessing=[[
                ProcessingStep("norm", "preprocessing", {"method": "a"}),
                ProcessingStep("norm", "preprocessing", {"method": "b"})
            ]],  # 2 options
            postprocessing=[]
        )
        
        search_config = SearchConfig(strategy=SearchStrategy.EXHAUSTIVE)
        global_config = GlobalConfig(
            output_stage=OutputStage.RTL,
            working_directory="./builds"
        )
        
        design_space = DesignSpace(
            model_path="model.onnx",
            hw_compiler_space=hw_space,
            processing_space=proc_space,
            search_config=search_config,
            global_config=global_config
        )
        
        # Total should be: 2 (kernels) * 2 (transforms) * 2 (preprocessing) * 1 (postprocessing) = 8
        # Kernels: MatMul + Softmax[hls], MatMul + Softmax[rtl] = 2 combinations
        # Transforms: quantization + fold1, quantization + fold2 = 2 combinations
        # Preprocessing: norm(method=a), norm(method=b) = 2 combinations
        assert design_space.get_total_combinations() == 8