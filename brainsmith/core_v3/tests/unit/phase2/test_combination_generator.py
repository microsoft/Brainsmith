"""Unit tests for the combination generator."""

import pytest
from datetime import datetime

from brainsmith.core_v3.phase2.combination_generator import CombinationGenerator
from brainsmith.core_v3.phase1.data_structures import (
    DesignSpace,
    HWCompilerSpace,
    ProcessingSpace,
    SearchConfig,
    SearchStrategy,
    GlobalConfig,
    ProcessingStep,
    SearchConstraint,
)


class TestCombinationGenerator:
    """Test the CombinationGenerator class."""
    
    def test_generate_simple_combinations(self):
        """Test generating combinations from a simple design space."""
        # Create design space
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=["Gemm", "Conv"],
                transforms={"default": ["quantize", "fold"]},
                build_steps=["synth", "opt"],
                config_flags={"target": "xcu250"}
            ),
            processing_space=ProcessingSpace(
                preprocessing=[],
                postprocessing=[]
            ),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig()
        )
        
        # Generate combinations
        generator = CombinationGenerator()
        configs = generator.generate_all(design_space)
        
        # Should have 1 combination (both kernels and both transforms together)
        assert len(configs) == 1
        
        # Check all combinations are unique
        config_ids = [c.id for c in configs]
        assert len(config_ids) == len(set(config_ids))
        
        # Check combination indices
        for i, config in enumerate(configs):
            assert config.combination_index == i
            assert config.total_combinations == 1
    
    def test_generate_with_kernel_backends(self):
        """Test generating combinations with kernel backend options."""
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=[
                    ("Gemm", ["rtl", "hls"]),
                    ("Conv", ["hls"])
                ],
                transforms={"default": ["quantize"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig()
        )
        
        generator = CombinationGenerator()
        configs = generator.generate_all(design_space)
        
        # Should have 2 backends for Gemm * 1 backend for Conv * 1 transform = 2
        assert len(configs) == 2
        
        # Check kernel configurations
        kernel_configs = [tuple(c.kernels) for c in configs]
        expected = [
            (("Gemm", ["rtl"]), ("Conv", ["hls"])),
            (("Gemm", ["hls"]), ("Conv", ["hls"])),
        ]
        
        for expected_config in expected:
            assert expected_config in kernel_configs
    
    def test_generate_with_optional_transforms(self):
        """Test generating combinations with optional transforms."""
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=["Gemm"],
                transforms={"default": ["quantize", "fold", "~streamline"]},  # ~ prefix for optional
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig()
        )
        
        generator = CombinationGenerator()
        configs = generator.generate_all(design_space)
        
        # Should have 1 kernel * 2 options for optional transform = 2
        assert len(configs) == 2
        
        # Check transform configurations
        transform_lists = [c.transforms.get("default", []) for c in configs]
        
        # One with optional, one without
        assert ["quantize", "fold"] in transform_lists or ["quantize", "fold", ""] in transform_lists
        assert ["quantize", "fold", "streamline"] in transform_lists
    
    def test_generate_with_processing_steps(self):
        """Test generating combinations with processing steps."""
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=["Gemm"],
                transforms={"default": ["quantize"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(
                preprocessing=[
                    [ProcessingStep("resize", "transform", {"size": 224}, enabled=True)],
                    [ProcessingStep("normalize", "transform", {}, enabled=False)]
                ],
                postprocessing=[
                    [ProcessingStep("softmax", "activation", {})]
                ]
            ),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig()
        )
        
        generator = CombinationGenerator()
        configs = generator.generate_all(design_space)
        
        # Should have 1 combination (only enabled preprocessing)
        assert len(configs) == 1
        
        config = configs[0]
        assert len(config.preprocessing) == 1
        assert config.preprocessing[0].name == "resize"
        assert len(config.postprocessing) == 1
        assert config.postprocessing[0].name == "softmax"
    
    def test_generate_design_space_id(self):
        """Test design space ID generation."""
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=["Gemm", "Conv"],
                transforms={"default": ["quantize"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(
                strategy=SearchStrategy.EXHAUSTIVE,
                constraints=[
                    SearchConstraint("throughput", ">", 1000),
                    SearchConstraint("latency", "<", 10)
                ]
            ),
            global_config=GlobalConfig()
        )
        
        generator = CombinationGenerator()
        ds_id = generator._generate_design_space_id(design_space)
        
        # Should be a valid ID format
        assert ds_id.startswith("dse_")
        assert len(ds_id) == 12  # dse_ + 8 hex chars
        
        # Should be deterministic
        ds_id2 = generator._generate_design_space_id(design_space)
        assert ds_id == ds_id2
    
    def test_filter_by_indices(self):
        """Test filtering configurations by indices."""
        # Create a simple design space
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=[["K1", "K2", "K3"]],  # Mutually exclusive kernels
                transforms={"default": ["T1", "T2"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig()
        )
        
        generator = CombinationGenerator()
        all_configs = generator.generate_all(design_space)
        
        # Filter to specific indices
        filtered = generator.filter_by_indices(all_configs, [0, 2])
        
        assert len(filtered) == 2
        assert filtered[0].combination_index == 0
        assert filtered[1].combination_index == 2
        
        # Test with out-of-range indices
        filtered2 = generator.filter_by_indices(all_configs, [0, 100, -1])
        assert len(filtered2) == 1
        assert filtered2[0].combination_index == 0
    
    def test_filter_by_resume(self):
        """Test filtering configurations for resume."""
        # Create design space
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=[["K1", "K2"]],  # Mutually exclusive kernels
                transforms={"default": ["T1", "T2"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig()
        )
        
        generator = CombinationGenerator()
        all_configs = generator.generate_all(design_space)
        
        # Resume from middle (first config completed, second remains)
        last_completed_id = all_configs[0].id
        remaining = generator.filter_by_resume(all_configs, last_completed_id)
        
        assert len(remaining) == 1  # config at index 1
        assert remaining[0].combination_index == 1
        
        # Resume from non-existent ID should return all
        remaining2 = generator.filter_by_resume(all_configs, "non_existent_id")
        assert len(remaining2) == len(all_configs)
    
    def test_empty_kernel_filtering(self):
        """Test that empty kernels are filtered out."""
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=["Gemm", [None, "Conv"]],  # Gemm required, Conv optional
                transforms={"default": ["quantize"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig()
        )
        
        generator = CombinationGenerator()
        configs = generator.generate_all(design_space)
        
        # Should have 2 configs: one with Conv, one without
        assert len(configs) == 2
        
        # Check kernel counts
        kernel_counts = [len(c.kernels) for c in configs]
        assert 1 in kernel_counts  # Just Gemm
        assert 2 in kernel_counts  # Gemm and Conv
    
    def test_output_dir_generation(self):
        """Test that output directories are generated correctly with dynamic padding."""
        # Test with small number of combinations (< 10)
        design_space = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=[["K1", "K2"]],  # 2 combinations
                transforms={"default": ["T1"]},
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig(working_directory="/tmp/test")
        )
        
        generator = CombinationGenerator()
        configs = generator.generate_all(design_space)
        
        assert len(configs) == 2
        # Should use single digit padding
        assert configs[0].output_dir.endswith("builds/config_0")
        assert configs[1].output_dir.endswith("builds/config_1")
        
        # Test with larger number of combinations (100+)
        design_space2 = DesignSpace(
            model_path="/path/to/model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=[["K1", "K2", "K3", "K4", "K5"]],  # 5 options
                transforms={
                    "stage1": [["T1", "T2", "T3", "T4"]],  # 4 options
                    "stage2": [["T5", "T6", "T7", "T8", "T9"]]  # 5 options
                },  # Total: 5 * 4 * 5 = 100 combinations
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(),
            search_config=SearchConfig(strategy=SearchStrategy.EXHAUSTIVE),
            global_config=GlobalConfig(working_directory="/tmp/test")
        )
        
        configs2 = generator.generate_all(design_space2)
        
        assert len(configs2) == 100
        # Should use 2 digit padding (99 needs 2 digits)
        assert configs2[0].output_dir.endswith("builds/config_00")
        assert configs2[50].output_dir.endswith("builds/config_50")
        assert configs2[99].output_dir.endswith("builds/config_99")
        
        # Verify full path structure
        assert "/tmp/test/" in configs2[0].output_dir
        assert "/builds/config_" in configs2[0].output_dir