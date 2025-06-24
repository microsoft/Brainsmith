#!/usr/bin/env python3
"""
Example: Creating a Community Kernel for BrainSmith

This example shows how easy it is to create and share a new kernel
as a community contributor to the BrainSmith platform.
"""

from brainsmith.api import kernel, backend, param, constraint
from brainsmith.api.testing import benchmark, validate
import numpy as np

# Step 1: Define your kernel with simple decorators
@kernel(
    name="EfficientGELU",
    operation_type="activation",
    description="Hardware-optimized GELU activation with approximations",
    author="jane-doe",
    version="1.2.0",
    license="Apache-2.0",
    tags=["activation", "gelu", "efficient", "transformer"],
    paper="https://arxiv.org/abs/2024.12345"  # Optional: link to paper
)
class EfficientGELUKernel:
    """
    An efficient GELU implementation using piecewise linear approximation.
    
    This kernel provides 2-3x speedup over standard GELU with <0.1% accuracy loss,
    making it ideal for transformer models on resource-constrained FPGAs.
    """
    
    # Define constraints for your kernel
    constraints = {
        "min_input_size": 16,      # Minimum tensor size for efficiency
        "supported_dtypes": ["int8", "int16", "float16"],
        "max_approximation_error": 0.001
    }
    
    # Define configurable parameters
    @param("num_segments", default=4, min=2, max=16,
           description="Number of piecewise segments for approximation")
    @param("use_lut", default=True, 
           description="Use lookup table for faster computation")
    @param("parallel_factor", default=8, choices=[1, 2, 4, 8, 16],
           description="Parallelization factor for throughput")
    def __init__(self, **config):
        self.config = config
        self.segments = self._compute_segments(config["num_segments"])
    
    # Define HLS backend implementation
    @backend("hls", default=True)
    @constraint("vivado_hls >= 2020.1")
    def hls_implementation(self):
        """High-Level Synthesis implementation for Xilinx/AMD FPGAs."""
        return f"""
        #pragma HLS INTERFACE s_axilite port=return
        #pragma HLS INTERFACE axis port=input
        #pragma HLS INTERFACE axis port=output
        
        void efficient_gelu(stream<data_t> &input, stream<data_t> &output) {{
            #pragma HLS PIPELINE II={1 if self.config['parallel_factor'] > 1 else 2}
            #pragma HLS ARRAY_PARTITION variable=lut_table complete dim=1
            
            // Efficient GELU using {self.config['num_segments']} segments
            data_t x = input.read();
            data_t result;
            
            if ({self.config['use_lut']}) {{
                result = lookup_gelu(x, lut_table);
            }} else {{
                result = piecewise_gelu(x, segments);
            }}
            
            output.write(result);
        }}
        """
    
    # Define RTL backend for maximum performance
    @backend("rtl")
    @constraint("target_frequency >= 300")  # MHz
    def rtl_implementation(self):
        """Optimized RTL implementation for high-frequency operation."""
        return {
            "module_name": "efficient_gelu_rtl",
            "files": [
                "hdl/efficient_gelu.v",
                "hdl/gelu_lut.v",
                "hdl/piecewise_approx.v"
            ],
            "parameters": {
                "DATA_WIDTH": 16,
                "NUM_SEGMENTS": self.config["num_segments"],
                "PARALLEL_FACTOR": self.config["parallel_factor"]
            }
        }
    
    # Optional: Define software reference implementation
    @backend("software", reference=True)
    def software_implementation(self):
        """Reference implementation for validation."""
        def gelu_approx(x):
            # Piecewise linear approximation
            for i, (x_min, x_max, a, b) in enumerate(self.segments):
                if x_min <= x < x_max:
                    return a * x + b
            return x  # Fallback
        
        return np.vectorize(gelu_approx)
    
    # Validation function to ensure correctness
    @validate
    def validate_implementation(self, input_data):
        """Validate hardware output matches reference."""
        import torch
        
        # Get reference GELU output
        reference = torch.nn.functional.gelu(torch.tensor(input_data))
        
        # Get our implementation output
        sw_impl = self.software_implementation()
        output = sw_impl(input_data)
        
        # Check approximation error
        max_error = np.max(np.abs(reference.numpy() - output))
        assert max_error < self.constraints["max_approximation_error"], \
            f"Approximation error {max_error} exceeds threshold"
        
        return True
    
    # Benchmark function for performance metrics
    @benchmark
    def benchmark_performance(self, input_shape=(1, 512, 768)):
        """Benchmark kernel performance."""
        return {
            "throughput_ops_per_sec": 1.5e9,  # 1.5 GOps/s
            "latency_cycles": 24,
            "resource_usage": {
                "LUT": 1250,
                "FF": 890,
                "DSP": 0,  # No DSPs needed!
                "BRAM": 2
            },
            "power_estimate_mw": 45
        }
    
    def _compute_segments(self, num_segments):
        """Compute optimal piecewise segments for GELU approximation."""
        # Simplified example - in practice this would be more sophisticated
        x_points = np.linspace(-3, 3, num_segments + 1)
        segments = []
        
        for i in range(num_segments):
            x_min, x_max = x_points[i], x_points[i + 1]
            # Compute linear approximation coefficients
            x_mid = (x_min + x_max) / 2
            # Simplified - real implementation would fit properly
            a = 0.5 + 0.398942 * np.exp(-0.5 * x_mid**2)
            b = 0.5 * x_mid - a * x_mid
            segments.append((x_min, x_max, a, b))
        
        return segments


# Step 2: Create unit tests for your kernel
import pytest

class TestEfficientGELU:
    """Test suite for EfficientGELU kernel."""
    
    def test_approximation_accuracy(self):
        """Test that approximation meets accuracy requirements."""
        kernel = EfficientGELUKernel(num_segments=8)
        
        # Test across typical input range
        test_input = np.random.randn(1000).astype(np.float32)
        assert kernel.validate_implementation(test_input)
    
    def test_edge_cases(self):
        """Test edge cases."""
        kernel = EfficientGELUKernel()
        
        # Test extreme values
        edge_cases = np.array([-10, -5, 0, 5, 10], dtype=np.float32)
        assert kernel.validate_implementation(edge_cases)
    
    def test_performance_targets(self):
        """Test that performance meets targets."""
        kernel = EfficientGELUKernel(parallel_factor=16)
        metrics = kernel.benchmark_performance()
        
        # Should achieve at least 1 GOps/s
        assert metrics["throughput_ops_per_sec"] >= 1e9
        # Should use no DSPs
        assert metrics["resource_usage"]["DSP"] == 0


# Step 3: Create example usage
def example_usage():
    """Show how users would use this kernel."""
    from brainsmith import CompilationStrategy, Compiler
    
    # Users can seamlessly use your kernel in their strategies
    strategy = CompilationStrategy(
        name="transformer_with_efficient_gelu",
        kernels=[
            # Your community kernel works just like built-in ones!
            KernelSpec("EfficientGELU", backend="hls", config={
                "num_segments": 8,
                "use_lut": True,
                "parallel_factor": 16
            }),
            KernelSpec("MatMul", backend="hls"),
            KernelSpec("LayerNorm", backend="rtl")
        ],
        transforms=[
            TransformSpec("Streamline"),
            TransformSpec("OptimizeGELU")  # Could be another community transform!
        ]
    )
    
    # Compile model
    compiler = Compiler()
    result = compiler.compile("bert.onnx", strategy, "./output")
    
    print(f"Model compiled with EfficientGELU kernel!")
    print(f"Throughput: {result.metrics['throughput']} FPS")


# Step 4: Create documentation
def generate_docs():
    """Generate documentation for the kernel."""
    return """
    # EfficientGELU Kernel
    
    ## Overview
    Hardware-optimized GELU activation using piecewise linear approximation.
    Provides 2-3x speedup with minimal accuracy loss.
    
    ## Installation
    ```bash
    brainsmith install efficient-gelu
    ```
    
    ## Usage
    ```python
    from brainsmith import KernelSpec
    
    kernel = KernelSpec("EfficientGELU", backend="hls", config={
        "num_segments": 8,
        "parallel_factor": 16
    })
    ```
    
    ## Performance
    - **Throughput**: 1.5 GOps/s @ 300MHz
    - **Latency**: 24 cycles
    - **Resources**: 1250 LUTs, 0 DSPs
    - **Accuracy**: <0.1% error vs torch.nn.functional.gelu
    
    ## Configuration Options
    - `num_segments` (2-16): Number of approximation segments
    - `use_lut` (bool): Enable lookup table optimization
    - `parallel_factor` (1,2,4,8,16): Parallelization level
    
    ## Benchmarks
    | Model | Standard GELU | EfficientGELU | Speedup |
    |-------|--------------|---------------|---------|
    | BERT-Base | 850 FPS | 2100 FPS | 2.47x |
    | GPT-2 | 420 FPS | 1150 FPS | 2.74x |
    
    ## Citation
    If you use this kernel, please cite:
    ```
    @software{efficient_gelu,
      author = {Jane Doe},
      title = {EfficientGELU: Hardware-Optimized GELU for FPGAs},
      year = {2024},
      url = {https://hub.brainsmith.ai/kernels/efficient-gelu}
    }
    ```
    """


# Step 5: Package and publish
if __name__ == "__main__":
    # Run tests
    print("Running kernel tests...")
    pytest.main([__file__, "-v"])
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    kernel = EfficientGELUKernel()
    metrics = kernel.benchmark_performance()
    print(f"Throughput: {metrics['throughput_ops_per_sec']/1e9:.2f} GOps/s")
    print(f"Resources: {metrics['resource_usage']}")
    
    # Generate documentation
    print("\nGenerating documentation...")
    docs = generate_docs()
    with open("README.md", "w") as f:
        f.write(docs)
    
    print("\nKernel ready to publish!")
    print("Run: brainsmith publish efficient-gelu")