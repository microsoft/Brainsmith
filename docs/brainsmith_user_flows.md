# Brainsmith User Flows and Experience Design

## Overview

This document defines the user experience flows for Brainsmith, from basic single-model compilation to advanced design space exploration. The flows are designed to be progressive - users can start simple and gradually move to more sophisticated usage as their needs evolve.

## User Personas and Use Cases

### **Persona 1: FPGA Developer (Basic User)**
- **Goal**: Quickly convert a trained ONNX model to FPGA accelerator
- **Experience Level**: Familiar with FPGA development, new to Brainsmith
- **Priority**: Simple, fast workflow with good defaults

### **Persona 2: Performance Engineer (Intermediate User)**
- **Goal**: Optimize a specific model for performance/resource targets
- **Experience Level**: Understands FPGA optimization, familiar with FINN
- **Priority**: Control over key parameters, visibility into trade-offs

### **Persona 3: Researcher (Advanced User)**
- **Goal**: Systematic design space exploration and algorithm research
- **Experience Level**: Deep optimization knowledge, develops custom DSE approaches
- **Priority**: Extensibility, data collection, integration with research tools

## User Flow 1: Basic Model Compilation

### **Scenario**: "I have a quantized BERT model and need an FPGA accelerator quickly"

#### **CLI Flow (Recommended for Beginners)**
```bash
# Step 1: Check available blueprints
$ brainsmith blueprints list
Available blueprints:
  bert          - BERT transformer models (default config)
  resnet        - ResNet CNN models
  mobilenet     - MobileNet efficient CNNs

# Step 2: Basic compilation with defaults
$ brainsmith build my_bert_model.onnx --blueprint bert --output ./my_build
Brainsmith Platform v1.0.0

✓ Loading model: my_bert_model.onnx (2.3 MB)
✓ Loading blueprint: bert (19 build steps)
✓ Validating configuration

Build Progress:
[████████████████████████████████████████] 100% | Step 19/19: shell_metadata_handover

✓ Build completed successfully in 2m 15s

Results:
  Output directory: ./my_build/
  Final model: ./my_build/output.onnx
  Stitched IP: ./my_build/stitched_ip/
  Reports: ./my_build/reports/
  
Performance Summary:
  Estimated throughput: 1,847 inferences/sec
  LUT utilization: 62% (247,680 / 400,000)
  DSP utilization: 45% (900 / 2,000)
  Power estimate: 23.4W

Next steps:
  • Review reports in ./my_build/reports/
  • Check IP integration guide in ./my_build/stitched_ip/README.md
  • For optimization, try: brainsmith optimize --help
```

#### **Python Library Flow**
```python
import brainsmith
import onnx

# Load model
model = onnx.load("my_bert_model.onnx")

# Simple compilation
result = brainsmith.build_model(
    model=model,
    blueprint="bert",
    output_dir="./my_build"
)

# Check results
if result.success:
    print(f"Build completed in {result.build_time:.1f}s")
    print(f"Throughput: {result.metrics.performance.throughput_ops_sec:.0f} ops/sec")
    print(f"LUT usage: {result.metrics.resources.lut_utilization_percent:.1f}%")
    print(f"Output: {result.final_model_path}")
else:
    print("Build failed:")
    for error in result.errors:
        print(f"  - {error}")
```

#### **Success Criteria**
- Build completes without user configuration
- Clear progress indication and results summary
- Actionable next steps provided
- All artifacts ready for integration

---

## User Flow 2: Parameter Optimization

### **Scenario**: "I need higher throughput and am willing to use more resources"

#### **CLI Flow (Parameter Exploration)**
```bash
# Step 1: Check current performance
$ brainsmith build my_bert_model.onnx --blueprint bert --output ./baseline \
  --target-fps 3000 --board V80

# Results: 1,847 ops/sec, 62% LUT usage

# Step 2: Explore higher performance configurations
$ brainsmith optimize my_bert_model.onnx --blueprint bert --output ./optimized \
  --target-fps 5000,7500,10000 \
  --clock-freq 250,300 \
  --optimize-for throughput

Brainsmith Optimization

Exploring 6 configurations:
  Config 1/6: fps=5000, clk=250MHz  → 4,231 ops/sec, 78% LUT ✓
  Config 2/6: fps=7500, clk=250MHz  → 6,445 ops/sec, 89% LUT ✓
  Config 3/6: fps=10000, clk=250MHz → Failed (resource limit)
  Config 4/6: fps=5000, clk=300MHz  → 5,077 ops/sec, 78% LUT ✓
  Config 5/6: fps=7500, clk=300MHz  → 7,734 ops/sec, 89% LUT ✓  ⭐ Best
  Config 6/6: fps=10000, clk=300MHz → Failed (resource limit)

Best configuration:
  Target FPS: 7500, Clock: 300MHz
  Achieved: 7,734 ops/sec (103% of target)
  Resources: 89% LUT, 67% DSP
  Power: 31.2W

Apply best configuration? [y/N]: y

✓ Building optimized configuration...
✓ Optimized build completed in 2m 48s

Compare results:
                  Baseline    Optimized    Improvement
  Throughput      1,847       7,734        +319%
  LUT usage       62%         89%          +27%
  Power           23.4W       31.2W        +33%
```

#### **Interactive CLI Flow**
```bash
# Step 1: Start interactive optimization
$ brainsmith interactive my_bert_model.onnx --blueprint bert

Brainsmith Interactive Mode

Model: my_bert_model.onnx (BERT-base, 12 layers)
Blueprint: bert (19 optimization steps)

Current configuration:
  Target FPS: 3000
  Clock frequency: 250 MHz
  Board: V80
  
What would you like to optimize for?
1) Higher throughput
2) Lower resource usage  
3) Better power efficiency
4) Custom configuration

Choice [1-4]: 1

Higher throughput optimization:
  We can increase target FPS and clock frequency.
  This will use more FPGA resources but increase performance.

  Current estimate: ~1,800 ops/sec
  
  Target throughput [ops/sec]: 6000
  
Exploring configurations for 6000 ops/sec target...

Found 3 viable configurations:
1) fps=6000, clk=250MHz  → 6,127 ops/sec, 85% LUT, 28.9W
2) fps=6000, clk=275MHz  → 6,742 ops/sec, 85% LUT, 30.1W  ⭐ Recommended
3) fps=6000, clk=300MHz  → 7,356 ops/sec, 85% LUT, 31.2W

Select configuration [1-3]: 2

✓ Building configuration 2...

Would you like to:
1) Apply this configuration
2) Try different target
3) Export configuration file
4) Exit

Choice [1-4]: 3

Configuration saved to: bert_optimized_config.yaml
Use with: brainsmith build model.onnx --config bert_optimized_config.yaml
```

#### **Python Library Flow (Parameter Sweep)**
```python
import brainsmith

# Parameter sweep
configs = [
    {"target_fps": 3000, "clk_period_ns": 4.0},
    {"target_fps": 5000, "clk_period_ns": 4.0},
    {"target_fps": 7500, "clk_period_ns": 3.33},
    {"target_fps": 10000, "clk_period_ns": 3.33},
]

results = []
for i, config in enumerate(configs):
    print(f"Testing configuration {i+1}/{len(configs)}")
    
    result = brainsmith.build_model(
        model=model,
        blueprint="bert",
        output_dir=f"./sweep_{i+1}",
        **config
    )
    
    if result.success:
        results.append({
            'config': config,
            'throughput': result.metrics.performance.throughput_ops_sec,
            'lut_utilization': result.metrics.resources.lut_utilization_percent,
            'power': result.metrics.resources.estimated_power_w,
            'build_time': result.build_time
        })

# Find best configuration
best = max(results, key=lambda x: x['throughput'])
print(f"Best config: {best['config']}")
print(f"Throughput: {best['throughput']:.0f} ops/sec")
print(f"LUT usage: {best['lut_utilization']:.1f}%")
```

---

## User Flow 3: Design Space Exploration

### **Scenario**: "I need to systematically explore trade-offs for research/product decisions"

#### **CLI Flow (Systematic Exploration)**
```bash
# Step 1: Analyze design space
$ brainsmith analyze-space --blueprint bert --model my_bert_model.onnx

Design Space Analysis for 'bert' blueprint:

Dimensions:
  platforms: 3 options (V80, ZCU104, U250)
  target_fps: continuous [1000, 10000]  
  clock_freq: continuous [150, 300] MHz
  streamlining: 3 options (conservative, balanced, aggressive)
  
Estimated design space size: ~1,000,000 configurations
Recommended sample size: 50-200 evaluations

Constraints:
  Hard: max_lut_utilization <= 0.9
  Hard: max_dsp_utilization <= 0.9
  Soft: preferred_power <= 50W

# Step 2: Generate exploration plan
$ brainsmith plan-exploration my_bert_model.onnx --blueprint bert \
  --sample-size 100 \
  --objectives throughput,efficiency,resources \
  --export-plan exploration_plan.yaml

Generated exploration plan:
  Sample strategy: Latin hypercube sampling
  Sample size: 100 configurations
  Estimated time: 3.5 hours
  
Objectives:
  - Maximize: throughput (ops/sec)
  - Maximize: efficiency (ops/joule)  
  - Minimize: resource_usage (LUT+DSP)

Plan saved to: exploration_plan.yaml

# Step 3: Execute exploration
$ brainsmith explore --plan exploration_plan.yaml --output ./exploration \
  --parallel 4

Brainsmith Design Space Exploration

Configuration: exploration_plan.yaml
Parallel builds: 4
Progress tracking: ./exploration/progress.json

Progress: [██████████████████████████████      ] 75/100 | ETA: 42m

Current best configurations:
  Best throughput: 8,234 ops/sec (config #47)
  Best efficiency: 234 ops/joule (config #23)  
  Best resource:   45% utilization (config #12)

# Step 4: Analyze results
$ brainsmith analyze-results ./exploration/results.json \
  --pareto-plot throughput,power \
  --sensitivity-analysis \
  --export-report exploration_report.pdf

Analysis Summary:

Pareto frontier: 12 non-dominated configurations found
Key insights:
  • Clock frequency most impactful for throughput (+73% sensitivity)
  • Streamlining strategy affects resource usage (+45% sensitivity)
  • Platform choice dominates power consumption (+89% sensitivity)

Recommendations:
  1. For highest throughput: V80, 300MHz, aggressive streamlining
  2. For best efficiency: ZCU104, 225MHz, balanced streamlining
  3. For lowest resource: U250, 175MHz, conservative streamlining

Detailed analysis saved to: exploration_report.pdf
Interactive plots saved to: ./exploration/plots/
```

#### **Python Library Flow (Research Integration)**
```python
import brainsmith
import numpy as np
import pandas as pd

# Define design space programmatically
design_space = brainsmith.DesignSpace.from_blueprint("bert")

# Custom parameter ranges
design_space.set_parameter_range("target_fps", [2000, 8000])
design_space.set_parameter_range("clk_period_ns", [3.0, 6.0])
design_space.add_categorical("board", ["V80", "ZCU104"])

# Generate sample points (research-grade sampling)
sample_points = brainsmith.sample_design_space(
    design_space,
    n_samples=50,
    strategy="latin_hypercube",
    seed=42  # reproducible
)

# Execute exploration
exploration = brainsmith.DesignSpaceExploration(
    model=model,
    design_space=design_space,
    output_dir="./research_exploration"
)

results = []
for i, point in enumerate(sample_points):
    print(f"Evaluating point {i+1}/{len(sample_points)}")
    
    result = exploration.evaluate_point(point)
    results.append(result)
    
    # Export intermediate results for monitoring
    if i % 10 == 0:
        brainsmith.export_results(results, f"intermediate_{i}.json")

# Comprehensive analysis
analysis = brainsmith.analyze_exploration_results(results)

# Extract research data
df = pd.DataFrame([r.to_research_dict() for r in results])
df.to_csv("dse_dataset.csv")

# Pareto frontier analysis
pareto = analysis.compute_pareto_frontier(
    objectives=["throughput", "efficiency", "resource_usage"],
    directions=["maximize", "maximize", "minimize"]
)

# Sensitivity analysis
sensitivity = analysis.compute_sensitivity_analysis()
print("Most influential parameters:")
for param, influence in sensitivity.items():
    print(f"  {param}: {influence:.2f}")

# Export for external research tools
research_export = {
    'design_space': design_space.to_dict(),
    'sample_points': [p.to_dict() for p in sample_points],
    'results': [r.to_research_dict() for r in results],
    'pareto_frontier': pareto.to_dict(),
    'sensitivity_analysis': sensitivity
}

brainsmith.export_research_data(research_export, "research_dataset.json")
```

#### **External Tool Integration Flow**
```python
# Integration with external optimization framework
import brainsmith
from my_optimization_framework import BayesianOptimizer

# Set up Brainsmith platform
platform = brainsmith.Platform()
design_space = platform.load_design_space("bert")

# Configure external optimizer  
optimizer = BayesianOptimizer(
    design_space=design_space.export_for_external_tool(),
    objectives=["throughput", "power"],
    constraints=design_space.get_constraints()
)

# Optimization loop
for iteration in range(100):
    # External tool suggests next point
    next_point = optimizer.suggest_next_point()
    
    # Evaluate through Brainsmith
    result = platform.evaluate_design_point(
        model=model,
        design_point=next_point,
        output_dir=f"./optimization/iter_{iteration}"
    )
    
    # Feed back to optimizer
    optimizer.update_with_result(
        point=next_point,
        objectives={
            'throughput': result.metrics.performance.throughput_ops_sec,
            'power': result.metrics.resources.estimated_power_w
        }
    )
    
    # Check convergence
    if optimizer.has_converged():
        break

# Get final recommendations
best_config = optimizer.get_best_configuration()
pareto_frontier = optimizer.get_pareto_frontier()

print(f"Optimization converged after {iteration+1} iterations")
print(f"Best configuration: {best_config}")
```

---

## User Experience Principles

### **Progressive Complexity**
1. **Start Simple**: Single command with sensible defaults
2. **Add Control**: Parameter tuning with guided optimization
3. **Enable Research**: Full programmatic control and extensibility

### **Clear Feedback**
1. **Progress Indication**: Real-time build progress and time estimates
2. **Result Summary**: Key metrics prominently displayed
3. **Next Steps**: Actionable recommendations for further optimization

### **Data-Driven Decisions**
1. **Metric Visibility**: All relevant performance/resource data exposed
2. **Comparison Tools**: Easy comparison between configurations
3. **Insight Generation**: Automated analysis and recommendations

### **Extensibility**
1. **Plugin Points**: Custom metrics, analysis tools, optimization strategies
2. **Export Formats**: Standard formats for external tool integration
3. **Research APIs**: Clean interfaces for algorithm development

## Success Metrics by User Flow

### **Basic Compilation (Flow 1)**
- ✅ Single command success rate > 95%
- ✅ Average build time < 5 minutes for typical models
- ✅ Clear error messages with actionable guidance
- ✅ Complete documentation and integration guides

### **Parameter Optimization (Flow 2)**
- ✅ Interactive mode guides users to better configurations
- ✅ Optimization suggestions improve target metrics by >20%
- ✅ Parameter sweep completes in reasonable time (< 1 hour for 5 configs)
- ✅ Clear trade-off visualization and recommendations

### **Design Space Exploration (Flow 3)**
- ✅ Generate research-grade datasets with comprehensive metrics
- ✅ Support 100+ evaluation runs with progress tracking
- ✅ Pareto frontier analysis with statistical significance
- ✅ Seamless integration with external research tools

These user flows ensure that Brainsmith serves users across the spectrum from quick prototyping to advanced research, with clear progression paths and appropriate levels of control at each stage.