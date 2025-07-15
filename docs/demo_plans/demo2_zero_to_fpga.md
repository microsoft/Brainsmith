# Demo 2: "Zero to FPGA" - End-to-End Pipeline Flow

## Overview
Showcase the complete journey from a PyTorch model to FPGA bitstream with real-time visualization of each transformation stage.

## Demo Duration
4-5 minutes

## Key Message
"From AI researcher's PyTorch model to optimized FPGA implementation - fully automated"

## Implementation Plan

### 1. Pipeline Visualization Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Progress Bar                     │
├─────────┬──────────┬──────────┬──────────┬────────────────┤
│ PyTorch │ Brevitas │   ONNX   │   DSE    │   FPGA Output  │
│   ✓     │    ✓     │    →     │          │                │
└─────────┴──────────┴──────────┴──────────┴────────────────┘

[Main Visualization Area - Changes per stage]

[Metrics Dashboard - Updates in real-time]
```

### 2. Stage-by-Stage Implementation

#### Stage 1: PyTorch Model Loading
```python
class PyTorchStage:
    def visualize(self):
        return {
            "type": "network_graph",
            "data": {
                "layers": [
                    {"name": "Embedding", "params": "30522×768"},
                    {"name": "LayerNorm", "params": "768"},
                    {"name": "MultiHeadAttention", "params": "768×768×4"},
                    {"name": "FFN", "params": "768×3072×2"}
                ],
                "connections": [...],
                "metrics": {
                    "parameters": "110M",
                    "size": "440MB (FP32)",
                    "ops": "22.5 GFLOPs"
                }
            }
        }
```

**Visualization**: Interactive neural network diagram with layer details on hover

#### Stage 2: Brevitas Quantization
```python
class BrevitasStage:
    def visualize(self, model):
        return {
            "type": "quantization_comparison",
            "data": {
                "before": {
                    "precision": "FP32",
                    "size": "440MB",
                    "example_weights": [0.0234, -0.1847, 0.0012, ...]
                },
                "after": {
                    "precision": "INT8",
                    "size": "110MB",
                    "example_weights": [3, -23, 0, ...],
                    "scale_factors": {...}
                },
                "accuracy_impact": {
                    "original": 0.942,
                    "quantized": 0.938,
                    "delta": -0.004
                }
            }
        }
```

**Visualization**: 
- Split view showing FP32 vs INT8 weight distributions
- Size reduction animation (440MB → 110MB)
- Accuracy preservation meter

#### Stage 3: ONNX Conversion
```python
class ONNXStage:
    def visualize(self, model):
        return {
            "type": "graph_transformation",
            "data": {
                "original_ops": ["torch.nn.Linear", "torch.nn.LayerNorm"],
                "onnx_ops": ["MatMul", "Add", "Reshape", "ReduceMean"],
                "graph_diff": {
                    "nodes_before": 45,
                    "nodes_after": 128,
                    "explanation": "PyTorch ops expanded to ONNX primitives"
                }
            }
        }
```

**Visualization**: 
- Side-by-side graph comparison
- Animated transformation showing op decomposition
- ONNX operator inventory

#### Stage 4: Design Space Exploration
```python
class DSEStage:
    def __init__(self):
        self.current_config = 0
        self.total_configs = 0
        self.results = []
        
    def visualize_realtime(self):
        return {
            "type": "exploration_dashboard",
            "data": {
                "progress": f"{self.current_config}/{self.total_configs}",
                "current_best": {
                    "config_id": "dse_abc_00234",
                    "throughput": "1,250 inf/s",
                    "latency": "0.8ms",
                    "resources": {"LUT": "45%", "BRAM": "62%"}
                },
                "pareto_frontier": self.calculate_pareto(),
                "eta": self.estimate_completion_time()
            }
        }
```

**Visualization**:
- Real-time scatter plot of configurations
- Current best configuration details
- Progress bar with ETA
- Success/failure rate gauge

#### Stage 5: FPGA Output Generation
```python
class FPGAStage:
    def visualize(self, best_config):
        return {
            "type": "fpga_results",
            "data": {
                "rtl_preview": self.get_rtl_snippet(),
                "resource_map": {
                    "LUT": {"used": 45000, "total": 100000},
                    "BRAM": {"used": 312, "total": 500},
                    "DSP": {"used": 180, "total": 200}
                },
                "timing": {
                    "clock_freq": "300MHz",
                    "critical_path": "2.8ns"
                },
                "power": {
                    "dynamic": "15W",
                    "static": "5W"
                }
            }
        }
```

**Visualization**:
- FPGA floorplan with resource utilization heatmap
- RTL code preview with syntax highlighting
- Performance comparison chart (CPU vs GPU vs FPGA)

### 3. Real-time Animation System

```javascript
class PipelineAnimator {
    constructor() {
        this.stages = ['pytorch', 'brevitas', 'onnx', 'dse', 'fpga'];
        this.currentStage = 0;
    }
    
    async animateTransition(fromStage, toStage) {
        // Particle flow animation between stages
        await this.particleFlow(fromStage, toStage);
        
        // Morph visualization
        await this.morphVisualization(
            this.getStageVisual(fromStage),
            this.getStageVisual(toStage)
        );
        
        // Update metrics
        this.updateMetrics(toStage);
    }
}
```

### 4. Interactive Elements

#### Model Selector
```javascript
const ModelSelector = () => {
    const models = [
        { name: "BERT-Base", layers: 12, params: "110M" },
        { name: "BERT-Large", layers: 24, params: "340M" },
        { name: "DistilBERT", layers: 6, params: "66M" },
        { name: "Custom Model", upload: true }
    ];
    
    return (
        <Select onChange={loadModel}>
            {models.map(m => <Option key={m.name} value={m}>{m.name}</Option>)}
        </Select>
    );
};
```

#### Stage Controls
- **Play/Pause**: Control automatic progression
- **Step Forward/Back**: Manual stage control
- **Speed Control**: Adjust animation speed
- **Details Toggle**: Show/hide technical details

### 5. Backend Architecture

```python
# WebSocket for real-time updates
class PipelineServer:
    def __init__(self):
        self.websocket_clients = []
        self.pipeline = BrainsmithPipeline()
    
    async def run_demo(self, model_path, blueprint_path):
        # Stage 1: Load PyTorch
        await self.broadcast_update("stage", "pytorch")
        model = torch.load(model_path)
        await self.broadcast_update("pytorch_complete", self.analyze_model(model))
        
        # Stage 2: Quantize
        await self.broadcast_update("stage", "brevitas")
        quant_model = self.quantize_model(model)
        await self.broadcast_update("brevitas_complete", self.compare_models(model, quant_model))
        
        # Stage 3: Export ONNX
        await self.broadcast_update("stage", "onnx")
        onnx_model = self.export_onnx(quant_model)
        await self.broadcast_update("onnx_complete", self.analyze_onnx(onnx_model))
        
        # Stage 4: DSE (with progress updates)
        await self.broadcast_update("stage", "dse")
        async for update in self.run_dse_async(onnx_model, blueprint_path):
            await self.broadcast_update("dse_progress", update)
        
        # Stage 5: Generate FPGA
        await self.broadcast_update("stage", "fpga")
        fpga_results = await self.generate_fpga(best_config)
        await self.broadcast_update("fpga_complete", fpga_results)
```

### 6. Metrics Dashboard

```javascript
const MetricsDashboard = ({ stage, metrics }) => {
    return (
        <div className="metrics-grid">
            <MetricCard 
                title="Model Size"
                value={metrics.size}
                trend={metrics.sizeTrend}
                icon="database"
            />
            <MetricCard
                title="Throughput"
                value={metrics.throughput}
                unit="inf/s"
                icon="lightning"
            />
            <MetricCard
                title="Latency"
                value={metrics.latency}
                unit="ms"
                icon="clock"
            />
            <MetricCard
                title="Power"
                value={metrics.power}
                unit="W"
                icon="battery"
            />
        </div>
    );
};
```

### 7. Visualization Technologies

- **Frontend**: React + D3.js + Three.js (for 3D network viz)
- **Backend**: FastAPI + WebSocket for real-time updates
- **Animation**: Framer Motion for smooth transitions
- **Graphs**: Plotly for interactive charts

### 8. Demo Data Preparation

```python
# Pre-computed demo data for smooth presentation
demo_data = {
    "bert_base": {
        "pytorch_analysis": {...},
        "quantization_results": {...},
        "onnx_graph": {...},
        "dse_progression": [
            {"time": 0, "configs": 0, "best": None},
            {"time": 30, "configs": 150, "best": {...}},
            # ... more snapshots
        ],
        "final_results": {...}
    }
}
```

### 9. Error Handling & Edge Cases

- **Model too large**: Show warning and suggest optimizations
- **DSE timeout**: Display partial results with explanation
- **Failed builds**: Show failure reasons and recovery

### 10. Presentation Mode Features

- **Guided tour**: Step-by-step explanation bubbles
- **Highlight mode**: Emphasize important changes
- **Comparison mode**: Side-by-side with baseline
- **Export results**: Generate report PDF/HTML