# Demo 6: "BERT Accelerator Showcase" - Real Model Optimization

## Overview
Demonstrate real-world BERT model acceleration from PyTorch to FPGA, showcasing actual performance improvements and resource utilization.

## Demo Duration
4-5 minutes

## Key Message
"Real AI acceleration - 10x faster inference, 5x better power efficiency than GPU"

## Implementation Plan

### 1. Dashboard Layout

```
┌────────────────────────────────────────────────────────────────────┐
│                    BERT Model Acceleration Dashboard                │
├────────────────────┬───────────────────────┬──────────────────────┤
│   Model Overview   │   Layer Analysis      │  Performance Metrics │
│                    │                       │                      │
│  BERT-Base (12L)   │ [Layer Heatmap]      │  CPU:    1,200 ms   │
│  110M Parameters   │ [Resource Allocation] │  GPU:      120 ms   │
│  Sequence: 384     │ [Folding Decisions]   │  FPGA:      12 ms   │
│                    │                       │  ⚡ 100x faster!     │
├────────────────────┴───────────────────────┴──────────────────────┤
│                        Optimization Timeline                        │
│  [█████░░░░░] Quantization  [██████████░] DSE  [████████████] RTL │
└────────────────────────────────────────────────────────────────────┘
```

### 2. Key Visualizations

#### A. Model Architecture View
```javascript
const BERTArchitectureView = ({ model }) => {
    const layers = [
        { name: "Embedding", params: "23.8M", type: "embedding" },
        { name: "Encoder 1", params: "7.1M", type: "encoder" },
        { name: "Encoder 2", params: "7.1M", type: "encoder" },
        // ... 12 encoder layers
        { name: "Pooler", params: "590K", type: "pooler" }
    ];
    
    return (
        <div className="architecture-view">
            <SVGDiagram>
                {layers.map((layer, idx) => (
                    <Layer
                        key={idx}
                        {...layer}
                        optimized={layer.optimizationLevel}
                        onClick={() => showLayerDetails(layer)}
                    />
                ))}
                <DataFlow animated={true} />
            </SVGDiagram>
        </div>
    );
};
```

#### B. Layer-wise Resource Allocation
```javascript
const ResourceAllocationHeatmap = ({ allocation }) => {
    // Heatmap showing how FPGA resources are distributed
    return (
        <Heatmap
            data={allocation}
            xLabels={["LUT", "BRAM", "DSP", "URAM"]}
            yLabels={["Embed", "Enc1", "Enc2", ..., "Pool"]}
            colorScale={["#ffffff", "#0891b2", "#0c4a6e"]}
            tooltip={(x, y, value) => 
                `${yLabels[y]} uses ${value}% of ${xLabels[x]}`
            }
        />
    );
};
```

#### C. Performance Comparison Chart
```javascript
const PerformanceComparison = ({ results }) => {
    const data = {
        categories: ['Latency', 'Throughput', 'Power Efficiency'],
        series: [
            {
                name: 'CPU (Intel Xeon)',
                data: [1200, 0.83, 0.5],
                color: '#94a3b8'
            },
            {
                name: 'GPU (V100)',
                data: [120, 8.3, 2.5],
                color: '#0ea5e9'
            },
            {
                name: 'FPGA (ZCU104)',
                data: [12, 83.3, 12.5],
                color: '#10b981'
            }
        ]
    };
    
    return (
        <RadarChart
            data={data}
            showValues={true}
            animation={true}
            normalize={true}
        />
    );
};
```

### 3. Real-Time Optimization Process

#### A. Quantization Impact Visualization
```javascript
const QuantizationImpact = ({ beforeModel, afterModel }) => {
    const [showComparison, setShowComparison] = useState(false);
    
    return (
        <div className="quantization-section">
            <SplitView>
                <div className="before">
                    <h4>Original (FP32)</h4>
                    <ModelStats>
                        <Stat label="Size" value="440 MB" />
                        <Stat label="Ops/Inference" value="22.5 GFLOPs" />
                        <Stat label="Accuracy" value="89.2%" />
                    </ModelStats>
                    <WeightDistribution
                        weights={beforeModel.weights}
                        precision="FP32"
                    />
                </div>
                
                <div className="after">
                    <h4>Quantized (INT8)</h4>
                    <ModelStats>
                        <Stat label="Size" value="110 MB" trend="↓75%" />
                        <Stat label="Ops/Inference" value="5.6 GOPs" trend="↓75%" />
                        <Stat label="Accuracy" value="88.7%" trend="↓0.5%" />
                    </ModelStats>
                    <WeightDistribution
                        weights={afterModel.weights}
                        precision="INT8"
                    />
                </div>
            </SplitView>
            
            <AccuracyPreservation>
                <Chart
                    type="line"
                    data={accuracyOverQuantization}
                    highlight={selectedQuantization}
                />
            </AccuracyPreservation>
        </div>
    );
};
```

#### B. DSE Progress with Live Results
```javascript
const DSEProgressView = ({ exploration }) => {
    const [currentConfig, setCurrentConfig] = useState(null);
    const [bestConfig, setBestConfig] = useState(null);
    
    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/dse-progress');
        
        ws.onmessage = (event) => {
            const update = JSON.parse(event.data);
            setCurrentConfig(update.current);
            if (update.newBest) {
                setBestConfig(update.best);
            }
        };
        
        return () => ws.close();
    }, []);
    
    return (
        <div className="dse-progress">
            <ProgressBar
                current={currentConfig?.index}
                total={exploration.totalConfigs}
                eta={exploration.eta}
            />
            
            <CurrentConfigPanel>
                <h4>Evaluating: {currentConfig?.id}</h4>
                <ConfigDetails>
                    <div>Kernels: {currentConfig?.kernels.join(', ')}</div>
                    <div>Folding: SIMD={currentConfig?.simd}, PE={currentConfig?.pe}</div>
                </ConfigDetails>
            </CurrentConfigPanel>
            
            <BestConfigPanel highlight={bestConfig?.isNew}>
                <h4>Current Best: {bestConfig?.id}</h4>
                <Metrics>
                    <Metric label="Latency" value={`${bestConfig?.latency} ms`} />
                    <Metric label="Resources" value={`${bestConfig?.lut}% LUT`} />
                </Metrics>
            </BestConfigPanel>
        </div>
    );
};
```

### 4. Detailed Analysis Views

#### A. Layer Folding Decisions
```javascript
const FoldingAnalysis = ({ model, folding }) => {
    return (
        <div className="folding-analysis">
            <h3>Hardware Parallelization Strategy</h3>
            
            {model.layers.map((layer, idx) => (
                <LayerFoldingCard key={idx}>
                    <LayerName>{layer.name}</LayerName>
                    <FoldingParams>
                        <Param>
                            <Label>SIMD</Label>
                            <Value>{folding[idx].simd}</Value>
                            <Explanation>Input parallelism</Explanation>
                        </Param>
                        <Param>
                            <Label>PE</Label>
                            <Value>{folding[idx].pe}</Value>
                            <Explanation>Output parallelism</Explanation>
                        </Param>
                        <Param>
                            <Label>Cycles</Label>
                            <Value>{folding[idx].cycles}</Value>
                            <Explanation>Execution time</Explanation>
                        </Param>
                    </FoldingParams>
                    <ResourceImpact>
                        <MiniBar label="LUT" value={folding[idx].lutUsage} />
                        <MiniBar label="BRAM" value={folding[idx].bramUsage} />
                    </ResourceImpact>
                </LayerFoldingCard>
            ))}
        </div>
    );
};
```

#### B. Attention Mechanism Optimization
```javascript
const AttentionOptimization = ({ attentionLayers }) => {
    return (
        <div className="attention-optimization">
            <h3>Multi-Head Attention Acceleration</h3>
            
            <OptimizationTechniques>
                <Technique>
                    <Icon type="parallelization" />
                    <Title>Head Parallelization</Title>
                    <Description>12 attention heads computed in parallel</Description>
                    <Impact>12x speedup</Impact>
                </Technique>
                
                <Technique>
                    <Icon type="fusion" />
                    <Title>QKV Fusion</Title>
                    <Description>Fused query, key, value computation</Description>
                    <Impact>3x memory bandwidth reduction</Impact>
                </Technique>
                
                <Technique>
                    <Icon type="approximation" />
                    <Title>Softmax Approximation</Title>
                    <Description>Hardware-friendly exponential approximation</Description>
                    <Impact>0.1% accuracy loss, 5x faster</Impact>
                </Technique>
            </OptimizationTechniques>
            
            <AttentionHeatmap
                data={attentionLayers}
                metric="computeTime"
            />
        </div>
    );
};
```

### 5. Real-World Benchmarks

#### A. Latency Breakdown
```javascript
const LatencyBreakdown = ({ profiling }) => {
    const data = profiling.layers.map(layer => ({
        name: layer.name,
        embedding: layer.embedding_time,
        attention: layer.attention_time,
        ffn: layer.ffn_time,
        layernorm: layer.layernorm_time,
        other: layer.other_time
    }));
    
    return (
        <StackedBarChart
            data={data}
            categories={['embedding', 'attention', 'ffn', 'layernorm', 'other']}
            colors={['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#6b7280']}
            horizontal={true}
            showTotal={true}
            unit="μs"
        />
    );
};
```

#### B. Power Efficiency Analysis
```javascript
const PowerEfficiency = ({ measurements }) => {
    return (
        <div className="power-analysis">
            <PowerComparison>
                <Device name="CPU" power={180} inferences={0.83} efficiency={0.0046} />
                <Device name="GPU" power={250} inferences={8.3} efficiency={0.033} />
                <Device name="FPGA" power={15} inferences={83.3} efficiency={5.55} />
            </PowerComparison>
            
            <EfficiencyChart
                title="Inferences per Watt"
                data={measurements}
                highlight="FPGA"
                annotation="1200x more efficient than CPU!"
            />
            
            <CostAnalysis>
                <h4>24/7 Operation Cost (Annual)</h4>
                <CostBar device="CPU" kwh={1577} cost="$189" />
                <CostBar device="GPU" kwh={2190} cost="$263" />
                <CostBar device="FPGA" kwh={131} cost="$16" highlight />
            </CostAnalysis>
        </div>
    );
};
```

### 6. Interactive Demo Features

#### A. Model Variant Selector
```javascript
const ModelSelector = ({ onSelect }) => {
    const variants = [
        { name: "BERT-Base", layers: 12, params: "110M" },
        { name: "BERT-Large", layers: 24, params: "340M" },
        { name: "DistilBERT", layers: 6, params: "66M" },
        { name: "TinyBERT", layers: 4, params: "14M" }
    ];
    
    return (
        <VariantSelector>
            {variants.map(v => (
                <VariantCard
                    key={v.name}
                    {...v}
                    onClick={() => onSelect(v)}
                    preview={<MiniArchitecture layers={v.layers} />}
                />
            ))}
        </VariantSelector>
    );
};
```

#### B. Target Platform Selector
```javascript
const PlatformSelector = ({ onSelect }) => {
    const platforms = [
        { name: "ZCU104", vendor: "Xilinx", luts: 274080, brams: 1824 },
        { name: "Alveo U250", vendor: "Xilinx", luts: 1728000, brams: 5376 },
        { name: "Arria 10", vendor: "Intel", luts: 427200, brams: 2713 }
    ];
    
    return platforms.map(p => (
        <PlatformOption
            {...p}
            onClick={() => onSelect(p)}
            feasibility={checkFeasibility(currentModel, p)}
        />
    ));
};
```

### 7. Live Inference Demo

```javascript
const LiveInferenceDemo = ({ fpgaEndpoint }) => {
    const [inputText, setInputText] = useState("");
    const [results, setResults] = useState(null);
    const [inferenceTime, setInferenceTime] = useState(null);
    
    const runInference = async () => {
        const start = performance.now();
        
        const response = await fetch(fpgaEndpoint, {
            method: 'POST',
            body: JSON.stringify({ text: inputText }),
            headers: { 'Content-Type': 'application/json' }
        });
        
        const result = await response.json();
        const end = performance.now();
        
        setResults(result);
        setInferenceTime(end - start);
    };
    
    return (
        <div className="live-demo">
            <TextInput
                placeholder="Enter text for BERT inference..."
                value={inputText}
                onChange={setInputText}
            />
            
            <RunButton onClick={runInference}>
                Run on FPGA
            </RunButton>
            
            {results && (
                <Results>
                    <InferenceTime>{inferenceTime.toFixed(2)} ms</InferenceTime>
                    <Predictions>{results.predictions}</Predictions>
                    <AttentionVisualization weights={results.attention} />
                </Results>
            )}
        </div>
    );
};
```

### 8. Deployment Readiness

```javascript
const DeploymentReadiness = ({ buildArtifacts }) => {
    return (
        <div className="deployment-section">
            <h3>Ready for Production</h3>
            
            <ArtifactsList>
                <Artifact
                    name="Bitstream"
                    file="bert_optimized.bit"
                    size="45 MB"
                    icon="chip"
                />
                <Artifact
                    name="Host Driver"
                    file="bert_driver.so"
                    size="2.3 MB"
                    icon="code"
                />
                <Artifact
                    name="Python API"
                    file="bert_fpga.py"
                    size="15 KB"
                    icon="python"
                />
                <Artifact
                    name="Performance Report"
                    file="optimization_report.pdf"
                    size="1.2 MB"
                    icon="document"
                />
            </ArtifactsList>
            
            <IntegrationCode>
                {`# Easy integration
from bert_fpga import BERTAccelerator

# Initialize FPGA
accelerator = BERTAccelerator("bert_optimized.bit")

# Run inference
result = accelerator.predict("Hello world!")
print(f"Latency: {result.latency_ms} ms")`}
            </IntegrationCode>
        </div>
    );
};
```

### 9. Demo Flow Script

1. **Introduction (30s)**
   - Show BERT model overview
   - Highlight challenge: 1.2s latency on CPU

2. **Quantization (1m)**
   - Show FP32 → INT8 conversion
   - Demonstrate minimal accuracy loss
   - 4x size reduction

3. **DSE Process (1.5m)**
   - Launch exploration
   - Show real-time progress
   - Highlight Pareto-optimal configs
   - Select best configuration

4. **Results Analysis (1.5m)**
   - Show 100x latency improvement
   - Display resource utilization
   - Compare power efficiency
   - Show cost savings

5. **Live Demo (30s)**
   - Run actual inference
   - Show real-time performance
   - Demonstrate accuracy

### 10. Key Metrics to Highlight

- **Latency**: 1200ms → 12ms (100x improvement)
- **Throughput**: 0.83 → 83.3 inferences/sec
- **Power**: 180W → 15W (12x reduction)
- **Cost**: $189/year → $16/year operation
- **Accuracy**: 89.2% → 88.7% (0.5% loss)
- **Development Time**: 3 days → 3 hours