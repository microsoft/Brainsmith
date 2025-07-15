# Demo 3: "Pareto Explorer" - Interactive Multi-Objective Optimization

## Overview
Interactive 3D visualization demonstrating how Brainsmith finds optimal trade-offs between performance, resources, and power consumption.

## Demo Duration
3-4 minutes

## Key Message
"No more guessing - automatically find the best configuration for YOUR constraints"

## Implementation Plan

### 1. Main Visualization Components

#### A. 3D Scatter Plot (Center)
```javascript
const Pareto3DPlot = () => {
    // Three.js or Plotly 3D scatter
    // X-axis: Throughput (fps)
    // Y-axis: Resource Utilization (%)
    // Z-axis: Power Consumption (W)
    // Color: Pareto-optimal (gold) vs dominated (gray)
    
    return (
        <Plot3D
            data={configurations}
            axes={{
                x: { label: "Throughput (inf/s)", range: [0, 2000] },
                y: { label: "LUT Utilization (%)", range: [0, 100] },
                z: { label: "Power (W)", range: [0, 50] }
            }}
            highlighting={{
                pareto: { color: "gold", size: 8 },
                dominated: { color: "gray", size: 4 },
                selected: { color: "red", size: 12 }
            }}
        />
    );
};
```

#### B. Constraint Sliders (Left Panel)
```javascript
const ConstraintPanel = ({ onChange }) => {
    const [constraints, setConstraints] = useState({
        maxLUT: 85,
        minThroughput: 500,
        maxPower: 30
    });
    
    return (
        <Panel title="Design Constraints">
            <Slider
                label="Max LUT Utilization"
                value={constraints.maxLUT}
                min={0}
                max={100}
                unit="%"
                onChange={(val) => updateConstraint('maxLUT', val)}
            />
            <Slider
                label="Min Throughput"
                value={constraints.minThroughput}
                min={0}
                max={2000}
                unit="inf/s"
                onChange={(val) => updateConstraint('minThroughput', val)}
            />
            <Slider
                label="Max Power"
                value={constraints.maxPower}
                min={0}
                max={50}
                unit="W"
                onChange={(val) => updateConstraint('maxPower', val)}
            />
        </Panel>
    );
};
```

#### C. Optimization Weights (Right Panel)
```javascript
const WeightPanel = ({ onChange }) => {
    const [weights, setWeights] = useState({
        performance: 0.4,
        resources: 0.3,
        power: 0.3
    });
    
    return (
        <Panel title="Optimization Priorities">
            <PieChart
                data={[
                    { name: "Performance", value: weights.performance },
                    { name: "Resources", value: weights.resources },
                    { name: "Power", value: weights.power }
                ]}
                interactive={true}
                onChange={updateWeights}
            />
            <WeightAdjuster weights={weights} onChange={setWeights} />
        </Panel>
    );
};
```

#### D. Configuration Details (Bottom)
```javascript
const ConfigDetails = ({ selectedConfig }) => {
    if (!selectedConfig) return null;
    
    return (
        <DetailPanel>
            <h3>Configuration: {selectedConfig.id}</h3>
            <MetricsGrid>
                <Metric label="Throughput" value={selectedConfig.throughput} unit="inf/s" />
                <Metric label="Latency" value={selectedConfig.latency} unit="ms" />
                <Metric label="LUT" value={selectedConfig.lut} unit="%" />
                <Metric label="BRAM" value={selectedConfig.bram} unit="%" />
                <Metric label="Power" value={selectedConfig.power} unit="W" />
            </MetricsGrid>
            <ConfigBreakdown>
                <div>Kernels: {selectedConfig.kernels.join(", ")}</div>
                <div>Transforms: {selectedConfig.transforms.join(" → ")}</div>
                <div>Backend: {selectedConfig.backend}</div>
            </ConfigBreakdown>
        </DetailPanel>
    );
};
```

### 2. Interactive Features

#### Real-time Filtering
```javascript
class ParetoExplorer {
    filterConfigurations(configs, constraints) {
        return configs.map(config => ({
            ...config,
            visible: this.meetsConstraints(config, constraints),
            feasible: this.isFeasible(config, constraints)
        }));
    }
    
    updateParetoFrontier(configs, weights) {
        // Recalculate Pareto frontier with new weights
        const scored = configs.map(c => ({
            ...c,
            score: this.calculateWeightedScore(c, weights)
        }));
        
        return this.findParetoOptimal(scored);
    }
}
```

#### Animated Transitions
```javascript
const animateConstraintChange = (oldConstraints, newConstraints) => {
    // Fade out violating configurations
    // Highlight newly feasible region
    // Animate Pareto frontier update
    
    const transition = d3.transition()
        .duration(750)
        .ease(d3.easeCubicInOut);
    
    // Update point visibility
    points.transition(transition)
        .style("opacity", d => meetsConstraints(d, newConstraints) ? 1 : 0.1)
        .attr("r", d => isParetoOptimal(d) ? 8 : 4);
};
```

### 3. Visualization Modes

#### Mode 1: Constraint Exploration
- User adjusts constraint sliders
- Invalid configurations fade out
- Feasible region highlights
- Pareto frontier updates in real-time

#### Mode 2: Weight Adjustment
- User drags pie chart segments
- Optimal point moves smoothly
- Trade-off curves highlight
- "What-if" analysis enabled

#### Mode 3: Comparative Analysis
- Select multiple configurations
- Spider/radar chart comparison
- Parallel coordinates view
- Head-to-head metrics

### 4. Backend Integration

```python
from brainsmith.core.phase2 import ParetoRanker

class ParetoAPI:
    def __init__(self):
        self.ranker = ParetoRanker()
        self.cache = {}
    
    @app.post("/api/pareto/filter")
    async def filter_configurations(self, constraints: dict):
        # Apply constraints
        filtered = self.apply_constraints(self.all_configs, constraints)
        
        # Find Pareto frontier
        pareto = self.ranker.find_pareto_optimal(filtered)
        
        return {
            "total": len(self.all_configs),
            "feasible": len(filtered),
            "pareto_optimal": len(pareto),
            "configurations": self.serialize_configs(filtered),
            "frontier": self.serialize_configs(pareto)
        }
    
    @app.post("/api/pareto/optimize")
    async def weighted_optimization(self, weights: dict):
        # Calculate weighted scores
        scored = self.ranker.score_configurations(self.all_configs, weights)
        
        # Find best configuration
        best = max(scored, key=lambda x: x.score)
        
        return {
            "best_config": best,
            "top_10": scored[:10],
            "score_distribution": self.get_score_histogram(scored)
        }
```

### 5. Advanced Visualizations

#### Pareto Surface (3D Mesh)
```javascript
// Generate Pareto surface mesh
const generateParetoSurface = (points) => {
    // Delaunay triangulation of Pareto points
    const delaunay = d3.Delaunay.from(
        points,
        d => d.throughput,
        d => d.resources,
        d => d.power
    );
    
    // Create mesh geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setFromPoints(delaunay.triangles);
    
    // Semi-transparent golden surface
    const material = new THREE.MeshPhongMaterial({
        color: 0xffd700,
        opacity: 0.3,
        transparent: true
    });
    
    return new THREE.Mesh(geometry, material);
};
```

#### Trade-off Curves (2D Projections)
```javascript
const TradeoffProjections = ({ data, selectedAxes }) => {
    return (
        <div className="projections-grid">
            <Plot2D
                title="Performance vs Resources"
                x={data.map(d => d.throughput)}
                y={data.map(d => d.lut)}
                pareto={data.filter(d => d.isParetoOptimal)}
            />
            <Plot2D
                title="Performance vs Power"
                x={data.map(d => d.throughput)}
                y={data.map(d => d.power)}
                pareto={data.filter(d => d.isParetoOptimal)}
            />
            <Plot2D
                title="Resources vs Power"
                x={data.map(d => d.lut)}
                y={data.map(d => d.power)}
                pareto={data.filter(d => d.isParetoOptimal)}
            />
        </div>
    );
};
```

### 6. User Interaction Flows

#### Flow 1: "Find My Optimal Config"
1. User sets hard constraints (sliders)
2. System filters to feasible configurations
3. User adjusts preference weights
4. System highlights optimal configuration
5. User explores nearby alternatives

#### Flow 2: "What-If Analysis"
1. Select baseline configuration
2. Adjust one constraint/weight
3. See impact on optimal choice
4. Compare before/after configurations
5. Save interesting scenarios

### 7. Demo Scenarios

#### Scenario 1: Resource-Constrained Edge Device
```javascript
const edgeDeviceDemo = {
    constraints: {
        maxLUT: 50,        // Limited resources
        maxPower: 10,      // Battery powered
        minThroughput: 100 // Modest performance
    },
    narrative: "Finding optimal config for edge deployment..."
};
```

#### Scenario 2: High-Performance Server
```javascript
const serverDemo = {
    constraints: {
        maxLUT: 95,         // Can use most resources
        maxPower: 100,      // Power available
        minThroughput: 1500 // Need high performance
    },
    narrative: "Maximizing throughput for datacenter..."
};
```

### 8. Visual Polish

#### Particle Effects
- Particles flow along Pareto frontier
- Constraint changes trigger particle bursts
- Optimal point has pulsing glow

#### Smooth Animations
```css
.configuration-point {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.pareto-frontier {
    stroke-dasharray: 1000;
    stroke-dashoffset: 1000;
    animation: dash 2s linear forwards;
}

@keyframes dash {
    to { stroke-dashoffset: 0; }
}
```

### 9. Performance Optimizations

- **LOD System**: Reduce detail for distant points
- **Octree Spatial Index**: Fast point queries
- **WebGL Instancing**: Render thousands of points efficiently
- **Constraint Caching**: Pre-compute common constraint combinations

### 10. Export Capabilities

- **Configuration Report**: PDF with selected config details
- **Pareto Analysis**: CSV of all Pareto-optimal configs
- **Interactive HTML**: Standalone visualization
- **Python Notebook**: Reproducible analysis code