# 🔗 BrainSmith Analysis Hooks

**Data exposure framework for external analysis tools**

---

## 🎯 **Goals & Philosophy**

The BrainSmith Analysis module follows a **hooks-first philosophy**: instead of maintaining custom analysis implementations, we expose structured data that allows users to integrate their preferred external analysis tools.

### **Key Principles:**
- ✅ **Data exposure, not implementation** - Provide structured data for external tools
- ✅ **Zero maintenance burden** - No custom analysis algorithms to maintain
- ✅ **User choice** - Support pandas, scipy, scikit-learn, and custom tools
- ✅ **Better functionality** - Access to full capabilities of mature libraries

### **Why Hooks Instead of Custom Analysis?**
- **External libraries are better**: pandas, scipy, and scikit-learn are maintained by specialists
- **Reduced maintenance**: No need to keep up with statistical research or ML trends
- **User flexibility**: Choose the best tool for each analysis task
- **Future-proof**: New analysis tools can be easily integrated

---

## 📦 **Module Structure**

```
brainsmith/analysis/
├── __init__.py          # Main exports and documentation
├── hooks.py            # Core data exposure functions
├── adapters.py         # External tool format converters
├── utils.py            # Basic utility functions
└── README.md           # This documentation
```

---

## 🚀 **Quick Start**

### **1. Get Analysis Data from BrainSmith**
```python
import brainsmith

# Run DSE and get results
results = brainsmith.forge("model.onnx", "blueprint.yaml")
analysis_data = results['analysis_data']  # Structured data for external tools
```

### **2. Use Your Preferred Analysis Tool**

#### **Option A: Pandas (Data Analysis)**
```python
import pandas as pd
from brainsmith.analysis import pandas_adapter

# Convert to pandas DataFrame
df = pandas_adapter(analysis_data)

# Analyze with pandas
summary = df.describe()
best_solution = df.loc[df['objective_0'].idxmax()]
print(f"Best throughput: {best_solution['objective_0']}")
```

#### **Option B: SciPy (Statistical Analysis)**
```python
import scipy.stats as stats
from brainsmith.analysis import scipy_adapter

# Convert to scipy format
scipy_data = scipy_adapter(analysis_data)

# Statistical analysis
throughput = scipy_data['arrays']['objective_0']
stat, p_value = stats.normaltest(throughput)
print(f"Normal distribution: {'Yes' if p_value > 0.05 else 'No'}")
```

#### **Option C: Scikit-learn (Machine Learning)**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from brainsmith.analysis import sklearn_adapter

# Convert to ML format
ml_data = sklearn_adapter(analysis_data)
X, y = ml_data['X'], ml_data['y']

# Train regression model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression().fit(X_scaled, y[:, 0])
print(f"Model R² score: {model.score(X_scaled, y[:, 0]):.3f}")
```

---

## 📊 **Core Functions**

### **Data Exposure**
```python
from brainsmith.analysis import expose_analysis_data

# Expose structured data for external tools
data = expose_analysis_data(dse_results)
# Returns:
# {
#   'solutions': [...],        # Design solutions with parameters/objectives
#   'metrics': {...},         # Metric arrays (numpy)
#   'pareto_frontier': [...], # Pareto-optimal solution indices  
#   'metadata': {...}         # Analysis metadata
# }
```

### **External Tool Adapters**
```python
from brainsmith.analysis import pandas_adapter, scipy_adapter, sklearn_adapter

# Convert to pandas DataFrame
df = pandas_adapter(analysis_data)

# Prepare for scipy analysis
scipy_data = scipy_adapter(analysis_data)

# Format for scikit-learn
ml_data = sklearn_adapter(analysis_data)
```

### **Custom Analyzer Registration**
```python
from brainsmith.analysis import register_analyzer

def power_efficiency_analyzer(analysis_data):
    """Custom analyzer for power efficiency."""
    solutions = analysis_data['solutions']
    efficiencies = []
    
    for sol in solutions:
        objectives = sol['objectives']
        if len(objectives) >= 3:  # [throughput, latency, power]
            throughput, power = objectives[0], objectives[2]
            efficiency = throughput / power if power > 0 else 0
            efficiencies.append(efficiency)
    
    return {
        'mean_efficiency': np.mean(efficiencies),
        'max_efficiency': np.max(efficiencies),
        'efficiency_values': efficiencies
    }

# Register custom analyzer
register_analyzer('power_efficiency', power_efficiency_analyzer)
```

---

## 💡 **Usage Examples**

### **Example 1: Performance Analysis with Pandas**
```python
import pandas as pd
import matplotlib.pyplot as plt
from brainsmith.analysis import pandas_adapter

# Get BrainSmith results
results = brainsmith.forge("cnn_model.onnx", "fpga_blueprint.yaml")
df = pandas_adapter(results['analysis_data'])

# Performance analysis
print("Performance Summary:")
print(df[['objective_0', 'objective_1', 'objective_2']].describe())

# Find best solutions
best_throughput = df.loc[df['objective_0'].idxmax()]
best_power = df.loc[df['objective_2'].idxmin()]

print(f"\nBest Throughput: {best_throughput['objective_0']:.1f} ops/s")
print(f"  Parameters: PE={best_throughput['param_pe']}, SIMD={best_throughput['param_simd']}")

print(f"\nLowest Power: {best_power['objective_2']:.1f} W")
print(f"  Parameters: PE={best_power['param_pe']}, SIMD={best_power['param_simd']}")

# Plot trade-offs
plt.scatter(df['objective_0'], df['objective_2'])
plt.xlabel('Throughput (ops/s)')
plt.ylabel('Power (W)')
plt.title('Throughput vs Power Trade-off')
plt.show()
```

### **Example 2: Statistical Analysis with SciPy**
```python
import scipy.stats as stats
import numpy as np
from brainsmith.analysis import scipy_adapter

# Get BrainSmith results
results = brainsmith.forge("transformer_model.onnx", "edge_blueprint.yaml")
scipy_data = scipy_adapter(results['analysis_data'])

# Statistical analysis
for metric_name, values in scipy_data['arrays'].items():
    print(f"\n{metric_name} Statistics:")
    
    # Descriptive statistics
    print(f"  Mean: {np.mean(values):.2f}")
    print(f"  Std:  {np.std(values):.2f}")
    
    # Test for normality
    stat, p_value = stats.shapiro(values)
    is_normal = "Yes" if p_value > 0.05 else "No"
    print(f"  Normal distribution: {is_normal} (p={p_value:.3f})")
    
    # Confidence interval
    ci = stats.t.interval(0.95, len(values)-1, 
                         loc=np.mean(values), 
                         scale=stats.sem(values))
    print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
```

### **Example 3: Machine Learning with Scikit-learn**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from brainsmith.analysis import sklearn_adapter

# Get BrainSmith results
results = brainsmith.forge("resnet_model.onnx", "datacenter_blueprint.yaml")
ml_data = sklearn_adapter(results['analysis_data'])

if ml_data:
    X, y = ml_data['X'], ml_data['y']
    feature_names = ml_data['feature_names']
    
    # Preprocess features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model to predict throughput
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_scaled, y[:, 0], cv=5)
    
    print(f"Throughput Prediction Model:")
    print(f"  Cross-validation R² score: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Fit model and analyze feature importance
    model.fit(X_scaled, y[:, 0])
    importances = model.feature_importances_
    
    print(f"\nFeature Importance for Throughput:")
    for feature, importance in zip(feature_names, importances):
        print(f"  {feature}: {importance:.3f}")
```

### **Example 4: Custom Domain Analysis**
```python
import numpy as np
from brainsmith.analysis import register_analyzer, expose_analysis_data

def fpga_efficiency_analyzer(analysis_data):
    """FPGA-specific efficiency analyzer."""
    solutions = analysis_data['solutions']
    
    # Calculate multiple efficiency metrics
    power_efficiency = []
    area_efficiency = []
    
    for sol in solutions:
        params = sol['parameters']
        objectives = sol['objectives']  # [throughput, latency, power]
        
        if len(objectives) >= 3:
            throughput, latency, power = objectives[0], objectives[1], objectives[2]
            
            # Power efficiency (ops/watt)
            power_eff = throughput / power if power > 0 else 0
            power_efficiency.append(power_eff)
            
            # Estimate area efficiency (ops/LUT)
            est_luts = params.get('pe', 1) * params.get('simd', 1) * 100  # Rough estimate
            area_eff = throughput / est_luts if est_luts > 0 else 0
            area_efficiency.append(area_eff)
    
    # Find best solutions
    best_power_idx = np.argmax(power_efficiency)
    best_area_idx = np.argmax(area_efficiency)
    
    return {
        'power_efficiency': {
            'values': power_efficiency,
            'mean': np.mean(power_efficiency),
            'best': np.max(power_efficiency),
            'best_solution_idx': best_power_idx
        },
        'area_efficiency': {
            'values': area_efficiency,
            'mean': np.mean(area_efficiency),
            'best': np.max(area_efficiency),
            'best_solution_idx': best_area_idx
        },
        'recommendations': [
            f"Best power efficiency: Solution #{best_power_idx} ({np.max(power_efficiency):.1f} ops/W)",
            f"Best area efficiency: Solution #{best_area_idx} ({np.max(area_efficiency):.3f} ops/LUT)"
        ]
    }

# Register and use custom analyzer
register_analyzer('fpga_efficiency', fpga_efficiency_analyzer)

# Use with BrainSmith results
results = brainsmith.forge("model.onnx", "blueprint.yaml")
analysis = fpga_efficiency_analyzer(results['analysis_data'])

for recommendation in analysis['recommendations']:
    print(recommendation)
```

---

## 🔧 **Advanced Features**

### **Raw Data Access**
```python
from brainsmith.analysis import get_raw_data

# Get raw metric arrays
raw_metrics = get_raw_data(dse_results)
# Returns: {'objective_0': array([...]), 'objective_1': array([...]), ...}

# Direct numpy operations
throughput_mean = np.mean(raw_metrics['objective_0'])
latency_std = np.std(raw_metrics['objective_1'])
```

### **Pareto Frontier Analysis**
```python
# Access Pareto-optimal solutions
analysis_data = expose_analysis_data(dse_results)
pareto_indices = analysis_data['pareto_frontier']
pareto_solutions = [analysis_data['solutions'][i] for i in pareto_indices]

print(f"Found {len(pareto_solutions)} Pareto-optimal solutions:")
for i, sol in enumerate(pareto_solutions):
    print(f"  Solution {sol['id']}: {sol['objectives']}")
```

### **Multi-Tool Workflow**
```python
# Combine multiple analysis tools
results = brainsmith.forge("model.onnx", "blueprint.yaml")
data = results['analysis_data']

# Step 1: Basic analysis with pandas
df = pandas_adapter(data)
top_10_percent = df.nlargest(int(len(df) * 0.1), 'objective_0')

# Step 2: Statistical significance with scipy
from scipy.stats import ttest_ind
top_throughput = df.nlargest(5, 'objective_0')['objective_0'].values
bottom_throughput = df.nsmallest(5, 'objective_0')['objective_0'].values
stat, p_value = ttest_ind(top_throughput, bottom_throughput)
print(f"Top vs bottom throughput difference significant: {p_value < 0.05}")

# Step 3: Predictive modeling with scikit-learn
ml_data = sklearn_adapter(data)
# ... train model as shown above

# Step 4: Custom domain analysis
custom_analysis = fpga_efficiency_analyzer(data)
print("Custom insights:", custom_analysis['recommendations'])
```

---

## 📚 **Integration with External Libraries**

### **Supported Libraries**
- **Pandas**: Data manipulation and analysis
- **SciPy**: Statistical analysis and hypothesis testing
- **Scikit-learn**: Machine learning and preprocessing
- **NumPy**: Numerical operations (built-in support)
- **Matplotlib/Seaborn**: Visualization (via pandas/numpy data)
- **Plotly**: Interactive visualization (via pandas data)
- **Statsmodels**: Advanced statistical modeling
- **Any custom library**: Via analyzer registration

### **Installation Recommendations**
```bash
# Core analysis stack
pip install pandas scipy scikit-learn

# Visualization
pip install matplotlib seaborn plotly

# Advanced statistics
pip install statsmodels

# High-performance alternatives
pip install polars  # Faster than pandas for large datasets
pip install numpy   # Already included with BrainSmith
```

---

## ⚡ **Performance Tips**

### **Large Datasets**
```python
# For large DSE results, consider using polars instead of pandas
try:
    import polars as pl
    # Convert pandas output to polars for better performance
    df_pandas = pandas_adapter(analysis_data)
    df_polars = pl.from_pandas(df_pandas)
    # Use polars for analysis...
except ImportError:
    # Fallback to pandas
    df = pandas_adapter(analysis_data)
```

### **Memory Efficiency**
```python
# Get only raw metrics for memory-efficient processing
raw_metrics = get_raw_data(dse_results)

# Process metrics individually to reduce memory usage
for metric_name, values in raw_metrics.items():
    # Process one metric at a time
    analysis = process_metric(values)
    save_analysis(metric_name, analysis)
```

---

## 🛠 **Troubleshooting**

### **Common Issues**

**Q: `pandas_adapter()` returns `None`**  
A: Pandas is not installed. Install with `pip install pandas`

**Q: Empty analysis data**  
A: Check that DSE results contain solutions with `objective_values`

**Q: Missing metrics in output**  
A: Ensure solution objects have `design_parameters` and `objective_values` attributes

**Q: Pareto frontier is empty**  
A: Verify that objectives are properly formatted as numeric values

### **Debug Mode**
```python
# Enable debug logging
import logging
logging.getLogger('brainsmith.analysis').setLevel(logging.DEBUG)

# Check data structure
data = expose_analysis_data(dse_results)
print("Analysis data structure:", data.keys())
print("Number of solutions:", len(data['solutions']))
print("Available metrics:", list(data['metrics'].keys()))
```

---

## 🔄 **Migration from Custom Analysis**

If you previously used custom BrainSmith analysis functions:

### **Old Custom Analysis → New Hooks**
```python
# OLD: Custom analysis (removed)
# from brainsmith.analysis import PerformanceAnalyzer
# analyzer = PerformanceAnalyzer()
# results = analyzer.analyze_performance(solutions)

# NEW: External tools
import pandas as pd
from brainsmith.analysis import pandas_adapter

results = brainsmith.forge("model.onnx", "blueprint.yaml")
df = pandas_adapter(results['analysis_data'])
summary = df.describe()  # Better than custom analyzer
```

### **Old Statistical Analysis → SciPy**
```python
# OLD: Custom statistics (removed)
# from brainsmith.analysis import StatisticalAnalyzer
# stats = StatisticalAnalyzer().analyze(data)

# NEW: SciPy
import scipy.stats as stats
from brainsmith.analysis import scipy_adapter

scipy_data = scipy_adapter(results['analysis_data'])
for metric, values in scipy_data['arrays'].items():
    stat, p_value = stats.normaltest(values)  # Better than custom stats
```

---

## 🤝 **Contributing**

### **Adding New Adapters**
To support a new external library, add an adapter function:

```python
def new_library_adapter(analysis_data: Dict[str, Any]) -> Any:
    """Adapter for NewLibrary."""
    try:
        import new_library
        # Convert analysis_data to new_library format
        return new_library.from_dict(analysis_data)
    except ImportError:
        return None
```

### **Reporting Issues**
- Missing external library support
- Data format compatibility issues  
- Performance bottlenecks
- Documentation improvements

---

## 📄 **License**

This module is part of the BrainSmith project and follows the same license terms.

---

**🔗 Remember: The goal is to expose data for external tools, not to reimplement analysis algorithms. Use the power of the Python data science ecosystem!**