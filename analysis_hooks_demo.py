"""
BrainSmith Analysis Hooks - Demo Script

Demonstrates how to use the new hooks-based analysis module with external libraries.
This script shows the transformation from custom analysis to external tool integration.
"""

import numpy as np
from typing import Dict, Any

print("🔗 BrainSmith Analysis Hooks Demo")
print("=" * 50)

# Simulate a forge() result with analysis data
def simulate_forge_results():
    """Simulate results from brainsmith.forge() with analysis hooks."""
    
    # Mock DSE results (what would come from actual DSE)
    class MockSolution:
        def __init__(self, pe, simd, freq, throughput, latency, power):
            self.design_parameters = {'pe': pe, 'simd': simd, 'freq': freq}
            self.objective_values = [throughput, latency, power]
            self.constraint_violations = []
            self.metadata = {'valid': True}
    
    dse_results = [
        MockSolution(4, 2, 100, 150.0, 45.0, 12.5),
        MockSolution(8, 4, 125, 280.0, 28.0, 18.2),
        MockSolution(16, 8, 150, 520.0, 15.0, 32.1),
        MockSolution(32, 16, 175, 950.0, 8.5, 58.7),
        MockSolution(64, 32, 200, 1800.0, 4.2, 105.3)
    ]
    
    # Import analysis hooks
    try:
        from brainsmith.analysis import expose_analysis_data, register_analyzer
        analysis_data = expose_analysis_data(dse_results)
        
        return {
            'dataflow_graph': {'onnx_model': 'mock_model.onnx'},
            'dataflow_core': None,
            'metrics': {'performance': {'throughput': 1800.0}, 'resources': {'power': 105.3}},
            'analysis_data': analysis_data,
            'analysis_hooks': {
                'register_analyzer': register_analyzer,
                'available_adapters': ['pandas', 'scipy', 'sklearn']
            }
        }
    except ImportError:
        print("❌ Analysis hooks not available - using fallback")
        return {
            'dataflow_graph': {'onnx_model': 'mock_model.onnx'},
            'metrics': {'performance': {'throughput': 1800.0}},
            'analysis_data': {'solutions': [], 'metrics': {}},
            'analysis_hooks': None
        }

# Demo 1: Basic Data Exposure
print("\n📊 Demo 1: Basic Data Exposure")
print("-" * 30)

results = simulate_forge_results()
analysis_data = results.get('analysis_data', {})

print(f"✅ Exposed {len(analysis_data.get('solutions', []))} design solutions")
print(f"✅ Available metrics: {list(analysis_data.get('metrics', {}).keys())}")
print(f"✅ Pareto frontier: {len(analysis_data.get('pareto_frontier', []))} solutions")

# Demo 2: Pandas Integration
print("\n🐼 Demo 2: Pandas Analysis")
print("-" * 30)

try:
    import pandas as pd
    from brainsmith.analysis import pandas_adapter
    
    # Convert to pandas DataFrame
    df = pandas_adapter(analysis_data)
    
    if df is not None and not df.empty:
        print("✅ Successfully converted to pandas DataFrame")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Perform pandas analysis
        print("\n📈 Pandas Analysis Results:")
        print("   Throughput (objective_0) statistics:")
        if 'objective_0' in df.columns:
            stats = df['objective_0'].describe()
            print(f"     Mean: {stats['mean']:.1f}")
            print(f"     Max:  {stats['max']:.1f}")
            print(f"     Min:  {stats['min']:.1f}")
        
        # Find best solution
        if 'objective_0' in df.columns:
            best_idx = df['objective_0'].idxmax()
            best_solution = df.loc[best_idx]
            print(f"\n   🏆 Best solution (highest throughput):")
            print(f"     PE={best_solution.get('param_pe', 'N/A')}, "
                  f"SIMD={best_solution.get('param_simd', 'N/A')}, "
                  f"Throughput={best_solution.get('objective_0', 'N/A'):.1f}")
    else:
        print("❌ Pandas conversion failed")

except ImportError:
    print("❌ Pandas not available")
    print("   Install with: pip install pandas")
except Exception as e:
    print(f"❌ Pandas analysis failed: {e}")

# Demo 3: SciPy Statistical Analysis
print("\n🔬 Demo 3: SciPy Statistical Analysis")
print("-" * 40)

try:
    import scipy.stats as stats
    from brainsmith.analysis import scipy_adapter
    
    # Convert to scipy format
    scipy_data = scipy_adapter(analysis_data)
    
    print("✅ Successfully converted to scipy format")
    print(f"   Sample size: {scipy_data['sample_size']}")
    print(f"   Metrics: {scipy_data['metric_names']}")
    
    # Perform statistical analysis
    print("\n📊 Statistical Analysis Results:")
    for metric_name, values in scipy_data['arrays'].items():
        if len(values) > 2:
            # Normality test
            stat, p_value = stats.shapiro(values)
            normal = "Yes" if p_value > 0.05 else "No"
            
            print(f"   {metric_name}:")
            print(f"     Normal distribution: {normal} (p={p_value:.3f})")
            print(f"     Mean ± Std: {np.mean(values):.1f} ± {np.std(values):.1f}")

except ImportError:
    print("❌ SciPy not available")
    print("   Install with: pip install scipy")
except Exception as e:
    print(f"❌ SciPy analysis failed: {e}")

# Demo 4: Scikit-learn Machine Learning
print("\n🤖 Demo 4: Scikit-learn Machine Learning")
print("-" * 42)

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from brainsmith.analysis import sklearn_adapter
    
    # Convert to sklearn format
    sklearn_data = sklearn_adapter(analysis_data)
    
    if sklearn_data and len(sklearn_data['X']) > 1:
        X = sklearn_data['X']  # Features (design parameters)
        y = sklearn_data['y']  # Targets (objectives)
        
        print("✅ Successfully converted to scikit-learn format")
        print(f"   Features: {sklearn_data['feature_names']}")
        print(f"   Targets: {sklearn_data['target_names']}")
        print(f"   Training samples: {X.shape[0]}")
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train regression model for throughput prediction
        if y.shape[1] > 0:  # Has objectives
            model = LinearRegression()
            model.fit(X_scaled, y[:, 0])  # Predict first objective (throughput)
            
            r2_score = model.score(X_scaled, y[:, 0])
            print(f"\n🎯 Throughput Prediction Model:")
            print(f"   R² score: {r2_score:.3f}")
            print(f"   Feature importance:")
            
            for feature, coef in zip(sklearn_data['feature_names'], model.coef_):
                print(f"     {feature}: {coef:.2f}")
    else:
        print("❌ Insufficient data for ML analysis")

except ImportError:
    print("❌ Scikit-learn not available")
    print("   Install with: pip install scikit-learn")
except Exception as e:
    print(f"❌ Scikit-learn analysis failed: {e}")

# Demo 5: Custom Analysis Registration
print("\n🔧 Demo 5: Custom Analysis Registration")
print("-" * 42)

try:
    from brainsmith.analysis import register_analyzer, get_registered_analyzers
    
    # Define custom analyzer
    def power_efficiency_analyzer(analysis_data):
        """Custom analyzer for power efficiency."""
        solutions = analysis_data.get('solutions', [])
        if not solutions:
            return {'error': 'No solutions available'}
        
        efficiencies = []
        for sol in solutions:
            objectives = sol.get('objectives', [])
            if len(objectives) >= 3:  # [throughput, latency, power]
                throughput, latency, power = objectives[0], objectives[1], objectives[2]
                efficiency = throughput / power if power > 0 else 0
                efficiencies.append(efficiency)
        
        if efficiencies:
            return {
                'power_efficiency': {
                    'values': efficiencies,
                    'mean': np.mean(efficiencies),
                    'max': np.max(efficiencies),
                    'best_solution_idx': np.argmax(efficiencies)
                }
            }
        return {'error': 'Could not calculate power efficiency'}
    
    # Register custom analyzer
    register_analyzer('power_efficiency', power_efficiency_analyzer)
    
    # Verify registration
    analyzers = get_registered_analyzers()
    print(f"✅ Registered {len(analyzers)} custom analyzers")
    print(f"   Available: {list(analyzers.keys())}")
    
    # Use custom analyzer
    custom_result = power_efficiency_analyzer(analysis_data)
    if 'power_efficiency' in custom_result:
        pe = custom_result['power_efficiency']
        print(f"\n⚡ Power Efficiency Analysis:")
        print(f"   Mean efficiency: {pe['mean']:.2f} ops/watt")
        print(f"   Best efficiency: {pe['max']:.2f} ops/watt")
        print(f"   Best solution: #{pe['best_solution_idx']}")
    
except Exception as e:
    print(f"❌ Custom analyzer registration failed: {e}")

# Summary
print("\n🎉 Demo Summary")
print("=" * 20)
print("✅ Analysis Hooks Implementation Complete!")
print(f"   📊 Data exposed for {len(analysis_data.get('solutions', []))} solutions")
print("   🐼 Pandas integration: DataFrame conversion")
print("   🔬 SciPy integration: Statistical analysis")
print("   🤖 Scikit-learn integration: ML preprocessing")
print("   🔧 Custom analyzer registration: Power efficiency")
print()
print("🚀 Next Steps:")
print("   • Use your preferred analysis libraries")
print("   • Implement domain-specific analyzers")
print("   • Integrate with existing workflows")
print("   • No more custom analysis maintenance!")

print("\n" + "=" * 50)
print("🔗 Hooks-based analysis complete! 🔗")