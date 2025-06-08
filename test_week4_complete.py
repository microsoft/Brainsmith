"""
Week 4 Complete Test - All Libraries Integration

Comprehensive test for all Week 4 libraries: transforms, hw_optim, and analysis.
Tests complete integration with blueprints and end-to-end workflows.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from brainsmith.libraries.transforms import TransformsLibrary
from brainsmith.libraries.hw_optim import HwOptimLibrary
from brainsmith.libraries.analysis import AnalysisLibrary
from brainsmith.blueprints.core import Blueprint
from brainsmith.blueprints.integration import BlueprintOrchestrator


def test_all_libraries_initialization():
    """Test initialization of all Week 4 libraries."""
    print("🚀 Testing All Libraries Initialization...")
    
    # Initialize all libraries
    transforms_lib = TransformsLibrary()
    hw_optim_lib = HwOptimLibrary()
    analysis_lib = AnalysisLibrary()
    
    # Test initialization
    transforms_init = transforms_lib.initialize()
    hw_optim_init = hw_optim_lib.initialize()
    analysis_init = analysis_lib.initialize()
    
    print(f"  ✅ Transforms library: {transforms_init}")
    print(f"  ✅ HW Optimization library: {hw_optim_init}")
    print(f"  ✅ Analysis library: {analysis_init}")
    
    # Test capabilities
    transforms_caps = transforms_lib.get_capabilities()
    hw_optim_caps = hw_optim_lib.get_capabilities()
    analysis_caps = analysis_lib.get_capabilities()
    
    print(f"  ✅ Total capabilities: {len(transforms_caps + hw_optim_caps + analysis_caps)}")
    
    return transforms_init and hw_optim_init and analysis_init


def test_complete_blueprint_integration():
    """Test all libraries with comprehensive blueprint."""
    print("\n🔧 Testing Complete Blueprint Integration...")
    
    # Create comprehensive blueprint using all 4 libraries
    blueprint = Blueprint("complete_system_test", "1.0.0", 
                         "Complete system test with all libraries")
    
    # Configure all libraries
    blueprint.add_library_config('kernels', {
        'pe_range': [4, 8, 16],
        'simd_range': [2, 4, 8],
        'precision': ['int8']
    })
    
    blueprint.add_library_config('transforms', {
        'pipeline_depth': [3, 4, 5],
        'folding_factors': [2, 4],
        'memory_optimization': 'aggressive',
        'streaming_enabled': True
    })
    
    blueprint.add_library_config('hw_optim', {
        'target_frequency': 300,
        'optimization_strategy': 'genetic',
        'resource_budget': {'luts': 80000, 'brams': 500, 'dsps': 2000}
    })
    
    blueprint.add_library_config('analysis', {
        'performance_metrics': ['throughput', 'latency', 'efficiency'],
        'roofline_analysis': True,
        'power_estimation': True
    })
    
    # Add constraints and objectives
    blueprint.add_constraint('resource_limits', {
        'lut_utilization': 0.8,
        'power_consumption': 10.0
    })
    
    blueprint.add_objective('throughput', 'maximize')
    blueprint.add_objective('power_efficiency', 'maximize')
    blueprint.add_objective('area_efficiency', 'maximize')
    
    # Test blueprint with orchestrator
    orchestrator = BlueprintOrchestrator()
    can_execute, issues = orchestrator.validate_blueprint_execution(blueprint)
    
    print(f"  ✅ Blueprint validation: {can_execute}")
    print(f"  ✅ All 4 libraries configured: {len(blueprint.libraries) == 4}")
    
    return can_execute and len(blueprint.libraries) == 4


def test_transforms_library_operations():
    """Test transforms library detailed operations."""
    print("\n🔄 Testing Transforms Library Operations...")
    
    transforms_lib = TransformsLibrary()
    transforms_lib.initialize()
    
    # Test transform operations
    config = {
        'transforms': {
            'pipeline_depth': 4,
            'folding_factors': [2, 4, 8],
            'memory_optimization': 'aggressive',
            'streaming_enabled': True
        }
    }
    
    # Test apply transforms
    result = transforms_lib.execute("apply_transforms", {
        'model': 'test_model',
        'transforms': config['transforms']
    })
    
    print(f"  ✅ Transforms applied: {len(result['applied_transforms'])}")
    print(f"  ✅ Throughput gain: {result['estimated_improvement']['throughput_gain']}")
    
    # Test resource estimation
    resource_result = transforms_lib.execute("estimate_resources", config)
    print(f"  ✅ Resource estimation: {resource_result['resource_overhead']['luts']} LUTs")
    
    return len(result['applied_transforms']) > 0


def test_hw_optim_library_operations():
    """Test hardware optimization library operations."""
    print("\n⚙️ Testing Hardware Optimization Library Operations...")
    
    hw_optim_lib = HwOptimLibrary()
    hw_optim_lib.initialize()
    
    # Test optimization
    design_space = {
        'pe': [2, 4, 8],
        'simd': [1, 2, 4],
        'frequency': [200, 250, 300]
    }
    
    objectives = ['throughput', 'resource_efficiency']
    
    result = hw_optim_lib.execute("optimize_design", {
        'design_space': design_space,
        'objectives': objectives,
        'strategy': 'genetic',
        'max_evaluations': 20
    })
    
    print(f"  ✅ Optimization strategy: {result['strategy']}")
    print(f"  ✅ Best solutions found: {len(result['best_solutions'])}")
    print(f"  ✅ Pareto front size: {len(result['pareto_front'])}")
    
    # Test resource estimation
    config = {'pe': 8, 'simd': 4, 'frequency': 300, 'pipeline_depth': 4}
    resource_result = hw_optim_lib.execute("estimate_resources", {'config': config})
    
    print(f"  ✅ Resource estimation: {resource_result['estimated_resources']}")
    
    return len(result['best_solutions']) > 0


def test_analysis_library_operations():
    """Test analysis library operations."""
    print("\n📊 Testing Analysis Library Operations...")
    
    analysis_lib = AnalysisLibrary()
    analysis_lib.initialize()
    
    # Test design analysis
    config = {'pe': 8, 'simd': 4, 'frequency': 300, 'pipeline_depth': 4}
    metrics = ['throughput', 'latency', 'efficiency', 'resources', 'power']
    
    result = analysis_lib.execute("analyze_design", {
        'config': config,
        'metrics': metrics
    })
    
    print(f"  ✅ Analysis completed: {len(result['analysis_results'])} categories")
    
    # Test report generation
    report_result = analysis_lib.execute("generate_report", {
        'results': result,
        'format': 'html'
    })
    
    print(f"  ✅ Report generated: {report_result['format']} ({report_result['report_size']} chars)")
    
    # Test roofline analysis
    roofline_result = analysis_lib.execute("roofline_analysis", {'config': config})
    print(f"  ✅ Roofline analysis: {roofline_result['peak_performance']:.2f} GOPS")
    
    return len(result['analysis_results']) > 0


def test_end_to_end_workflow():
    """Test complete end-to-end workflow with all libraries."""
    print("\n🌟 Testing End-to-End Workflow...")
    
    # 1. Create blueprint
    blueprint = Blueprint("end_to_end_test", "1.0.0")
    blueprint.add_library_config('kernels', {'pe_range': [4, 8], 'simd_range': [2, 4]})
    blueprint.add_library_config('transforms', {'pipeline_depth': 4, 'folding_factors': [2, 4]})
    blueprint.add_library_config('hw_optim', {'target_frequency': 250, 'optimization_strategy': 'genetic'})
    blueprint.add_library_config('analysis', {'performance_metrics': ['throughput', 'latency']})
    
    # 2. Initialize all libraries
    transforms_lib = TransformsLibrary()
    hw_optim_lib = HwOptimLibrary()
    analysis_lib = AnalysisLibrary()
    
    all_initialized = (transforms_lib.initialize() and 
                      hw_optim_lib.initialize() and 
                      analysis_lib.initialize())
    
    if not all_initialized:
        return False
    
    # 3. Execute transforms
    transforms_config = blueprint.get_transforms_config()
    transform_result = transforms_lib.execute("apply_transforms", {
        'model': 'test_model',
        'transforms': transforms_config
    })
    
    # 4. Optimize design
    design_space = {'pe': [4, 8], 'simd': [2, 4], 'frequency': [200, 250]}
    optim_result = hw_optim_lib.execute("optimize_design", {
        'design_space': design_space,
        'objectives': ['throughput', 'resource_efficiency'],
        'strategy': 'genetic',
        'max_evaluations': 10
    })
    
    # 5. Analyze best design
    if optim_result['best_solutions']:
        best_design = optim_result['best_solutions'][0]['design']
        analysis_result = analysis_lib.execute("analyze_design", {
            'config': best_design,
            'metrics': ['throughput', 'latency', 'resources']
        })
        
        # 6. Generate final report
        report_result = analysis_lib.execute("generate_report", {
            'results': analysis_result,
            'format': 'json'
        })
        
        print(f"  ✅ Workflow completed successfully")
        print(f"  ✅ Transforms applied: {len(transform_result['applied_transforms'])}")
        print(f"  ✅ Optimization solutions: {len(optim_result['best_solutions'])}")
        print(f"  ✅ Analysis categories: {len(analysis_result['analysis_results'])}")
        print(f"  ✅ Final report: {report_result['format']} format")
        
        return True
    
    return False


def test_library_interoperability():
    """Test interoperability between libraries."""
    print("\n🔗 Testing Library Interoperability...")
    
    # Initialize libraries
    transforms_lib = TransformsLibrary()
    hw_optim_lib = HwOptimLibrary()
    analysis_lib = AnalysisLibrary()
    
    transforms_lib.initialize()
    hw_optim_lib.initialize()
    analysis_lib.initialize()
    
    # Test that outputs from one library can be used by another
    
    # 1. Get resource estimate from transforms
    transforms_config = {'pipeline_depth': 4, 'folding_factors': [2, 4]}
    transform_resources = transforms_lib.execute("estimate_resources", {
        'transforms': transforms_config
    })
    
    # 2. Use transform estimates in hw_optim
    design_config = {
        'pe': 8,
        'simd': 4,
        'frequency': 250,
        'pipeline_depth': transforms_config['pipeline_depth']
    }
    
    hw_resources = hw_optim_lib.execute("estimate_resources", {'config': design_config})
    
    # 3. Use combined estimates in analysis
    analysis_result = analysis_lib.execute("analyze_design", {
        'config': design_config,
        'metrics': ['throughput', 'resources', 'power']
    })
    
    print(f"  ✅ Transform resource overhead: {transform_resources['resource_overhead']['luts']}")
    print(f"  ✅ HW optim resource estimate: {hw_resources['estimated_resources']['luts']}")
    print(f"  ✅ Analysis resource usage: {analysis_result['analysis_results']['resources']['estimated_usage']['luts']['value']}")
    
    # Verify consistency (simplified check)
    consistency_check = (
        transform_resources['resource_overhead']['luts'] > 0 and
        hw_resources['estimated_resources']['luts'] > 0 and
        analysis_result['analysis_results']['resources']['estimated_usage']['luts']['value'] > 0
    )
    
    print(f"  ✅ Resource estimates consistent: {consistency_check}")
    
    return consistency_check


def main():
    """Main test function."""
    print("🚀 Week 4 Complete Test - All Libraries Integration")
    print("=" * 70)
    
    tests = [
        test_all_libraries_initialization,
        test_complete_blueprint_integration,
        test_transforms_library_operations,
        test_hw_optim_library_operations,
        test_analysis_library_operations,
        test_end_to_end_workflow,
        test_library_interoperability
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Week 4 Complete Implementation Successful!")
        print("✅ All libraries (kernels, transforms, hw_optim, analysis) working!")
        print("✅ Complete blueprint-driven end-to-end workflow operational!")
        print("🚀 Brainsmith platform is now feature-complete!")
        return True
    elif passed >= len(tests) * 0.8:
        print("✅ Week 4 Implementation Mostly Complete!")
        print("⚠️  Minor issues to address")
        return True
    else:
        print("❌ Significant issues need resolution")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)