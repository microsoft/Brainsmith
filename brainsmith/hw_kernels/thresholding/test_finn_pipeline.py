############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
FINN pipeline integration test for thresholding AutoHWCustomOp.

This test validates complete FINN transformation pipeline integration
following established FINN testing patterns for comprehensive validation.
"""

import sys
import os
import numpy as np
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ONNX imports
import onnx
from onnx import TensorProto, helper
from onnx import numpy_helper

# QONNX imports
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import gen_finn_dt_tensor

# FINN imports
try:
    import finn.core.onnx_exec as oxe
    from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
    from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
    from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
    from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
    from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
    from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
    from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
    FINN_AVAILABLE = True
except ImportError:
    FINN_AVAILABLE = False
    print("⚠️ FINN transformations not available - pipeline testing will be skipped")


class FINNPipelineTester:
    """Tests complete FINN transformation pipeline integration."""
    
    def __init__(self):
        self.fpgapart = "xczu3eg-sbva484-1-e"
        self.test_config = {"channels": 64, "pe": 8, "input_dt": "INT8", "output_dt": "UINT4"}
    
    def create_thresholding_model(self, config: Dict[str, Any]) -> ModelWrapper:
        """Create thresholding model following FINN patterns."""
        
        # Create input/output tensors
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, config["channels"]])
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, config["channels"]])
        
        # Create threshold initializer (ascending order)
        thresh_vals = np.array([[-10, 0, 10]] * config["channels"], dtype=np.float32)
        thresh_vals = np.sort(thresh_vals, axis=1)  # Ensure ascending order
        thresh_init = numpy_helper.from_array(thresh_vals, name="thresh")
        
        # Create thresholding node with all required attributes
        node = helper.make_node(
            "Thresholding",
            ["inp", "thresh"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            NumChannels=config["channels"],
            PE=config["pe"],
            numSteps=3,  # Number of threshold levels
            inputDataType=config["input_dt"],
            weightDataType="INT8",
            outputDataType=config["output_dt"],
            ActVal=0,  # Activation bias
            numInputVectors=[1],  # Input shape
            preferred_impl_style="hls"
        )
        
        # Create graph and model
        graph = helper.make_graph([node], "thresholding_graph", [inp], [outp], [thresh_init])
        model = helper.make_model(graph, producer_name="thresholding-pipeline-model")
        model_wrapper = ModelWrapper(model)
        
        # Set datatypes explicitly
        model_wrapper.set_tensor_datatype("inp", DataType[config["input_dt"]])
        model_wrapper.set_tensor_datatype("outp", DataType[config["output_dt"]])
        model_wrapper.set_tensor_datatype("thresh", DataType["INT8"])
        
        return model_wrapper
    
    def apply_basic_transformations(self, model: ModelWrapper) -> ModelWrapper:
        """Apply basic FINN transformations."""
        print("   🔄 Applying basic transformations...")
        
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        
        print("   ✅ Basic transformations completed")
        return model
    
    def apply_specialization(self, model: ModelWrapper) -> ModelWrapper:
        """Apply layer specialization transformation."""
        print("   🔧 Applying layer specialization...")
        
        model = model.transform(SpecializeLayers(self.fpgapart))
        
        print("   ✅ Layer specialization completed")
        return model
    
    def apply_dataflow_transformations(self, model: ModelWrapper) -> ModelWrapper:
        """Apply dataflow-specific transformations."""
        print("   ⚡ Applying dataflow transformations...")
        
        # Insert FIFO and data width converter if needed
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = model.transform(InsertDWC())
        
        print("   ✅ Dataflow transformations completed")
        return model
    
    def test_cppsim_pipeline(self, model: ModelWrapper, input_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Test CPPSIM execution pipeline."""
        print("   🔧 Testing CPPSIM pipeline...")
        
        # Set execution mode and prepare CPPSIM
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        
        # Execute and return output
        output = oxe.execute_onnx(model, input_dict)["outp"]
        print("   ✅ CPPSIM pipeline completed")
        return output
    
    def analyze_model_performance(self, model: ModelWrapper) -> Dict[str, Any]:
        """Analyze model performance characteristics."""
        print("   📊 Analyzing model performance...")
        
        analysis_results = {}
        
        try:
            # Cycle analysis
            cycles_dict = model.analysis(exp_cycles_per_layer)
            analysis_results["cycles"] = cycles_dict
            
            # Resource estimation (if available)
            try:
                res_dict = model.analysis(hls_synth_res_estimation)
                analysis_results["resources"] = res_dict
            except Exception as e:
                print(f"   ⚠️ Resource estimation not available: {e}")
                analysis_results["resources"] = None
            
        except Exception as e:
            print(f"   ⚠️ Performance analysis not available: {e}")
            analysis_results["cycles"] = None
            analysis_results["resources"] = None
        
        print("   ✅ Performance analysis completed")
        return analysis_results
    
    def test_complete_pipeline(self, config: Dict[str, Any]) -> bool:
        """Test complete FINN transformation pipeline."""
        print(f"\n🚀 Testing Complete FINN Pipeline: channels={config['channels']}, pe={config['pe']}")
        
        try:
            # Create model
            print("   📝 Creating thresholding model...")
            model = self.create_thresholding_model(config)
            
            # Generate test data
            input_shape = (1, config["channels"])
            input_dtype = DataType[config["input_dt"]]
            test_input = gen_finn_dt_tensor(input_dtype, input_shape).astype(np.float32)
            input_dict = {"inp": test_input}
            
            print(f"   📊 Generated test data: shape={test_input.shape}, range=[{test_input.min():.1f}, {test_input.max():.1f}]")
            
            # Get golden reference
            print("   🏆 Getting golden reference...")
            golden_output = oxe.execute_onnx(model, input_dict)["outp"]
            
            # Apply transformation pipeline
            model = self.apply_basic_transformations(model)
            model = self.apply_specialization(model)
            model = self.apply_dataflow_transformations(model)
            
            # Test CPPSIM execution (create copy by serializing/deserializing)
            import copy
            model_copy = copy.deepcopy(model)
            cppsim_output = self.test_cppsim_pipeline(model_copy, input_dict)
            
            # Analyze performance
            analysis = self.analyze_model_performance(model)
            
            # Validate outputs
            print("   🔍 Validating outputs...")
            np.testing.assert_array_equal(
                cppsim_output, golden_output,
                err_msg="CPPSIM output differs from golden reference"
            )
            
            # Report results
            print(f"   📊 Golden output: shape={golden_output.shape}, range=[{golden_output.min():.1f}, {golden_output.max():.1f}]")
            print(f"   📊 CPPSIM output: shape={cppsim_output.shape}, range=[{cppsim_output.min():.1f}, {cppsim_output.max():.1f}]")
            
            if analysis["cycles"]:
                print(f"   ⏱️ Cycle analysis: {analysis['cycles']}")
            
            if analysis["resources"]:
                print(f"   💾 Resource estimation: {analysis['resources']}")
            
            print("   ✅ Complete pipeline test passed!")
            return True
            
        except Exception as e:
            print(f"   ❌ Pipeline test failed: {e}")
            return False
    
    def test_pipeline_robustness(self) -> bool:
        """Test pipeline with multiple configurations."""
        print("\n🔬 Testing Pipeline Robustness...")
        
        test_configs = [
            {"channels": 32, "pe": 8, "input_dt": "INT8", "output_dt": "UINT4"},
            {"channels": 64, "pe": 16, "input_dt": "INT8", "output_dt": "UINT4"},
            {"channels": 128, "pe": 32, "input_dt": "INT8", "output_dt": "UINT4"},
        ]
        
        passed = 0
        for config in test_configs:
            try:
                if self.test_complete_pipeline(config):
                    passed += 1
            except Exception as e:
                print(f"   ❌ Configuration {config} failed: {e}")
        
        print(f"\n📊 Robustness Results: {passed}/{len(test_configs)} configurations passed")
        return passed == len(test_configs)
    
    def run_all_tests(self) -> bool:
        """Run all FINN pipeline tests."""
        if not FINN_AVAILABLE:
            print("⚠️ Skipping FINN pipeline tests - FINN transformations not available")
            return False
        
        print("🚀 Starting FINN Pipeline Integration Testing...")
        print("=" * 70)
        
        # Test main configuration
        main_test = self.test_complete_pipeline(self.test_config)
        
        # Test robustness
        robustness_test = self.test_pipeline_robustness()
        
        print("\n" + "=" * 70)
        success = main_test and robustness_test
        
        if success:
            print("🎉 All FINN pipeline tests passed!")
        else:
            print("❌ Some FINN pipeline tests failed!")
        
        return success


def main():
    """Main test function."""
    tester = FINNPipelineTester()
    success = tester.run_all_tests()
    return 0 if success else 1


# Pytest integration
@pytest.mark.parametrize("channels", [32, 64])
@pytest.mark.parametrize("pe", [8, 16])
@pytest.mark.parametrize("input_dt", ["INT8"])
@pytest.mark.parametrize("output_dt", ["UINT4"])
def test_finn_pipeline_parametrized(channels, pe, input_dt, output_dt):
    """Parametrized FINN pipeline test."""
    if not FINN_AVAILABLE:
        pytest.skip("FINN transformations not available")
    
    # Skip invalid configurations
    if channels % pe != 0:
        pytest.skip("Invalid PE configuration")
    
    config = {
        "channels": channels,
        "pe": pe,
        "input_dt": input_dt,
        "output_dt": output_dt
    }
    
    tester = FINNPipelineTester()
    assert tester.test_complete_pipeline(config), f"FINN pipeline test failed for config {config}"


if __name__ == "__main__":
    sys.exit(main())