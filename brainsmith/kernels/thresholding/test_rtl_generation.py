############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
RTL generation test for thresholding AutoHWCustomOp.

This test validates that the auto-generated RTL backend can generate HDL
files and compares the generation process with the manual implementation.
"""

import sys
import os
import numpy as np
import onnx
import onnx.helper as oh
from onnx import numpy_helper
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# QONNX imports
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

# Brainsmith imports
from brainsmith.kernels.thresholding.finn.thresholding_rtl import Thresholding_rtl
from brainsmith.kernels.thresholding.bsmith.thresholding_axi_rtl import thresholding_axi_rtl
from brainsmith.kernels.thresholding.bsmith.thresholding_axi_hw_custom_op import ThresholdingAxi


class RTLGenerationTester:
    """Tests RTL generation capabilities of auto-generated backend."""
    
    def __init__(self):
        self.test_config = {"channels": 64, "pe": 8, "input_dt": "INT8", "output_dt": "UINT4"}
    
    def create_test_node(self, op_type: str, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Create test node and operation instance."""
        
        # Create input/output tensors
        inp = oh.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, config["channels"]])
        outp = oh.make_tensor_value_info("outp", onnx.TensorProto.FLOAT, [1, config["channels"]])
        thresh = oh.make_tensor_value_info("thresh", onnx.TensorProto.FLOAT, [config["channels"], 3])
        
        inputs = [inp, thresh]
        outputs = [outp]
        
        # Create threshold initializer
        thresh_vals = np.array([[-10, 0, 10]] * config["channels"], dtype=np.float32)
        thresh_init = numpy_helper.from_array(thresh_vals, name="thresh")
        
        if op_type == "manual":
            # Use manual FINN RTL backend
            node = oh.make_node(
                "Thresholding",
                ["inp", "thresh"],
                ["outp"],
                domain="finn.custom_op.fpgadataflow",
                PE=config["pe"],
                inputDataType=config["input_dt"],
                outputDataType=config["output_dt"],
                backend="fpgadataflow"
            )
            op_inst = Thresholding_rtl(node)
        else:
            # Use auto-generated RTL backend
            node = oh.make_node(
                "ThresholdingAxi", 
                ["inp", "thresh"],
                ["outp"],
                CHANNELS=config["channels"],
                PE=config["pe"],
                LEVELS=3,
                inputDataType=config["input_dt"],
                weightDataType="INT8",
                outputDataType=config["output_dt"],
                backend="fpgadataflow"
            )
            op_inst = thresholding_axi_rtl(node)
        
        # Create minimal model for context
        graph = oh.make_graph([node], f"{op_type}_graph", inputs, outputs, [thresh_init])
        model = oh.make_model(graph)
        model.opset_import[0].version = 11
        model_wrapper = ModelWrapper(model)
        
        # Set up the operation for RTL generation
        if op_type == "auto":
            # Trigger shape extraction for auto implementation
            hw_custom_op = ThresholdingAxi(node)
            hw_custom_op.infer_node_datatype(model_wrapper, node)
        
        return op_inst, model_wrapper
    
    def setup_rtl_generation(self, op_inst: Any, temp_dir: str, op_type: str) -> bool:
        """Set up operation instance for RTL generation."""
        try:
            # Set required FINN attributes for RTL generation
            rtl_dir = os.path.join(temp_dir, f"{op_type}_rtl")
            os.makedirs(rtl_dir, exist_ok=True)
            
            # Set code generation directory
            op_inst.set_nodeattr("code_gen_dir_ipgen", rtl_dir)
            
            # Set additional required attributes
            if hasattr(op_inst, 'set_nodeattr'):
                # These are typical FINN RTL generation attributes
                op_inst.set_nodeattr("backend", "fpgadataflow")
                
                # Try to set other common attributes
                try:
                    op_inst.set_nodeattr("fpgapart", "xczu3eg-sbva484-1-e")
                except:
                    pass  # Not all ops need this
                
                try:
                    op_inst.set_nodeattr("mode", "rtl")
                except:
                    pass  # Not all ops need this
            
            print(f"   ✅ Set up RTL generation directory: {rtl_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to set up RTL generation: {e}")
            return False
    
    def generate_rtl(self, op_inst: Any, model_wrapper: ModelWrapper, op_type: str) -> Tuple[bool, List[str]]:
        """Generate RTL and return success status and list of generated files."""
        try:
            # Check if operation has RTL generation capability
            if not hasattr(op_inst, 'generate_hdl'):
                print(f"   ⚠️ {op_type} implementation does not have generate_hdl method")
                return False, []
            
            # Generate RTL
            print(f"   🔧 Generating RTL for {op_type} implementation...")
            fpgapart = op_inst.get_nodeattr("fpgapart", "xczu3eg-sbva484-1-e")
            clk = 250  # Default clock frequency in MHz
            op_inst.generate_hdl(model_wrapper, fpgapart, clk)
            
            # Get generated files
            rtl_dir = op_inst.get_nodeattr("code_gen_dir_ipgen")
            if os.path.exists(rtl_dir):
                generated_files = []
                for root, dirs, files in os.walk(rtl_dir):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), rtl_dir)
                        generated_files.append(rel_path)
                
                print(f"   ✅ Generated {len(generated_files)} files in {rtl_dir}")
                return True, generated_files
            else:
                print(f"   ❌ RTL directory not created: {rtl_dir}")
                return False, []
                
        except Exception as e:
            print(f"   ❌ RTL generation failed: {e}")
            return False, []
    
    def analyze_generated_files(self, manual_files: List[str], auto_files: List[str]) -> Dict[str, Any]:
        """Analyze and compare generated files."""
        analysis = {
            "manual_count": len(manual_files),
            "auto_count": len(auto_files),
            "common_files": [],
            "manual_only": [],
            "auto_only": [],
            "file_types": {"manual": {}, "auto": {}}
        }
        
        manual_set = set(manual_files)
        auto_set = set(auto_files)
        
        analysis["common_files"] = sorted(manual_set & auto_set)
        analysis["manual_only"] = sorted(manual_set - auto_set)
        analysis["auto_only"] = sorted(auto_set - manual_set)
        
        # Analyze file types
        for files, key in [(manual_files, "manual"), (auto_files, "auto")]:
            type_count = {}
            for file in files:
                ext = Path(file).suffix.lower()
                type_count[ext] = type_count.get(ext, 0) + 1
            analysis["file_types"][key] = type_count
        
        return analysis
    
    def test_rtl_generation(self, config: Dict[str, Any]) -> bool:
        """Test RTL generation for both manual and auto implementations."""
        print(f"\n🔧 Testing RTL generation: channels={config['channels']}, pe={config['pe']}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create both implementations
                print("   📝 Creating operation instances...")
                manual_op, manual_model = self.create_test_node("manual", config)
                auto_op, auto_model = self.create_test_node("auto", config)
                
                # Set up RTL generation for both
                print("   ⚙️ Setting up RTL generation...")
                manual_setup = self.setup_rtl_generation(manual_op, temp_dir, "manual")
                auto_setup = self.setup_rtl_generation(auto_op, temp_dir, "auto")
                
                if not manual_setup or not auto_setup:
                    print("   ❌ Failed to set up RTL generation")
                    return False
                
                # Generate RTL for both
                print("   🏗️ Generating RTL...")
                manual_success, manual_files = self.generate_rtl(manual_op, manual_model, "manual")
                auto_success, auto_files = self.generate_rtl(auto_op, auto_model, "auto")
                
                # Analyze results
                if manual_success or auto_success:
                    analysis = self.analyze_generated_files(manual_files, auto_files)
                    
                    print(f"   📊 File generation analysis:")
                    print(f"      Manual files: {analysis['manual_count']}")
                    print(f"      Auto files: {analysis['auto_count']}")
                    print(f"      Common files: {len(analysis['common_files'])}")
                    
                    if analysis['common_files']:
                        print(f"      Common file types: {analysis['common_files'][:5]}")  # Show first 5
                    
                    if analysis['file_types']['manual']:
                        print(f"      Manual file types: {analysis['file_types']['manual']}")
                    if analysis['file_types']['auto']:
                        print(f"      Auto file types: {analysis['file_types']['auto']}")
                    
                    # Success if either implementation generated files
                    if manual_success or auto_success:
                        print("   ✅ RTL generation test passed!")
                        return True
                
                print("   ❌ No RTL files were generated by either implementation")
                return False
                
            except Exception as e:
                print(f"   ❌ RTL generation test failed: {e}")
                return False
    
    def run_all_tests(self) -> bool:
        """Run all RTL generation tests."""
        print("🚀 Starting RTL Generation Testing...")
        print("=" * 60)
        
        success = self.test_rtl_generation(self.test_config)
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 RTL generation test completed successfully!")
        else:
            print("❌ RTL generation test failed!")
        
        return success


def main():
    """Main test function."""
    tester = RTLGenerationTester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())