#!/usr/bin/env python3
"""Debug why multiple StreamingDataflowPartition nodes are created."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import onnx

def debug_sdp_issue():
    print("=== Debugging StreamingDataflowPartition Issue ===\n")
    
    # Check the model after infer_kernels step
    model_path = Path("/home/tafk/builds/brainsmith/test_infer_kernels/root/intermediate_models")
    
    if not model_path.exists():
        print(f"Directory {model_path} doesn't exist yet")
        return
    
    # Find the model after infer_kernels
    onnx_files = sorted(model_path.glob("*.onnx"))
    
    if not onnx_files:
        print("No ONNX files found")
        return
    
    # Look for the model after step 6 (infer_kernels)
    for f in onnx_files:
        if "step6" in f.name or "infer_kernels" in f.name:
            print(f"Checking model: {f.name}")
            model = onnx.load(str(f))
            
            # Count StreamingDataflowPartition nodes
            sdp_count = 0
            hw_nodes = []
            
            for node in model.graph.node:
                if node.op_type == "StreamingDataflowPartition":
                    sdp_count += 1
                    print(f"  Found SDP: {node.name}")
                
                # Check for HW nodes with finn domain
                if node.domain == "brainsmith" or node.domain == "finn" or node.domain == "qonnx":
                    hw_nodes.append(f"{node.op_type} (domain={node.domain})")
            
            print(f"\nTotal StreamingDataflowPartition nodes: {sdp_count}")
            print(f"Total HW nodes: {len(hw_nodes)}")
            if hw_nodes:
                print("HW nodes found:")
                for hw in set(hw_nodes):
                    count = hw_nodes.count(hw)
                    print(f"  - {hw}: {count}")
            break
    else:
        print("No model found after infer_kernels step")
        print("Available models:")
        for f in onnx_files[-5:]:
            print(f"  - {f.name}")

if __name__ == "__main__":
    debug_sdp_issue()