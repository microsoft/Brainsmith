#!/usr/bin/env python3
"""
Extract FIFO configuration from a FINN model that has gone through step_set_fifo_depths.
This can be used to create a complete folding config that includes FIFO sizes.
"""

import json
import sys
import argparse
from pathlib import Path

try:
    import onnx
    from qonnx.core.modelwrapper import ModelWrapper
except ImportError:
    print("Error: qonnx package required. Install with: pip install qonnx")
    sys.exit(1)


def extract_fifo_config(model_path: str, base_folding_config: str = None) -> dict:
    """
    Extract FIFO depths from a model and optionally merge with existing folding config.
    
    Args:
        model_path: Path to ONNX model with FIFO depths set
        base_folding_config: Optional path to existing folding config to merge with
        
    Returns:
        Complete folding configuration with FIFO depths
    """
    print(f"Loading model from: {model_path}")
    model = ModelWrapper(model_path)
    
    # Load base config if provided
    if base_folding_config:
        print(f"Loading base folding config from: {base_folding_config}")
        with open(base_folding_config, 'r') as f:
            config = json.load(f)
    else:
        config = {"Defaults": {}}
    
    # Extract FIFO depths from model
    fifo_layers = []
    for node in model.graph.node:
        if node.op_type == "StreamingFIFO":
            # Get node name and depth
            node_name = node.name
            depth = None
            
            # Look for depth attribute
            for attr in node.attribute:
                if attr.name == "depth":
                    depth = attr.i
                    break
            
            if depth:
                # FINN uses the previous node's name for FIFO config
                # Find the producer of this FIFO's input
                input_tensor = node.input[0]
                producer = model.find_producer(input_tensor)
                
                if producer:
                    producer_name = producer.name
                    if producer_name not in config:
                        config[producer_name] = {}
                    
                    # Determine if this is input or output FIFO
                    # This is a simplified heuristic - may need adjustment
                    if "in" in node_name.lower() or len(fifo_layers) == 0:
                        if "inFIFODepths" not in config[producer_name]:
                            config[producer_name]["inFIFODepths"] = []
                        config[producer_name]["inFIFODepths"].append(depth)
                    else:
                        config[producer_name]["outFIFODepth"] = depth
                    
                    fifo_layers.append(f"{producer_name}: {node_name} = {depth}")
                    print(f"  Found FIFO: {node_name} with depth {depth} (attached to {producer_name})")
    
    print(f"\nExtracted FIFO configuration for {len(fifo_layers)} FIFOs")
    
    # Alternative approach: Look for annotated FIFO depths in layer nodes
    for node in model.graph.node:
        node_name = node.name
        
        # Check for FIFO depth annotations
        for attr in node.attribute:
            if attr.name == "inFIFODepths":
                if node_name not in config:
                    config[node_name] = {}
                # Parse the attribute value (might be stored as string)
                try:
                    depths = json.loads(attr.s.decode() if isinstance(attr.s, bytes) else attr.s)
                    config[node_name]["inFIFODepths"] = depths
                    print(f"  Found annotated inFIFODepths for {node_name}: {depths}")
                except:
                    pass
                    
            elif attr.name == "outFIFODepth":
                if node_name not in config:
                    config[node_name] = {}
                config[node_name]["outFIFODepth"] = attr.i
                print(f"  Found annotated outFIFODepth for {node_name}: {attr.i}")
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Extract FIFO configuration from FINN model")
    parser.add_argument("model_path", help="Path to ONNX model with FIFO depths")
    parser.add_argument("-b", "--base-config", help="Base folding config to merge with")
    parser.add_argument("-o", "--output", help="Output path for complete config", 
                       default="complete_folding_config.json")
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if args.base_config and not Path(args.base_config).exists():
        print(f"Error: Base config file not found: {args.base_config}")
        sys.exit(1)
    
    try:
        config = extract_fifo_config(args.model_path, args.base_config)
        
        # Save the complete configuration
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nComplete folding config saved to: {args.output}")
        
        # Summary
        layers_with_fifo = sum(1 for v in config.values() 
                              if isinstance(v, dict) and 
                              ('inFIFODepths' in v or 'outFIFODepth' in v))
        print(f"Total layers in config: {len(config) - 1}")  # -1 for Defaults
        print(f"Layers with FIFO config: {layers_with_fifo}")
        
    except Exception as e:
        print(f"Error processing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()