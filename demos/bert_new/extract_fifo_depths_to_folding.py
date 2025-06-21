#!/usr/bin/env python3
"""
Extract FIFO depths from a FINN model and add them to an existing folding configuration.
This creates a complete folding config that can skip the expensive FIFO sizing step.
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from qonnx.core.modelwrapper import ModelWrapper
except ImportError:
    print("Error: qonnx package required. Install with: pip install qonnx")
    sys.exit(1)


def extract_fifo_depths_for_folding(model_path: str, folding_config_path: str) -> dict:
    """
    Extract FIFO depths and add them to folding configuration.
    
    FINN's folding config format expects:
    - "inFIFODepths": list of depths for input FIFOs
    - "outFIFODepth": single depth for output FIFO
    
    Args:
        model_path: Path to ONNX model after step_set_fifo_depths
        folding_config_path: Path to existing folding config (PE/SIMD params)
        
    Returns:
        Complete folding configuration with FIFO depths added
    """
    print(f"Loading model from: {model_path}")
    model = ModelWrapper(model_path)
    
    print(f"Loading folding config from: {folding_config_path}")
    with open(folding_config_path, 'r') as f:
        config = json.load(f)
    
    # Build a map of what connects to what
    tensor_to_consumer = {}
    tensor_to_producer = {}
    
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                tensor_to_consumer[inp] = node
        for out in node.output:
            if out:
                tensor_to_producer[out] = node
    
    # Find FIFOs and their connections
    fifo_info = []
    for node in model.graph.node:
        if node.op_type.startswith("StreamingFIFO"):
            depth = None
            for attr in node.attribute:
                if attr.name == "depth":
                    depth = attr.i
                    break
            
            if depth and node.input[0] in tensor_to_producer:
                producer = tensor_to_producer[node.input[0]]
                consumer = None
                if node.output[0] in tensor_to_consumer:
                    consumer = tensor_to_consumer[node.output[0]]
                
                fifo_info.append({
                    'fifo_name': node.name,
                    'depth': depth,
                    'producer': producer.name,
                    'consumer': consumer.name if consumer else None,
                    'producer_type': producer.op_type,
                    'consumer_type': consumer.op_type if consumer else None
                })
    
    print(f"\nFound {len(fifo_info)} FIFOs in model")
    
    # Group FIFOs by their associated compute layers
    layer_fifos = defaultdict(lambda: {'in': [], 'out': None})
    
    for fifo in fifo_info:
        # Determine if this FIFO is input or output for the compute layer
        producer_type = fifo['producer_type']
        consumer_type = fifo['consumer_type']
        
        # If producer is a compute layer (MVAU, Thresholding, etc), this is its output FIFO
        if any(compute in producer_type for compute in ['MVAU', 'Thresholding', 'MVU', 'Conv', 'Pool']):
            if fifo['producer'] in config:
                layer_fifos[fifo['producer']]['out'] = fifo['depth']
                print(f"  {fifo['producer']} output FIFO: depth={fifo['depth']}")
        
        # If consumer is a compute layer, this is its input FIFO
        if consumer_type and any(compute in consumer_type for compute in ['MVAU', 'Thresholding', 'MVU', 'Conv', 'Pool']):
            if fifo['consumer'] in config:
                layer_fifos[fifo['consumer']]['in'].append(fifo['depth'])
                print(f"  {fifo['consumer']} input FIFO: depth={fifo['depth']}")
    
    # Add FIFO depths to config
    added_count = 0
    for layer_name, fifos in layer_fifos.items():
        if layer_name in config:
            if fifos['in']:
                config[layer_name]['inFIFODepths'] = fifos['in']
                added_count += 1
            if fifos['out'] is not None:
                config[layer_name]['outFIFODepth'] = fifos['out']
                added_count += 1
    
    print(f"\nAdded FIFO depths to {added_count} layer configurations")
    
    # Handle special cases for other layer types that might need FIFO config
    for node in model.graph.node:
        if node.name in config and node.name not in layer_fifos:
            # Check if this node has FIFO depth attributes directly
            for attr in node.attribute:
                if attr.name == "inFIFODepths":
                    try:
                        depths = json.loads(attr.s.decode() if isinstance(attr.s, bytes) else attr.s)
                        config[node.name]["inFIFODepths"] = depths
                        print(f"  Added direct inFIFODepths to {node.name}: {depths}")
                    except:
                        pass
                elif attr.name == "outFIFODepth":
                    config[node.name]["outFIFODepth"] = attr.i
                    print(f"  Added direct outFIFODepth to {node.name}: {attr.i}")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Extract FIFO depths from FINN model and add to folding config"
    )
    parser.add_argument(
        "model_path", 
        help="Path to ONNX model after step_set_fifo_depths"
    )
    parser.add_argument(
        "folding_config", 
        help="Existing folding config with PE/SIMD parameters"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output path for complete config", 
        default="complete_folding_with_fifos.json"
    )
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not Path(args.folding_config).exists():
        print(f"Error: Folding config not found: {args.folding_config}")
        sys.exit(1)
    
    try:
        config = extract_fifo_depths_for_folding(args.model_path, args.folding_config)
        
        # Save the complete configuration
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nComplete folding config with FIFOs saved to: {args.output}")
        
        # Summary statistics
        total_layers = len(config) - 1  # -1 for Defaults
        layers_with_fifo = sum(
            1 for name, params in config.items() 
            if name != "Defaults" and isinstance(params, dict) and 
            ('inFIFODepths' in params or 'outFIFODepth' in params)
        )
        
        print(f"\nSummary:")
        print(f"  Total layers: {total_layers}")
        print(f"  Layers with FIFO config: {layers_with_fifo}")
        
        # Show example
        example_shown = False
        for name, params in config.items():
            if isinstance(params, dict) and 'inFIFODepths' in params and not example_shown:
                print(f"\nExample layer config ({name}):")
                print(f"  PE: {params.get('PE', 'N/A')}")
                print(f"  SIMD: {params.get('SIMD', 'N/A')}")
                print(f"  inFIFODepths: {params['inFIFODepths']}")
                print(f"  outFIFODepth: {params.get('outFIFODepth', 'N/A')}")
                example_shown = True
        
    except Exception as e:
        print(f"Error processing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()