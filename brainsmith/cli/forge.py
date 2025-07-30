#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Forge CLI for Brainsmith DSE"""

import sys
import argparse
from pathlib import Path
from brainsmith.core import explore_design_space, DSETree, BlueprintParser, DSETreeBuilder
from brainsmith.core.plugins import list_all_steps, list_all_kernels


def cmd_run(args):
    """Run DSE exploration"""
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if not Path(args.blueprint).exists():
        print(f"Error: Blueprint not found: {args.blueprint}")
        sys.exit(1)
    
    try:
        results = explore_design_space(args.model, args.blueprint, args.output)
        stats = results.stats
        print(f"\n✓ DSE Complete: {stats['successful']}/{stats['total']} succeeded")
        print(f"  Time: {results.total_time:.1f}s")
        print(f"  Output: {args.output or 'build/dse_*'}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def cmd_tree(args):
    """Visualize DSE tree"""
    try:
        parser = BlueprintParser()
        design_space, config = parser.parse(args.blueprint, "dummy.onnx")
        builder = DSETreeBuilder()
        tree = builder.build_tree(design_space, config)
        
        print(f"\nDSE Tree for {args.blueprint}:")
        print("="*50)
        tree.print_tree()
        
        if args.stats:
            stats = tree.get_statistics()
            print(f"\nStatistics:")
            print(f"  Paths: {stats['total_paths']}")
            print(f"  Segments: {stats['total_segments']}")
            print(f"  Efficiency: {stats['segment_efficiency']}%")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def cmd_list(args):
    """List available components"""
    if args.type == 'steps':
        steps = list_all_steps()
        print("\nAvailable Steps:")
        for step in sorted(steps):
            print(f"  - {step}")
    elif args.type == 'kernels':
        kernels = list_all_kernels()
        print("\nAvailable Kernels:")
        for kernel, backends in sorted(kernels.items()):
            print(f"  - {kernel}: {', '.join(backends)}")


def main():
    parser = argparse.ArgumentParser(
        description='Forge - Brainsmith FPGA accelerator synthesis'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Run command (default if no subcommand)
    run_parser = subparsers.add_parser('run', help='Run DSE exploration')
    run_parser.add_argument('model', help='ONNX model path')
    run_parser.add_argument('blueprint', help='Blueprint YAML path')
    run_parser.add_argument('-o', '--output', help='Output directory')
    run_parser.set_defaults(func=cmd_run)
    
    # Tree visualization
    tree_parser = subparsers.add_parser('tree', help='Show DSE tree')
    tree_parser.add_argument('blueprint', help='Blueprint YAML path')
    tree_parser.add_argument('--stats', action='store_true', help='Show statistics')
    tree_parser.set_defaults(func=cmd_tree)
    
    # List components
    list_parser = subparsers.add_parser('list', help='List available components')
    list_parser.add_argument('type', choices=['steps', 'kernels'])
    list_parser.set_defaults(func=cmd_list)
    
    # Handle no subcommand = run
    args = parser.parse_args()
    if args.command is None:
        # Allow "forge model.onnx blueprint.yaml" syntax
        if len(sys.argv) >= 3 and sys.argv[1].endswith('.onnx'):
            args.model = sys.argv[1]
            args.blueprint = sys.argv[2]
            args.output = sys.argv[3] if len(sys.argv) > 3 else None
            args.func = cmd_run
        else:
            parser.print_help()
            sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()