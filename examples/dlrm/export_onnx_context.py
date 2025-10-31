#!/usr/bin/env python3
"""
Export ONNX models to a human-readable context format.

This script generates a comprehensive but readable representation of an ONNX model
that can be easily consumed by AI assistants or used for documentation.
"""

import argparse
import sys
from pathlib import Path
from io import StringIO
import onnx
from onnxscript import ir


def truncate_array_repr(text, max_elements=10):
    """Truncate large array representations in the text."""
    # This is a simple heuristic - could be improved
    return text


def export_onnx_context(onnx_path, output_path=None, max_array_size=20, verbose=False):
    """
    Export ONNX model to a readable context file.

    Args:
        onnx_path: Path to the ONNX model file
        output_path: Path to save the output (default: {model_name}_context.txt)
        max_array_size: Maximum number of elements to show in arrays
        verbose: Print progress messages
    """
    onnx_path = Path(onnx_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # Determine output path
    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_context.txt"
    else:
        output_path = Path(output_path)

    if verbose:
        print(f"Loading ONNX model from: {onnx_path}")

    # Load and deserialize the model
    proto = onnx.load(str(onnx_path))
    ir_model = ir.serde.deserialize_model(proto)

    if verbose:
        print(f"Writing context to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"ONNX Model Context: {onnx_path.name}\n")
        f.write("=" * 80 + "\n\n")

        # 1. High-level summary
        f.write("# Model Metadata\n")
        f.write("-" * 80 + "\n")
        f.write(f"IR Version:      {ir_model.ir_version}\n")
        f.write(f"Opset Imports:   {ir_model.opset_imports}\n")
        f.write(f"Producer Name:   {ir_model.producer_name}\n")
        f.write(f"Producer Version: {ir_model.producer_version}\n")
        f.write(f"Domain:          {ir_model.domain}\n")
        f.write(f"Model Version:   {ir_model.model_version}\n")
        f.write(f"Graph Name:      {ir_model.graph.name}\n")
        f.write("\n")

        # 2. Inputs
        f.write("# Graph Inputs\n")
        f.write("-" * 80 + "\n")
        if ir_model.graph.inputs:
            for i, inp in enumerate(ir_model.graph.inputs):
                f.write(f"{i+1}. {inp.name}\n")
                f.write(f"   Type: {inp.type}\n")
                f.write(f"   Shape: {inp.shape if hasattr(inp, 'shape') else 'N/A'}\n")
        else:
            f.write("No inputs\n")
        f.write("\n")

        # 3. Outputs
        f.write("# Graph Outputs\n")
        f.write("-" * 80 + "\n")
        if ir_model.graph.outputs:
            for i, out in enumerate(ir_model.graph.outputs):
                f.write(f"{i+1}. {out.name}\n")
                f.write(f"   Type: {out.type}\n")
                f.write(f"   Shape: {out.shape if hasattr(out, 'shape') else 'N/A'}\n")
        else:
            f.write("No outputs\n")
        f.write("\n")

        # 4. Initializers summary
        f.write("# Initializers (Weights & Constants)\n")
        f.write("-" * 80 + "\n")
        initializers = list(ir_model.graph.initializers.values())
        if initializers:
            f.write(f"Total: {len(initializers)}\n\n")
            for init in initializers:
                f.write(f"  • {init.name}: {init.type}")
                if hasattr(init, 'shape') and init.shape:
                    total_elements = 1
                    for dim in init.shape.dims:
                        if dim:
                            total_elements *= dim
                    f.write(f" [{total_elements} elements]")
                f.write("\n")
        else:
            f.write("No initializers\n")
        f.write("\n")

        # 5. Operation statistics
        f.write("# Operation Statistics\n")
        f.write("-" * 80 + "\n")
        op_counts = {}
        total_nodes = 0
        for node in ir_model.graph:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
            total_nodes += 1

        f.write(f"Total Nodes: {total_nodes}\n\n")
        f.write("Operation Type Distribution:\n")
        for op, count in sorted(op_counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"  {op:30s} : {count:4d}\n")
        f.write("\n")

        # 6. Node connectivity summary
        f.write("# Node Connectivity\n")
        f.write("-" * 80 + "\n")
        f.write("Input -> Consumers:\n")
        for inp in ir_model.graph.inputs:
            consumers = list(inp.consumers())
            f.write(f"  {inp.name} -> {len(consumers)} consumer(s)")
            if consumers:
                consumer_names = [c.name for c in consumers[:5]]
                if len(consumers) > 5:
                    consumer_names.append(f"... and {len(consumers)-5} more")
                f.write(f": {', '.join(consumer_names)}")
            f.write("\n")
        f.write("\n")

        # 7. Full IR display
        f.write("# Complete Graph Structure (ONNX Script IR Format)\n")
        f.write("=" * 80 + "\n")
        f.write("This section shows the complete computational graph in ONNX Script format.\n")
        f.write("Each node shows: operation type, inputs, outputs, and attributes.\n")
        f.write("=" * 80 + "\n\n")

        # Capture the display output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        ir_model.display()
        full_display = sys.stdout.getvalue()
        sys.stdout = old_stdout

        f.write(full_display)
        f.write("\n\n")

        # 8. Footer
        f.write("=" * 80 + "\n")
        f.write("End of ONNX Model Context\n")
        f.write("=" * 80 + "\n")

    if verbose:
        print(f"✓ Context file created successfully: {output_path}")
        print(f"  Total nodes: {total_nodes}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export ONNX model to readable context format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default output name
  python export_onnx_context.py model.onnx

  # Specify output file
  python export_onnx_context.py model.onnx -o context.txt

  # Verbose output
  python export_onnx_context.py model.onnx -v
        """
    )

    parser.add_argument(
        "onnx_file",
        help="Path to the ONNX model file"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: {model_name}_context.txt)",
        default=None
    )

    parser.add_argument(
        "--max-array-size",
        type=int,
        default=20,
        help="Maximum number of array elements to display (default: 20)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    try:
        output_path = export_onnx_context(
            args.onnx_file,
            args.output,
            args.max_array_size,
            args.verbose
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
