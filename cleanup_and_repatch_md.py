import argparse
import onnx
from onnx import StringStringEntryProto
import subprocess
from qonnx.util.cleanup import cleanup
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Add metadata to the first node of an ONNX model.")
    parser.add_argument("input_file", type=str, help="Path to the input ONNX file")
    parser.add_argument("-o", "--output_file", type=str, default="cleanedup_with_metadata.onnx", help="Path to the output ONNX file")

    # Parse arguments
    args = parser.parse_args()

    # Attempt a cleanup of previous runs (This could all be a lot nicer...)
    if os.path.exists("./_tmp_onnxsim.onnx"):
        os.remove("./_tmp_onnxsim.onnx")
    if os.path.exists("./_tmp_onnxsim_with_md.onnx"):
        os.remove("./_tmp_onnxsim_with_md.onnx")

    # Attempt an initial cleanup telling onnxsim to skip optimisation (errors out otherwise)
    command = ['python', '-m', 'onnxsim', args.input_file, '_tmp_onnxsim.onnx', '--skip-optimization']
    #command = ['python', '-m', 'onnxsim', args.input_file, '_tmp_onnxsim.onnx']

    try:
        # Run the command with error checking
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Standard Output:", result.stdout)

    except subprocess.CalledProcessError as e:
        # Print the error details and exit
        print("An error occurred while executing the command:")
        print(f"Return Code: {e.returncode}")
        print(f"Standard Output: {e.stdout}")
        print(f"Standard Error: {e.stderr}")
        raise RuntimeError(f"Simplifying and reapplying metadata to {args.input_file} failed")

    # Apply QONNX cleanup also

    # load in the inital model and extract the metadata
    og_mod = onnx.load(args.input_file)
    metadata = {}
    for node in og_mod.graph.node:
        md = {}
        for prop in node.metadata_props:
            md[prop.key] = prop.value
        metadata[node.name] = md

    # Load in the clenedup model and apply the metadata per node
    #cleaned_mod = onnx.load("_tmp_qonnx_cleanup.onnx")
    cleaned_mod = onnx.load("_tmp_onnxsim.onnx")

    for node in cleaned_mod.graph.node:
        if node.name in metadata:
            md_props = metadata[node.name]
            for key,value in md_props.items():
                new_md = StringStringEntryProto(key=key,value=value)
                node.metadata_props.append(new_md)

    onnx.save(cleaned_mod, "./_tmp_onnxsim_with_md.onnx",save_as_external_data=True, all_tensors_to_one_file=True, location="_tmp_onnxsim_with_md.onnx.data")
    cleanup(in_file="./_tmp_onnxsim_with_md.onnx", out_file=args.output_file)

    print(f"Saved the cleaned model with the metadata applied to it in {args.output_file}")


if __name__ == "__main__":
    main()
