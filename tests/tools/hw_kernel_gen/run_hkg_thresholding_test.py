import os
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.compiler_data_parser import CompilerDataParser
from brainsmith.tools.hw_kernel_gen.generators.hw_custom_op_generator import HWCustomOpGenerator
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Define paths relative to the project root (assuming this script is in the root)
    project_root = os.path.dirname(os.path.abspath(__file__)) # Or adjust if script is elsewhere
    rtl_file_path = os.path.join(project_root, "examples/thresholding/thresholding_axi.sv")
    
    # For compiler_data.py, let's create a dummy one if it doesn't exist for the test
    # In a real scenario, the user provides this.
    compiler_data_dir = os.path.join(project_root, "examples/thresholding/")
    compiler_data_file_path = os.path.join(compiler_data_dir, "compiler_data.py")

    if not os.path.exists(compiler_data_file_path):
        logger.info(f"Creating a dummy compiler_data.py at {compiler_data_file_path} for testing.")
        os.makedirs(compiler_data_dir, exist_ok=True)
        with open(compiler_data_file_path, "w") as f:
            f.write("# Dummy compiler_data.py for thresholding example\n")
            f.write("import numpy as np\n\n")
            f.write("# Example: User might define a helper function or a class\n")
            f.write("# class ThresholdingCompilerFunctions:\n")
            f.write("#     def my_custom_thresholding_logic(self, param1):\n")
            f.write("#         return param1 * 2\n")
    else:
        logger.info(f"Using existing compiler_data.py at {compiler_data_file_path}")


    # 1. Parse RTL
    logger.info(f"Parsing RTL file: {rtl_file_path}")
    try:
        rtl_parser = RTLParser()
        hw_kernel = rtl_parser.parse(rtl_file_path)
        logger.info(f"Successfully parsed HWKernel: {hw_kernel.name}")
        logger.info(f"Parameters found: {[p.name for p in hw_kernel.parameters]}")
        logger.info(f"Interfaces found: {list(hw_kernel.interfaces.keys())}")
    except Exception as e:
        logger.error(f"Error during RTL parsing: {e}", exc_info=True)
        return

    # 2. Parse Compiler Data Python File
    logger.info(f"Parsing compiler data file: {compiler_data_file_path}")
    try:
        compiler_data_parser = CompilerDataParser(compiler_data_file_path)
        # The parser stores data in its `parsed_data` attribute
        parsed_compiler_data = compiler_data_parser.parsed_data
        logger.info(f"Successfully parsed compiler data. Found imports: {parsed_compiler_data.get('imports_str') is not None}")
        logger.info(f"User functions: {list(parsed_compiler_data.get('functions', {}).keys())}")
        logger.info(f"User class methods: { {k: list(v.keys()) for k,v in parsed_compiler_data.get('class_methods', {}).items()} }")
    except Exception as e:
        logger.error(f"Error during compiler data parsing: {e}", exc_info=True)
        # We can still proceed with generation, it will just lack user overrides
        parsed_compiler_data = {"functions": {}, "class_methods": {}, "imports_str": ""}


    # 3. Generate HWCustomOp Code
    logger.info("Generating HWCustomOp code...")
    try:
        # The generator will look for templates in its relative "templates" folder
        # Ensure brainsmith/tools/hw_kernel_gen/templates/hw_custom_op.py.j2 exists
        op_generator = HWCustomOpGenerator() 
        
        # Specify the suffix for the generated class name, e.g., "HWCustomOp" or "RTLBackend"
        # For HWCustomOp, it's typically just "CustomOp" or the kernel name itself if it's unique
        # Let's use a clear suffix for the test.
        class_name_suffix = "ThresholdingCustomOp" # Or simply "CustomOp"

        generated_code = op_generator.generate(
            hw_kernel=hw_kernel,
            parsed_compiler_data=parsed_compiler_data,
            class_name_suffix=class_name_suffix # This will be appended to the capitalized kernel name
        )
        logger.info("Successfully generated HWCustomOp code.")
        
        # 4. Print or Save Generated Code
        print("\n--- Generated HWCustomOp Code ---")
        print(generated_code)
        print("--- End of Generated Code ---\n")

        output_dir = os.path.join(project_root, "generated_ops")
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"{hw_kernel.name.lower()}_{class_name_suffix.lower()}.py")
        with open(output_file_path, "w") as f:
            f.write(generated_code)
        logger.info(f"Generated code saved to: {output_file_path}")

    except Exception as e:
        logger.error(f"Error during HWCustomOp code generation: {e}", exc_info=True)

if __name__ == "__main__":
    main()