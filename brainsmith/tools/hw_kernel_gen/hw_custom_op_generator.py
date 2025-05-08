# filepath: /home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/hw_custom_op_generator.py
import datetime
import os
from jinja2 import Environment, FileSystemLoader

# Attempt to import HWKernel and related data structures.
# These are expected to be defined in brainsmith.tools.hw_kernel_gen.rtl_parser.data
try:
    from brainsmith.tools.hw_kernel_gen.rtl_parser.data import HWKernel, Parameter, Interface
except ImportError:
    # Fallback for cases where this module might be run standalone or in a different context
    # Define dummy classes if the real ones can\'t be imported, for basic script integrity.
    print("Warning: Could not import HWKernel data classes. Using dummy definitions for HWCustomOpGenerator.")
    class HWKernel: pass
    class Parameter: pass
    class Interface: pass

class HWCustomOpGenerator:
    """
    Generates the HWCustomOp Python class string for a given hardware kernel
    using a Jinja2 template.
    """
    def __init__(self, template_dir: str = None):
        if template_dir is None:
            # Default template directory relative to this file
            template_dir = os.path.join(os.path.dirname(__file__), "templates")
        
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _prepare_context(self, hw_kernel: HWKernel, parsed_compiler_data: dict, class_name_suffix: str = "HWCustomOp") -> dict:
        """
        Prepares the context dictionary for rendering the Jinja2 template.
        """
        context = {}
        context["generation_timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        kernel_name_capitalized = hw_kernel.name.capitalize().replace(\'_\', \'\')
        context["class_name"] = f"{kernel_name_capitalized}{class_name_suffix}"
        context["hw_kernel"] = hw_kernel  # Pass the whole object for detailed template access
        
        # Default description, can be overridden if specified in compiler_data or elsewhere
        context["description"] = f"Automatically generated HWCustomOp for {hw_kernel.name}."

        # Initialize method dictionaries - these will be populated based on rules
        user_methods = {}
        derived_methods = {}
        static_methods = {}
        placeholder_methods = {}
        user_imports = "" # For imports specified in compiler_data.py

        # --- Populate user_methods ---
        # Extract methods from parsed_compiler_data.
        # This needs a strategy: e.g., look for a specific class name in parsed_compiler_data,
        # or take all methods from the first class found, or a pre-defined list.
        # For now, assume parsed_compiler_data might contain a dict of {method_name: source_code}
        # that are intended to be user overrides.
        if parsed_compiler_data:
            # Example: if compiler data has a class named 'MyKernelCompilerData'
            # target_class_name = f"{kernel_name_capitalized}CompilerData" # Or a fixed name
            # if target_class_name in parsed_compiler_data.get("class_methods", {}):
            # user_methods.update(parsed_compiler_data["class_methods"][target_class_name])
            
            # A simpler approach for now: if parsed_compiler_data itself is the dict of methods
            if isinstance(parsed_compiler_data.get("user_methods"), dict): # Placeholder key
                 user_methods.update(parsed_compiler_data["user_methods"])


        # --- Populate derived_methods (based on HWKernel and HKG_Python_Function_Mapping.md) ---
        # Example:
        # derived_methods["get_input_datatype"] = self._generate_get_input_datatype_method(hw_kernel)
        # derived_methods["get_output_datatype"] = self._generate_get_output_datatype_method(hw_kernel)
        # derived_methods["get_verilog_top_module_intf_names"] = self._generate_get_verilog_top_module_intf_names(hw_kernel)


        # --- Populate static_methods (based on HKG_Python_Function_Mapping.md) ---
        # Example:
        # static_methods["get_number_output_values"] = (
        #     "def get_number_output_values(self):\\n"
        #     "    # return np.prod(self.get_folded_output_shape()[:-1])\\n"
        #     "    raise NotImplementedError(\\"Consider implementing based on folded_output_shape\\")"
        # )
        
        # --- Populate placeholder_methods ---
        # For methods that are expected to be implemented by the user eventually.
        # Refer to HKG_Python_Function_Mapping.md for "User implements"
        # Example:
        # placeholder_methods["bram_estimation"] = (
        #     "def bram_estimation(self):\\n"
        #     "    # FPGA resource estimation for BRAMs\\n"
        #     "    raise NotImplementedError(\\"bram_estimation must be implemented by the user or automated.\\")"
        # )
        # placeholder_methods["lut_estimation"] = (
        #     "def lut_estimation(self):\\n"
        #     "    # FPGA resource estimation for LUTs\\n"
        #     "    raise NotImplementedError(\\"lut_estimation must be implemented by the user or automated.\\")"
        # )
        # ... and other estimation functions, generate_params, derive_characteristic_fxns etc.


        context["user_methods"] = user_methods
        context["derived_methods"] = derived_methods
        context["static_methods"] = static_methods
        context["placeholder_methods"] = placeholder_methods
        context["user_imports"] = user_imports

        return context

    def generate(self, hw_kernel: HWKernel, parsed_compiler_data: dict, template_name: str = "hw_custom_op.py.j2") -> str:
        """
        Generates the Python code for the HWCustomOp class.

        Args:
            hw_kernel: An HWKernel object containing parsed RTL information.
            parsed_compiler_data: A dictionary containing data extracted from the
                                  user\'s compiler_data.py file by CompilerDataParser.
                                  The structure of this dict needs to be aligned with
                                  how CompilerDataParser provides it and how this generator
                                  expects to consume it (e.g., specific keys for methods, imports).
            template_name: The name of the Jinja2 template file to use.

        Returns:
            A string containing the generated Python code.
        """
        if not isinstance(hw_kernel, HWKernel):
            raise TypeError("hw_kernel must be an instance of HWKernel or a compatible type.")

        context = self._prepare_context(hw_kernel, parsed_compiler_data)
        template = self.template_env.get_template(template_name)
        generated_code = template.render(context)
        
        # Post-processing: could run a formatter like Black or autopep8 if desired
        # For now, return raw generated code
        return generated_code

    # Placeholder for helper methods to generate specific function bodies
    # def _generate_get_input_datatype_method(self, hw_kernel: HWKernel) -> str:
    #     # Logic to generate the get_input_datatype(self, ind) method string
    #     # based on hw_kernel.interfaces and HKG_Python_Function_Mapping.md
    #     pass 
