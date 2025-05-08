import datetime
import os
from jinja2 import Environment, FileSystemLoader
import logging
import pprint # Added for pretty printing HWKernel

# Attempt to import HWKernel and related data structures.
try:
    # Assumes rtl_parser is a sibling directory to generators/ when this file is in generators/
    from ..rtl_parser.data import HWKernel, Parameter, Interface, PragmaType, Port, DataType as HWDataType, InterfaceType, Direction
except ImportError:
    print("Warning: Could not import HWKernel data classes. Using dummy definitions for HWCustomOpGenerator.")
    class HWKernel:
        def __init__(self, name="DummyKernel", pe_param_name="PE", idt_param_name="IDT", odt_param_name="ODT", shape_info_param_name="SHAPE_INFO", num_input_streams=1, num_output_streams=1, has_axilite=False, clk_name="ap_clk", rst_name="ap_rst_n"):
            self.name = name
            self.parameters = []
            self.interfaces = {}
            self.pragmas = []
            self.metadata = {}
            # Add attributes expected by new derived methods
            self.pe_param_name = pe_param_name
            self.idt_param_name = idt_param_name
            self.odt_param_name = odt_param_name
            self.shape_info_param_name = shape_info_param_name # e.g., "IFMDim", "OFMDim"
            self.num_input_streams = num_input_streams
            self.num_output_streams = num_output_streams
            self.has_axilite = has_axilite # Based on AXI-Lite interface presence
            self.clk_name = clk_name
            self.rst_name = rst_name
            self.input_axi_stream_names = [f"s_axis_input_{i}" for i in range(num_input_streams)]
            self.output_axi_stream_names = [f"m_axis_output_{i}" for i in range(num_output_streams)]
            self.axilite_name = "s_axilite_control" if has_axilite else None


    class Parameter:
        def __init__(self, name, type=None, default_value=None):
            self.name = name
            self.type = type
            self.default_value = default_value
    class Interface: pass
    class PragmaType: # Dummy PragmaType
        DERIVED_PARAM = "derived_param"
        INPUT_DATATYPE = "input_datatype"
        OUTPUT_DATATYPE = "output_datatype"
        SHAPE_INFO = "shape_info"
        PE = "pe"

    # Dummy enums if full import fails
    class HWDataType:
        UINT8 = "UINT8"
        INT32 = "INT32"
        FLOAT32 = "FLOAT32"
        # Add other types as needed for dummy

    class InterfaceType:
        AXI_STREAM = "AXI_STREAM"
        AXI_LITE = "AXI_LITE"
        GLOBAL_CONTROL = "GLOBAL_CONTROL"

    class Direction:
        IN = "in"
        OUT = "out"


logger = logging.getLogger(__name__)

class HWCustomOpGenerator:
    """
    Generates the HWCustomOp Python class string for a given hardware kernel
    using a Jinja2 template.
    """
    def __init__(self, template_dir: str = None):
        if template_dir is None:
            # Default template directory relative to this file\'s new location
            # (brainsmith/tools/hw_kernel_gen/generators/ -> brainsmith/tools/hw_kernel_gen/templates)
            current_dir = os.path.dirname(__file__)
            template_dir = os.path.join(current_dir, "..", "templates")
        
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        # Ensure the template file exists
        template_path = os.path.join(template_dir, "hw_custom_op.py.j2")
        if not os.path.exists(template_path):
            logger.warning(f"Jinja2 template not found at {template_path}. Generation may fail.")

    def _is_derived_param(self, parameter: Parameter, pragmas: list) -> bool:
        # Helper to check if a parameter is tagged in a derived_param pragma
        for pragma in pragmas:
            # Assuming pragma has a 'type' and 'value' (which could be a dict)
            if hasattr(pragma, 'type') and pragma.type == PragmaType.DERIVED_PARAM:
                if isinstance(pragma.value, dict) and parameter.name in pragma.value.get("params", []):
                    return True
        return False

    def _generate_kernel_attributes_list(self, hw_kernel: HWKernel) -> list:
        """
        Generates a list of kernel-specific attributes for the Jinja2 template,
        based on HWKernel.parameters that are not part of a 'derived_param' pragma.
        Each item in the list is a tuple: (name, type_char, required, default_value_repr).
        """
        kernel_attributes = []
        # Ensure self._is_derived_param is correctly called and hw_kernel.pragmas is accessed safely
        non_derived_params = [
            p for p in hw_kernel.parameters 
            if not self._is_derived_param(p, getattr(hw_kernel, 'pragmas', []))
        ]
        
        for param in non_derived_params:
            default_val = param.default_value if param.default_value is not None else 0 
            attr_type_char = 'i' # Default to integer
            if param.type: # Assuming Parameter has a 'type' field from RTL parsing
                param_type_lower = str(param.type).lower() # Ensure type is string before lower()
                if "int" in param_type_lower:
                    attr_type_char = 'i'
                elif "float" in param_type_lower or "double" in param_type_lower:
                    attr_type_char = 'f'
                elif "string" in param_type_lower or "char" in param_type_lower:
                     attr_type_char = 's'
                elif "bool" in param_type_lower: # Handle boolean type
                    attr_type_char = 'i' # Booleans are often represented as 0/1 in FINN attributes
                    if isinstance(default_val, bool):
                        default_val = 1 if default_val else 0 # Convert bool to int for consistency
            
            if isinstance(default_val, str):
                default_val_str = f"'{default_val}'" # Ensure strings are quoted
            elif isinstance(default_val, bool): # Should be caught by bool conversion for attr_type_char 'i'
                default_val_str = str(1 if default_val else 0) 
            else: # Numbers
                default_val_str = str(default_val)

            # For now, assume all kernel-specific parameters are required=True
            # This might need refinement based on how HWKernel specifies optional parameters.
            required = True 
            kernel_attributes.append((param.name, attr_type_char, required, default_val_str))
        
        return kernel_attributes

    def _generate_derived_methods(self, hw_kernel: HWKernel) -> dict:
        """Generates Python code strings for derived methods."""
        derived_methods = {}

        # Helper to find a parameter by name (defined inside _generate_derived_methods in original)
        # def get_param_value(name, default=None):
        #     for p in hw_kernel.parameters:
        #         if p.name == name:
        #             return p.default_value if p.default_value is not None else default
        #     return default

        # --- get_input_datatype ---
        idt_param_name = hw_kernel.idt_param_name if hasattr(hw_kernel, 'idt_param_name') else "IDT"
        derived_methods["get_input_datatype"] = f'''\
    def get_input_datatype(self, ind=0):
        """Returns FINN DataType for input stream `ind`."""
        # This method should be adapted by the user if the DataType
        # is not directly specified by a top-level parameter.
        # Example: DataType[self.get_nodeattr("{idt_param_name}")]
        # or based on specific logic for `ind`.
        if ind >= {hw_kernel.num_input_streams}:
            raise ValueError(f"Invalid input stream index {{ind}} for {hw_kernel.num_input_streams} inputs.")
        # Fallback or more complex logic might be needed here.
        # This is a placeholder based on a common pattern.
        try:
            return DataType[self.get_nodeattr("{idt_param_name}")]
        except KeyError:
            raise NotImplementedError("get_input_datatype: Could not determine input DataType from '{idt_param_name}'. Please implement.")
'''

        # --- get_output_datatype ---
        odt_param_name = hw_kernel.odt_param_name if hasattr(hw_kernel, 'odt_param_name') else "ODT"
        derived_methods["get_output_datatype"] = f'''\
    def get_output_datatype(self, ind=0):
        """Returns FINN DataType for output stream `ind`."""
        if ind >= {hw_kernel.num_output_streams}:
            raise ValueError(f"Invalid output stream index {{ind}} for {hw_kernel.num_output_streams} outputs.")
        try:
            return DataType[self.get_nodeattr("{odt_param_name}")]
        except KeyError:
            raise NotImplementedError("get_output_datatype: Could not determine output DataType from '{odt_param_name}'. Please implement.")
'''
        pe_param_name = hw_kernel.pe_param_name if hasattr(hw_kernel, 'pe_param_name') else "PE"
        # --- get_folded_input_shape ---
        ifmdim_param_name = hw_kernel.shape_info_param_name if hasattr(hw_kernel, 'shape_info_param_name') else "IFMDim" # Example
        derived_methods["get_folded_input_shape"] = f'''\
    def get_folded_input_shape(self, ind=0):
        """Return folded input shape for stream `ind`."""
        if ind >= {hw_kernel.num_input_streams}:
            raise ValueError(f"Invalid input stream index {{ind}} for {hw_kernel.num_input_streams} inputs.")
        try:
            ifm_dim = self.get_nodeattr("{ifmdim_param_name}") 
            pe = self.get_nodeattr("{pe_param_name}")
        except Exception as e:
            raise NotImplementedError(
                f"Could not retrieve '{ifmdim_param_name}' or '{pe_param_name}' for get_folded_input_shape. "
                f"Ensure these are attributes or override this method. Error: {{e}}"
            )
        if not isinstance(ifm_dim, int) or not isinstance(pe, int):
            raise TypeError(
                f"'{ifmdim_param_name}' and '{pe_param_name}' must be integers for default folding. "
                f"Got {{type(ifm_dim)}} and {{type(pe)}}."
            )
        if pe == 0:
            raise ValueError("PE (Processing Elements) cannot be zero for folding.")
        if ifm_dim % pe != 0:
            pass 
        return (1, ifm_dim // pe, pe)
'''

        # --- get_folded_output_shape ---
        ofmdim_param_name = hw_kernel.shape_info_param_name.replace("IFM", "OFM") if hasattr(hw_kernel, 'shape_info_param_name') else "OFMDim" # Example
        derived_methods["get_folded_output_shape"] = f'''\
    def get_folded_output_shape(self, ind=0):
        """Return folded output shape for stream `ind`."""
        if ind >= {hw_kernel.num_output_streams}:
            raise ValueError(f"Invalid output stream index {{ind}} for {hw_kernel.num_output_streams} outputs.")
        try:
            ofm_dim = self.get_nodeattr("{ofmdim_param_name}")
            pe = self.get_nodeattr("{pe_param_name}")
        except Exception as e:
            raise NotImplementedError(
                f"Could not retrieve '{ofmdim_param_name}' or '{pe_param_name}' for get_folded_output_shape. "
                f"Ensure these are attributes or override this method. Error: {{e}}"
            )
        if not isinstance(ofm_dim, int) or not isinstance(pe, int):
            raise TypeError(
                f"'{ofmdim_param_name}' and '{pe_param_name}' must be integers for default folding. "
                f"Got {{type(ofm_dim)}} and {{type(pe)}}."
            )
        if pe == 0:
            raise ValueError("PE (Processing Elements) cannot be zero for folding.")
        if ofm_dim % pe != 0:
            pass
        return (1, ofm_dim // pe, pe)
'''

        # --- get_normal_input_shape ---
        derived_methods["get_normal_input_shape"] = f'''\
    def get_normal_input_shape(self, ind=0):
        """Return normal input shape for stream `ind` (unfolded)."""
        if ind >= {hw_kernel.num_input_streams}:
            raise ValueError(f"Invalid input stream index {{ind}} for {hw_kernel.num_input_streams} inputs.")
        try:
            ifm_dim = self.get_nodeattr("{ifmdim_param_name}")
        except Exception as e:
            raise NotImplementedError(
                f"Could not retrieve '{ifmdim_param_name}' for get_normal_input_shape. "
                f"Ensure this is an attribute or override this method. Error: {{e}}"
            )
        if not isinstance(ifm_dim, int):
            raise TypeError(f"'{ifmdim_param_name}' must be an integer. Got {{type(ifm_dim)}}.")
        return (1, ifm_dim)
'''

        # --- get_normal_output_shape ---
        derived_methods["get_normal_output_shape"] = f'''\
    def get_normal_output_shape(self, ind=0):
        """Return normal output shape for stream `ind` (unfolded)."""
        if ind >= {hw_kernel.num_output_streams}:
            raise ValueError(f"Invalid output stream index {{ind}} for {hw_kernel.num_output_streams} outputs.")
        try:
            ofm_dim = self.get_nodeattr("{ofmdim_param_name}")
        except Exception as e:
            raise NotImplementedError(
                f"Could not retrieve '{ofmdim_param_name}' for get_normal_output_shape. "
                f"Ensure this is an attribute or override this method. Error: {{e}}"
            )
        if not isinstance(ofm_dim, int):
            raise TypeError(f"'{ofmdim_param_name}' must be an integer. Got {{type(ofm_dim)}}.")
        return (1, ofm_dim)
'''

        # --- get_instream_width ---
        derived_methods["get_instream_width"] = f'''\
    def get_instream_width(self, ind=0):
        """Return input stream width in bits for stream `ind`."""
        if ind >= {hw_kernel.num_input_streams}:
            raise ValueError(f"Invalid input stream index {{ind}} for {hw_kernel.num_input_streams} inputs.")
        idt = self.get_input_datatype(ind)
        pe = self.get_nodeattr("{pe_param_name}")
        return pe * idt.bitwidth()
'''

        # --- get_outstream_width ---
        derived_methods["get_outstream_width"] = f'''\
    def get_outstream_width(self, ind=0):
        """Return output stream width in bits for stream `ind`."""
        if ind >= {hw_kernel.num_output_streams}:
            raise ValueError(f"Invalid output stream index {{ind}} for {hw_kernel.num_output_streams} outputs.")
        odt = self.get_output_datatype(ind)
        pe = self.get_nodeattr("{pe_param_name}")
        return pe * odt.bitwidth()
'''

        # --- get_instream_width_padded ---
        derived_methods["get_instream_width_padded"] = f'''\
    def get_instream_width_padded(self, ind=0):
        """Return input stream width in bits, padded to a multiple of 8."""
        if ind >= {hw_kernel.num_input_streams}:
            raise ValueError(f"Invalid input stream index {{ind}} for {hw_kernel.num_input_streams} inputs.")
        return roundup_to_integer_multiple(self.get_instream_width(ind), 8)
'''

        # --- get_outstream_width_padded ---
        derived_methods["get_outstream_width_padded"] = f'''\
    def get_outstream_width_padded(self, ind=0):
        """Return output stream width in bits, padded to a multiple of 8."""
        if ind >= {hw_kernel.num_output_streams}:
            raise ValueError(f"Invalid output stream index {{ind}} for {hw_kernel.num_output_streams} outputs.")
        return roundup_to_integer_multiple(self.get_outstream_width(ind), 8)
'''
        
        # --- get_verilog_top_module_intf_names ---
        s_axis_tuples = []
        for i in range(hw_kernel.num_input_streams):
            s_axis_name = hw_kernel.input_axi_stream_names[i] if hasattr(hw_kernel, 'input_axi_stream_names') and i < len(hw_kernel.input_axi_stream_names) else f"s_axis_input_{i}"
            s_axis_tuples.append(f'("{s_axis_name}", self.get_instream_width_padded({i}))')
        
        m_axis_tuples = []
        for i in range(hw_kernel.num_output_streams):
            m_axis_name = hw_kernel.output_axi_stream_names[i] if hasattr(hw_kernel, 'output_axi_stream_names') and i < len(hw_kernel.output_axi_stream_names) else f"m_axis_output_{i}"
            m_axis_tuples.append(f'("{m_axis_name}", self.get_outstream_width_padded({i}))')

        axilite_list_str = f'["{hw_kernel.axilite_name}"]' if hw_kernel.has_axilite and hasattr(hw_kernel, 'axilite_name') and hw_kernel.axilite_name else "[]"
        clk_name = hw_kernel.clk_name if hasattr(hw_kernel, 'clk_name') else "ap_clk"
        rst_name = hw_kernel.rst_name if hasattr(hw_kernel, 'rst_name') else "ap_rst_n"

        derived_methods["get_verilog_top_module_intf_names"] = f'''\
    def get_verilog_top_module_intf_names(self):
        """Return a dict of names of input and output interfaces."""
        intf_names = {{}}
        intf_names["clk"] = ["{clk_name}"]
        intf_names["rst"] = ["{rst_name}"]
        intf_names["s_axis"] = [{", ".join(s_axis_tuples)}]
        intf_names["m_axis"] = [{", ".join(m_axis_tuples)}]
        intf_names["aximm"] = [] 
        intf_names["axilite"] = {axilite_list_str}
        intf_names["ap_none"] = [] 
        return intf_names
'''
        # --- hls_sname (default implementation) ---
        derived_methods["hls_sname"] = '''\
    def hls_sname(self):
        """Return the HLS SNAME string, typically 'V' for AXI stream data."""
        return "V"
'''
        return derived_methods

    def _prepare_context(self, hw_kernel: HWKernel, parsed_compiler_data: dict, class_name_suffix: str = "HWCustomOp") -> dict:
        """
        Prepares the context dictionary for rendering the Jinja2 template.
        """
        context = {}
        context["generation_timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        kernel_name_capitalized = hw_kernel.name.capitalize().replace('_', '')
        context["class_name"] = f"{kernel_name_capitalized}{class_name_suffix}"
        context["hw_kernel"] = hw_kernel
        context["hw_kernel_repr"] = pprint.pformat(hw_kernel, indent=4) 
        context["description"] = f"Automatically generated HWCustomOp for {hw_kernel.name}."

        context["kernel_attributes"] = self._generate_kernel_attributes_list(hw_kernel)

        user_methods = {}
        derived_methods = self._generate_derived_methods(hw_kernel) 
        static_methods = {}
        placeholder_methods = {} 
        user_imports = "" 

        if parsed_compiler_data:
            # Ensure kernel_name_capitalized matches how CompilerDataParser might store class names
            # For "MyTestKernel", capitalize() -> "Mytestkernel"
            # If parser uses "MyTestKernel", adjust here or in parser.
            # Assuming parser uses exact name or we adapt.
            # For now, assume CompilerDataParser keys might be exact like "MyTestKernelCompilerData"
            # or based on kernel_name.capitalize() + "CompilerData".
            # The current logic uses kernel_name.capitalize().replace('_', '')
            
            target_compiler_data_class_name = f"{kernel_name_capitalized}CompilerData"
            
            class_methods_data = parsed_compiler_data.get("class_methods")
            if class_methods_data and target_compiler_data_class_name in class_methods_data:
                for method_name, method_code in class_methods_data[target_compiler_data_class_name].items():
                    if method_name in derived_methods:
                        logger.info(f"User method '{method_name}' overrides a derived method.")
                    
                    method_lines = method_code.strip().splitlines()
                    is_static = False
                    for line in method_lines:
                        stripped_line = line.strip()
                        if stripped_line.startswith("@staticmethod"):
                            is_static = True
                            break 
                        if stripped_line and not stripped_line.startswith("@"): # Reached code beyond decorators
                            break
                            
                    if is_static:
                        # Store the method code, template will add @staticmethod if needed or it's part of code
                        static_methods[method_name] = method_code 
                    else:
                        user_methods[method_name] = method_code
            
            top_level_functions = parsed_compiler_data.get("functions", {})
            for func_name, func_code in top_level_functions.items():
                if func_name not in static_methods and func_name not in user_methods:
                    # Treat top-level functions as static methods if not already categorized
                    # Check if it already has @staticmethod
                    is_static_in_code = False
                    func_lines = func_code.strip().splitlines()
                    for line in func_lines:
                        if line.strip().startswith("@staticmethod"):
                            is_static_in_code = True
                            break
                    
                    if is_static_in_code:
                        static_methods[func_name] = func_code
                    else:
                        # If template handles adding @staticmethod, just pass the code.
                        # Or prepend it here if template expects it.
                        # Current template adds @staticmethod before rendering static_methods.
                        static_methods[func_name] = func_code


            user_imports = parsed_compiler_data.get("imports_str", "")

        context["user_imports"] = user_imports
        context["user_methods"] = user_methods
        context["derived_methods"] = derived_methods 
        context["static_methods"] = static_methods
        
        default_placeholders = {
            "make_shape_compatible_op": {
                "signature": "self, model",
                "docstring": "Return a shape-compatible ONNX op for this node.",
                "body": 'raise NotImplementedError("make_shape_compatible_op is not implemented for this kernel.")'
            },
            # get_verilog_top_module_name is in template, but can be a placeholder if complex
            # "get_verilog_top_module_name": { 
            #     "signature": "self",
            #     "docstring": "Return the Verilog top module name.",
            #     "body": 'return self.get_nodeattr("entity_name")'
            # },
            # execute_node, infer_node_datatype, verify_node have defaults in template.
        }
        
        for name, details in default_placeholders.items():
            if name not in user_methods and name not in derived_methods and name not in static_methods:
                placeholder_methods[name] = details
        
        context["placeholder_methods"] = placeholder_methods
        # Add a flag for the main block in template (optional)
        context["add_main_block_for_testing"] = False # Set to True if template supports it

        return context

    def generate(self, hw_kernel: HWKernel, parsed_compiler_data: dict, class_name_suffix: str = "HWCustomOp") -> str:
        """
        Renders the HWCustomOp Python class string.
        """
        template = self.template_env.get_template("hw_custom_op.py.j2")
        context = self._prepare_context(hw_kernel, parsed_compiler_data, class_name_suffix)
        # Import necessary for derived methods using DataType or roundup_to_integer_multiple
        # These are already in the template, but good to be aware for context.
        # context['DataType'] = DataType # If directly used in generated method strings
        # context['roundup_to_integer_multiple'] = roundup_to_integer_multiple

        return template.render(context)

# Test harness
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting HWCustomOpGenerator test harness...")

    # Try to use real data classes, fall back to dummies if needed
    # These dummy classes (HWKernel, Parameter, etc.) are assumed to be defined 
    # at the top of this file if the `try...except ImportError` block for rtl_parser.data fails.
    # If the import was successful, HWKernel will refer to the real class.
    
    # 1. Create a dummy HWKernel object
    # Ensure the dummy HWKernel class is the one defined at the top if imports fail,
    # or the real one if imports succeed.
    
    # Check if the real HWKernel was imported, otherwise use the dummy defined in this file
    # This logic relies on how HWKernel is defined/imported at the top of the file.
    # For this test, we assume HWKernel and Parameter are available (either real or dummy).

    dummy_hw_kernel = HWKernel( # Use the (potentially dummy) HWKernel class
        name="MyTestKernel",
        pe_param_name="PE_COUNT",
        idt_param_name="INPUT_DATATYPE",
        odt_param_name="OUTPUT_DATATYPE",
        shape_info_param_name="IFM_DIM", 
        num_input_streams=1,
        num_output_streams=1,
        has_axilite=True,
        clk_name="ap_clk_fast",
        rst_name="ap_rst_n_fast"
    )
    dummy_hw_kernel.parameters = [
        Parameter(name="PE_COUNT", type="int", default_value=4),
        Parameter(name="INPUT_DATATYPE", type="string", default_value="INT4"),
        Parameter(name="OUTPUT_DATATYPE", type="string", default_value="INT4"), # Corrected from INT4 to string "INT4"
        Parameter(name="IFM_DIM", type="int", default_value=1024),
        Parameter(name="OFM_DIM", type="int", default_value=1024), 
        Parameter(name="SOME_OTHER_PARAM", type="float", default_value=0.5),
        Parameter(name="IS_ENABLED", type="bool", default_value=True),
        Parameter(name="KERNEL_SIZE", type="int", default_value=3), # Example derived
    ]
    # Example of a pragma to mark KERNEL_SIZE as derived (won't become a node attribute)
    # This requires PragmaType to be defined (dummy or real)
    # dummy_hw_kernel.pragmas = [
    #     type('DummyPragma', (object,), {'type': PragmaType.DERIVED_PARAM, 'value': {'params': ['KERNEL_SIZE']}})()
    # ]


    # 2. Simulate parsed_compiler_data (output from CompilerDataParser)
    # For "MyTestKernel", kernel_name_capitalized becomes "Mytestkernel"
    # So, class name in parsed_data should be "MytestkernelCompilerData"
    parsed_data_for_generator = {
        "imports_str": "import numpy as my_np\nfrom qonnx.core.datatype import DataType", # Added DataType for user_method_two
        "class_methods": {
            "MytestkernelCompilerData": { # Matches kernel_name_capitalized + "CompilerData"
                "user_method_one": '''def user_method_one(self, x):
    """Docstring for user_method_one."""
    # Example using an attribute
    pe_count = self.get_nodeattr("PE_COUNT")
    return x * pe_count''',
                "get_input_datatype": '''def get_input_datatype(self, ind=0):
    """User override for get_input_datatype."""
    if ind == 0:
        return DataType[self.get_nodeattr("INPUT_DATATYPE")]
    # Example: could have different types for different input streams
    # elif ind == 1: return DataType["UINT8"] 
    raise ValueError(f"Unknown input index {ind} for custom get_input_datatype")'''
            }
        },
        "functions": { # Top-level functions are treated as static methods
             "user_static_method": '''@staticmethod
def user_static_method(a, b):
    """Docstring for user_static_method."""
    return a + b''',
             "top_level_utility_func": '''def top_level_utility_func(data):
    """Docstring for top_level_utility_func."""
    return my_np.array(data)'''
        }
    }

    # 3. Instantiate HWCustomOpGenerator
    generator = HWCustomOpGenerator()
    logger.info(f"HWCustomOpGenerator instantiated. Using template from: {generator.template_env.loader.searchpath[0]}")

    # 4. Generate the code
    logger.info("Generating HWCustomOp class code for MyTestKernel...")
    generated_code = generator.generate(dummy_hw_kernel, parsed_data_for_generator, class_name_suffix="CustomOp")

    # 5. Print the generated code
    print("\n" + "="*50 + "\nGenerated Code:\n" + "="*50 + "\n")
    print(generated_code)
    print("\n" + "="*50 + "\nEnd of Generated Code\n" + "="*50 + "\n")

    output_filename = "generated_MyTestKernelCustomOp.py"
    try:
        with open(output_filename, "w") as f:
            f.write(generated_code)
        logger.info(f"Generated code also written to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to write generated code to {output_filename}: {e}")

    logger.info("HWCustomOpGenerator test harness finished.")
