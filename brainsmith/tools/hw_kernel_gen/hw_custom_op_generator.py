############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

# /home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/hw_custom_op_generator.py
"""
Generates the HWCustomOp Python class file for a given HWKernel.

This class takes parsed RTL information (HWKernel) and potentially
Python-defined metadata (HWKernelPy) to generate a Python class
that inherits from finn.custom_op.fpgadataflow.hwcustomop.HWCustomOp.
This generated class serves as the interface between the FINN compiler
and the custom hardware kernel.
"""

import jinja2
import logging
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import HWKernel
from brainsmith.tools.hw_kernel_gen.data import HWKernelPy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "hw_custom_op.py.j2"
)

class HWCustomOpGenerator:
    """
    Generates a Python class inheriting from HWCustomOp based on HWKernel data.

    Uses a Jinja2 template to structure the output Python file, populating it
    with node attributes derived from RTL parameters and standard placeholder
    methods required by the FINN HWCustomOp interface.

    Attributes:
        hw_kernel: The HWKernel object containing parsed RTL information.
        template_path: Path to the Jinja2 template file.
        kernel_name_override: Optional string to explicitly set the class name.
        hw_kernel_py: Optional HWKernelPy object with Python-defined metadata.
        template: Loaded Jinja2 template object.
        class_name: The derived or overridden name for the generated Python class.
    """
    def __init__(
        self,
        hw_kernel: HWKernel,
        template_path: str = DEFAULT_TEMPLATE_PATH,
        kernel_name_override: Optional[str] = None,
        hw_kernel_py: Optional[HWKernelPy] = None,
    ):
        """
        Initializes the HWCustomOpGenerator.

        Args:
            hw_kernel: The HWKernel object from the RTL parser.
            template_path: Path to the Jinja2 template file.
                         Defaults to DEFAULT_TEMPLATE_PATH.
            kernel_name_override: Optional name to override the derived class name.
                                  If None, the name is derived from hw_kernel.name.
            hw_kernel_py: Optional object containing Python-defined kernel data.
                          Used for future enhancements like cost modeling.

        Raises:
            TypeError: If hw_kernel is not an instance of HWKernel.
            ValueError: If class name cannot be derived and no override is provided.
            jinja2.TemplateNotFound: If the template file cannot be found.
            Exception: For other template loading errors.
        """
        if not isinstance(hw_kernel, HWKernel):
            raise TypeError("hw_kernel must be an instance of HWKernel")

        self.hw_kernel = hw_kernel
        self.template_path = template_path
        self.kernel_name_override = kernel_name_override
        self.hw_kernel_py = hw_kernel_py

        try:
            self.template = self._load_template()
        except (jinja2.TemplateNotFound, Exception) as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise # Re-raise after logging

        try:
            self.class_name = self._derive_class_name()
        except ValueError as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise # Re-raise after logging

        logger.info(f"HWCustomOpGenerator initialized for kernel '{self.hw_kernel.name}' -> class '{self.class_name}'")

    def _load_template(self) -> jinja2.Template:
        """
        Loads the Jinja2 template from the specified path.

        Returns:
            The loaded Jinja2 template object.

        Raises:
            jinja2.TemplateNotFound: If the template file doesn't exist.
            Exception: For other file reading or Jinja2 environment errors.
        """
        try:
            template_dir = os.path.dirname(self.template_path)
            template_filename = os.path.basename(self.template_path)
            if not os.path.exists(self.template_path):
                 raise jinja2.TemplateNotFound(template_filename)
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            return env.get_template(template_filename)
        except jinja2.TemplateNotFound:
            logger.error(f"Template not found at: {self.template_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading template {self.template_path}: {e}")
            raise

    def _derive_class_name(self) -> str:
        """
        Derives the Python class name (PascalCase) from the HWKernel name
        or uses the override name if provided.

        Returns:
            The derived or overridden class name as a string.

        Raises:
            ValueError: If hw_kernel.name is empty and no override is given,
                        or if the derived name is invalid.
        """
        if self.kernel_name_override:
            # Basic validation for override name
            if not re.match(r"^[A-Z][a-zA-Z0-9_]*$", self.kernel_name_override):
                 logger.warning(
                     f"kernel_name_override '{self.kernel_name_override}'"
                     " may not follow PascalCase convention."
                 )
            return self.kernel_name_override

        if not self.hw_kernel.name:
             raise ValueError("HWKernel name is empty, cannot derive class name.")

        # Convert snake_case or kebab-case to PascalCase
        name = self.hw_kernel.name.replace('-', '_')
        parts = name.split('_')
        # Capitalize first letter of each part, handle potential empty strings from multiple underscores
        pascal_case_name = "".join(p.capitalize() for p in parts if p)

        if not pascal_case_name:
             raise ValueError(f"Could not derive a valid class name from '{self.hw_kernel.name}'")

        # Ensure it starts with an uppercase letter (should be guaranteed by capitalize)
        if not re.match(r"^[A-Z]", pascal_case_name):
             # This case might happen if the original name was just underscores
             raise ValueError(f"Derived class name '{pascal_case_name}' is invalid.")

        logger.info(f"Derived class name: {pascal_case_name} from HWKernel name: {self.hw_kernel.name}")
        return pascal_case_name

    def _generate_node_attributes(self) -> Dict[str, tuple]:
        """
        Generates the node attributes dictionary for the HWCustomOp class.

        Maps RTL parameters (like WIDTH, DEPTH) from the HWKernel object
        to the format expected by FINN's `get_nodeattr_types` method:
        `(type_string, required_boolean, default_value, allowed_values)`.

        Current Type Mapping:
        - int, integer -> 'i'
        - bit, logic (single) -> 'i'
        - string -> 's'
        - Others default to 's'

        Handles basic Verilog integer literal formats (e.g., 'd10, 'hF, 'b1) for defaults.

        Returns:
            A dictionary where keys are attribute names (matching parameter names)
            and values are tuples describing the attribute type, requirement, and default.
        """
        my_attrs = {}
        logger.debug(f"Generating node attributes from {len(self.hw_kernel.parameters)} parameters.")
        for param in self.hw_kernel.parameters:
            finn_type = "s" # Default to string
            finn_default_value = None
            required = True # Default assumption

            # Basic Type Mapping (can be expanded)
            param_type_lower = param.param_type.lower() if param.param_type else ""
            if "int" in param_type_lower or "integer" in param_type_lower:
                finn_type = "i"
            # Treat single bit/logic as int (0/1), check for vectors
            elif param_type_lower in ["bit", "logic"] and not (
                '[' in param.param_type or 'signed' in param.param_type or 'unsigned' in param.param_type
            ):
                 finn_type = "i"
            elif param_type_lower == "string":
                 finn_type = "s"
            # Add more specific mappings if needed

            # Default Value Handling
            if param.default_value is not None:
                val_str = param.default_value.strip()
                if finn_type == "i":
                    try:
                        # Handle Verilog integer literals like 'd, 'h, 'b, 'o
                        if "'b" in val_str:
                            finn_default_value = int(val_str.split("'b")[-1], 2)
                        elif "'h" in val_str:
                            finn_default_value = int(val_str.split("'h")[-1], 16)
                        elif "'d" in val_str:
                            finn_default_value = int(val_str.split("'d")[-1], 10)
                        elif "'o" in val_str:
                            finn_default_value = int(val_str.split("'o")[-1], 8)
                        else:
                            # Try direct integer conversion
                            finn_default_value = int(val_str)
                    except ValueError:
                        logger.warning(
                            f"Could not convert default value '{val_str}' "
                            f"to int for parameter '{param.name}'. Keeping as None."
                        )
                        finn_default_value = None # Fallback if conversion fails
                elif finn_type == "s":
                    # Remove potential surrounding quotes
                    if (val_str.startswith('"') and val_str.endswith('"')) or \
                       (val_str.startswith("'") and val_str.endswith("'")):
                        finn_default_value = val_str[1:-1]
                    else:
                        finn_default_value = val_str
                else:
                     finn_default_value = val_str # Keep as string for other types initially

            # Store as tuple: (type_code, required, default_value, allowed_values)
            # allowed_values is None for now, will be handled by pragmas later
            my_attrs[param.name] = (finn_type, required, finn_default_value, None)
            logger.debug(f"  Mapped param '{param.name}' ({param.param_type}) -> {my_attrs[param.name]}")

        # --- Future: Add attributes based on interfaces (Phase 2b/Future Phase) ---
        # Example logic (needs InterfaceType enum from rtl_parser.data)
        # from brainsmith.tools.hw_kernel_gen.rtl_parser.data import InterfaceType
        # input_axis_count = sum(1 for iface in self.hw_kernel.interfaces.values() if iface.type == InterfaceType.AXI_STREAM and iface.name.startswith('s_axis')) # Crude direction check
        # output_axis_count = sum(1 for iface in self.hw_kernel.interfaces.values() if iface.type == InterfaceType.AXI_STREAM and iface.name.startswith('m_axis')) # Crude direction check
        #
        # if input_axis_count >= 1:
        #     my_attrs["inputDataType"] = ("s", True, "", None)
        # if input_axis_count >= 2:
        #      my_attrs["weightDataType"] = ("s", True, "", None)
        # if output_axis_count >= 1:
        #      my_attrs["outputDataType"] = ("s", True, "", None)
        # logger.debug(f"Added data type attributes based on interface counts.")
        # --- End Future ---

        logger.info(f"Generated {len(my_attrs)} node attributes from parameters.")
        return my_attrs

    def _generate_placeholder_methods(self) -> Dict[str, str]:
        """
        Generates a dictionary of standard HWCustomOp placeholder method implementations.

        Provides basic string representations of common methods required by HWCustomOp
        and its typical usage within FINN transformations (like shape inference,
        datatype setting, execution, resource estimation).
        Each method raises NotImplementedError by default.

        Returns:
            A dictionary where keys are method names and values are multi-line
            strings containing the Python code for the placeholder method.
        """
        # Define standard methods and their basic signatures/bodies using f-strings.
        # Literal curly braces within the strings must be doubled {{ }}
        # Interpolated variables use single braces { }.

        methods = {
            "make_shape_compatible_op": f"""
def make_shape_compatible_op(self, model):
    \"""Return a standard ONNX op that is shape-compatible with this node.\"""\
    # Example: return oh.make_node('Identity', [self.onnx_node.input[0]], [self.onnx_node.output[0]])
    raise NotImplementedError(f"make_shape_compatible_op needs implementation for {{self.onnx_node.name}}")
""",
            "infer_node_datatype": f"""
def infer_node_datatype(self, model):
    \"""Set the tensor datatypes for inputs/outputs of this node.\"""\
    # Example: Infer input datatype from context
    # node = self.onnx_node
    # idt = model.get_tensor_datatype(node.input[0])
    # self.set_nodeattr("inputDataType", idt.name)
    # Example: Set output datatype based on attribute
    # odt = self.get_output_datatype()
    # model.set_tensor_datatype(node.output[0], odt)
    raise NotImplementedError(f"infer_node_datatype needs implementation for {{self.onnx_node.name}}")
""",
            "verify_node": f"""
def verify_node(self):
    \"""Verify node attributes.\"""\
    # Add verification logic specific to this kernel's parameters and interfaces
    # Example: Check if PE divides NumChannels
    # try:
    #     pe = self.get_nodeattr("PE")
    #     num_channels = self.get_nodeattr("NumChannels")
    #     if num_channels % pe != 0:
    #         warnings.warn(f"{{self.onnx_node.name}}: NumChannels ({{num_channels}}) must be divisible by PE ({{pe}})")
    # except KeyError:
    #      # Handle cases where attributes might not be set yet
    #      pass
    # except Exception as e:
    #      warnings.warn(f"Could not verify node {{self.onnx_node.name}}: {{e}}")
    #
    # Call base class verification if needed
    # super().verify_node()
    pass # Start with a pass, add specific checks later
""",
            "get_input_datatype": f"""
def get_input_datatype(self, ind=0):
    \"""Return the FINN DataType for the input stream specified by ind.\"""\
    # Example: return DataType[self.get_nodeattr("inputDataType")]
    raise NotImplementedError(f"get_input_datatype needs implementation for {{self.onnx_node.name}}")
""",
            "get_output_datatype": f"""
def get_output_datatype(self, ind=0):
    \"""Return the FINN DataType for the output stream specified by ind.\"""\
    # Example: return DataType[self.get_nodeattr("outputDataType")]
    raise NotImplementedError(f"get_output_datatype needs implementation for {{self.onnx_node.name}}")
""",
            "get_weight_datatype": f"""
def get_weight_datatype(self):
    \"""Return the FINN DataType for the thresholds/weights (if applicable).\"""\
    # Example: return DataType[self.get_nodeattr("weightDataType")]
    raise NotImplementedError(f"get_weight_datatype needs implementation for {{self.onnx_node.name}}")
""",
            "get_instream_width": f"""
def get_instream_width(self, ind=0):
    \"""Return the width of the input stream specified by ind in bits.\"""\
    # Example: return self.get_input_datatype(ind).bitwidth()
    raise NotImplementedError(f"get_instream_width needs implementation for {{self.onnx_node.name}}")
""",
            "get_outstream_width": f"""
def get_outstream_width(self, ind=0):
    \"""Return the width of the output stream specified by ind in bits.\"""\
    # Example: return self.get_output_datatype(ind).bitwidth()
    raise NotImplementedError(f"get_outstream_width needs implementation for {{self.onnx_node.name}}")
""",
            "get_folded_input_shape": f"""
def get_folded_input_shape(self, ind=0):
    \"""Return the shape of the folded input tensor specified by ind.\"""\
    # Needs to be implemented based on kernel's folding logic (e.g., using PE, SIMD)
    raise NotImplementedError(f"get_folded_input_shape needs implementation for {{self.onnx_node.name}}")
""",
            "get_folded_output_shape": f"""
def get_folded_output_shape(self, ind=0):
    \"""Return the shape of the folded output tensor specified by ind.\"""\
    # Needs to be implemented based on kernel's folding logic
    raise NotImplementedError(f"get_folded_output_shape needs implementation for {{self.onnx_node.name}}")
""",
            "get_normal_input_shape": f"""
def get_normal_input_shape(self, ind=0):
    \"""Return the shape of the normal (unfolded) input tensor specified by ind.\"""\
    # Usually retrieved from the ONNX graph context
    # Example: return tuple(self.onnx_node.get_normal_input_shape(ind))
    raise NotImplementedError(f"get_normal_input_shape needs implementation for {{self.onnx_node.name}}")
""",
            "get_normal_output_shape": f"""
def get_normal_output_shape(self, ind=0):
    \"""Return the shape of the normal (unfolded) output tensor specified by ind.\"""\
    # Usually calculated based on input shape and kernel operation
    raise NotImplementedError(f"get_normal_output_shape needs implementation for {{self.onnx_node.name}}")
""",
            "get_number_output_values": f"""
def get_number_output_values(self, ind=0):
    \"""Return the number of output values produced (e.g., for cycle estimation). Index 'ind' for multi-output nodes.\"""\
    # Example: return np.prod(self.get_folded_output_shape(ind)[:-1])
    raise NotImplementedError(f"get_number_output_values needs implementation for {{self.onnx_node.name}}")
""",
            "get_exp_cycles": f"""
def get_exp_cycles(self):
    \"""Return the expected number of clock cycles for execution.\"""\
    # Needs to be implemented based on kernel's pipeline depth and data flow
    # Example: return self.get_number_output_values() + pipeline_depth
    raise NotImplementedError(f"get_exp_cycles needs implementation for {{self.onnx_node.name}}")
""",
            "execute_node": f"""
def execute_node(self, context, graph):
    \"""Execute the node in simulation (e.g., using pyverilator or a Python model).\"""\
    # This is crucial for verification and debugging
    raise NotImplementedError(f"execute_node needs implementation for {{self.onnx_node.name}}")
""",
            # Resource Estimation Placeholders
            "bram_estimation": f"""
def bram_estimation(self):
    \"""Estimate BRAM usage based on parameters.\"""\
    # Implement cost model based on hw_kernel_py.cost_functions or specific logic
    warnings.warn(f"BRAM estimation not implemented for {{self.onnx_node.name}}")
    return 0
""",
            "lut_estimation": f"""
def lut_estimation(self):
    \"""Estimate LUT usage based on parameters.\"""\
    # Implement cost model based on hw_kernel_py.cost_functions or specific logic
    warnings.warn(f"LUT estimation not implemented for {{self.onnx_node.name}}")
    return 0
""",
            "uram_estimation": f"""
def uram_estimation(self):
    \"""Estimate URAM usage based on parameters.\"""\
    # Implement cost model based on hw_kernel_py.cost_functions or specific logic
    warnings.warn(f"URAM estimation not implemented for {{self.onnx_node.name}}")
    return 0
""",
            # --- Potential RTLBackend methods (commented out in template) ---
            # These are often part of a separate RTLBackend class or mixin
            # "get_verilog_top_module_name": (...),
            # "get_verilog_top_module_intf_names": (...),
            # "generate_params": (...),
            # "code_generation_ipgen": (...),
            # "code_generation_ipi": (...),
        }
        logger.debug(f"Generated {len(methods)} placeholder method definitions.")
        return methods

