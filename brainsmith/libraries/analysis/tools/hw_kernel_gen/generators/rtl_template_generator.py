############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import jinja2
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import datetime

# --- Import InterfaceType directly ---
from ..rtl_parser import HWKernel, InterfaceType # Use type checking to avoid circular import

# Determine the path to the templates directory relative to this file
_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
_TEMPLATE_FILE = "rtl_wrapper.v.j2"

# --- Define the desired sort order ---
INTERFACE_ORDER = [
    InterfaceType.GLOBAL_CONTROL,
    InterfaceType.AXI_STREAM,
    InterfaceType.AXI_LITE,
]

def generate_rtl_template(hw_kernel_data: 'HWKernel', output_dir: Path) -> Path:
    """
    Generates a Verilog wrapper for the given hardware kernel using a Jinja2 template.

    Args:
        hw_kernel_data: The parsed HWKernel data object containing module info,
                        parameters, and interfaces.
        output_dir: The directory where the generated Verilog file will be saved.

    Returns:
        The Path object pointing to the generated Verilog wrapper file.

    Raises:
        FileNotFoundError: If the Jinja2 template file cannot be found.
        jinja2.TemplateError: If there's an error during template rendering.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up Jinja2 environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(_TEMPLATE_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined, # Raise error for undefined variables
        extensions=['jinja2.ext.do'] # Enable the 'do' extension
    )

    try:
        template = env.get_template(_TEMPLATE_FILE) # Load the template
    except jinja2.TemplateNotFound:
        print(f"Error: Template file not found at {_TEMPLATE_DIR / _TEMPLATE_FILE}")
        raise # Re-raise the exception

    # --- Sort interfaces before passing to context ---
    all_interfaces = list(hw_kernel_data.interfaces.values())

    def get_sort_key(interface):
        try:
            primary_key = INTERFACE_ORDER.index(interface.type)
        except ValueError:
            primary_key = float('inf')
        secondary_key = interface.name
        return (primary_key, secondary_key)

    # --- Ensure standard sort order ---
    sorted_interfaces_list = sorted(all_interfaces, key=get_sort_key)
    # --- End sorting ---

    # --- Add Debugging ---
    print("\n--- Debugging Data for Template ---")
    print(f"Kernel Name: {hw_kernel_data.name}")
    print("Parameters (from hw_kernel_data.parameters):")
    # Assuming hw_kernel_data.parameters is iterable (list, tuple, etc.)
    # and items have 'name' and 'template_param_name' attributes
    if hasattr(hw_kernel_data, 'parameters') and hw_kernel_data.parameters:
        try:
            for p in hw_kernel_data.parameters:
                p_name = getattr(p, 'name', 'N/A')
                p_tpl_name = getattr(p, 'template_param_name', 'N/A')
                print(f"  - Name: {p_name}, TemplateName: {p_tpl_name}")
        except Exception as e:
            print(f"  Error iterating/accessing parameters: {e}")
            print(f"  Parameters raw: {hw_kernel_data.parameters}")
    else:
        print("  No parameters found or attribute missing.")

    print("\nSorted Interfaces List (to be passed as 'interfaces_list'):")
    if sorted_interfaces_list:
        for i in sorted_interfaces_list:
            i_name = getattr(i, 'name', 'N/A')
            i_type_val = getattr(getattr(i, 'type', None), 'value', 'N/A')
            print(f"  - Interface Name: {i_name} (Type: {i_type_val})")
            if hasattr(i, 'ports') and i.ports:
                try:
                    port_names = [getattr(p, 'name', 'N/A') for p in i.ports.values()]
                    print(f"    Ports: {port_names}")
                except Exception as e:
                     print(f"    Error iterating/accessing ports: {e}")
                     print(f"    Ports raw: {i.ports}")
            else:
                print("    No ports found or attribute missing.")
    else:
        print("  Interfaces list is empty.")
    print("--- End Debugging ---\n")
    # --- End Debugging ---


    # Prepare context for the template
    context = {
        "kernel": hw_kernel_data,
        "interfaces_list": sorted_interfaces_list,
        "InterfaceType": InterfaceType,
        "generation_timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
    }

    # Render the template
    try:
        rendered_content = template.render(context)
    except Exception as e: # Catch broader exceptions during render
        print(f"!!! Error during template rendering: {type(e).__name__}: {e}")
        # Optionally print more details or traceback
        # import traceback
        # traceback.print_exc()
        raise # Re-raise the exception

    # Determine output filename and path
    output_filename = f"{hw_kernel_data.name}_wrapper.v"
    output_path = output_dir / output_filename

    # Write the rendered content to the output file
    with open(output_path, "w") as f:
        f.write(rendered_content)

    print(f"Successfully generated RTL wrapper: {output_path}")
    return output_path
