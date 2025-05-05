import jinja2 # Import jinja2
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ..rtl_parser import HWKernel, InterfaceType # Use type checking to avoid circular import

# Determine the path to the templates directory relative to this file
_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
_TEMPLATE_FILE = "rtl_wrapper.v.j2"

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

    # Prepare context for the template
    # Need to import InterfaceType here for use within the template context
    from ..rtl_parser import InterfaceType
    context = {
        "kernel": hw_kernel_data,
        "interfaces": hw_kernel_data.interfaces,
        "InterfaceType": InterfaceType, # Pass the enum itself
        "generation_timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'), # Add UTC marker
    }

    # Render the template
    try:
        rendered_content = template.render(context) # Render using the context
    except jinja2.TemplateError as e:
        print(f"Error rendering template {_TEMPLATE_FILE}: {e}")
        raise # Re-raise the exception

    # Determine output filename and path
    output_filename = f"{hw_kernel_data.name}_wrapper.v" # Use kernel.name and .v extension
    output_path = output_dir / output_filename

    # Write the rendered content to the output file
    with open(output_path, "w") as f:
        f.write(rendered_content) # Write the rendered content

    print(f"Successfully generated RTL wrapper: {output_path}")
    return output_path # Return the path
