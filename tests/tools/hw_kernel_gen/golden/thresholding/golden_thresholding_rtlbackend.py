# Placeholder for Golden RTLBackend generation for thresholding_axi
# This file would contain the expected FINN RTLBackend instance
# corresponding to the thresholding_axi kernel.

# Example structure (actual content depends on FINN integration)

def get_golden_rtlbackend(kernel_info, wrapper_path):
    """
    Generates the expected RTLBackend instance based on kernel info
    and the generated wrapper path.
    Placeholder implementation.
    """
    print("Placeholder: Generating golden RTLBackend")
    # Logic to create and configure the RTLBackend instance
    expected_backend = {
        "kernel_name": kernel_info.name,
        "wrapper_file": str(wrapper_path),
        "source_files": [
            # List of source files needed for synthesis/simulation
            "path/to/thresholding_axi.sv", # Original kernel
            str(wrapper_path)              # Generated wrapper
            # Potentially other dependencies
        ],
        "parameters": {p.name: p.template_param_name for p in kernel_info.parameters},
        "interfaces": {
            # Information about AXI interfaces needed by FINN backend
            "s_axilite": {"type": "AXI_LITE", "addr_width": "ADDR_BITS", "data_width": 32},
            "s_axis": {"type": "AXI_STREAM", "data_width": "((PE*WI+7)/8)*8"},
            "m_axis": {"type": "AXI_STREAM", "data_width": "((PE*O_BITS+7)/8)*8"}
        }
        # ... other RTLBackend attributes
    }
    return expected_backend
