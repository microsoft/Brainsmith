**RTL Template Generator Implementation Plan**

**Phase 1: Update Data Structures and Interface Processing**

1.  **Modify `Parameter` Dataclass (data.py):**
    *   Add the field: `template_param_name: str = field(init=False)`
    *   In `__post_init__`, add the line: `self.template_param_name = f"${self.name.upper()}$"`
2.  **Modify Interface Creation/Validation (Likely in `rtl_parser/interface_builder.py` and/or `protocol_validator.py`):**
    *   When creating the final `Interface` object for AXI-Stream and AXI-Lite:
        *   Iterate through the validated ports within the group.
        *   Identify key signals (like `tdata`, `awaddr`, `araddr`, `wdata`, `rdata`).
        *   Extract the `width` attribute (which is a string, e.g., `"[(WIDTH*PE)-1:0]"`) from the corresponding `Port` object.
        *   Store these width strings in the `Interface.metadata` dictionary. Use clear keys, for example:
            *   `metadata['data_width_expr'] = port.width` (for `tdata`, `wdata`, `rdata`)
            *   `metadata['addr_width_expr'] = port.width` (for `awaddr`, `araddr`)
            *   *Consider:* Also store the width for `tkeep`/`wstrb` if needed: `metadata['keep_width_expr'] = port.width`.

**Phase 2: Implement Verilog Jinja2 Template (`templates/rtl_wrapper.v.j2`)**

1.  **Rename Template:** Ensure the template file is named `rtl_wrapper.v.j2`.
2.  **Module Definition:**
    *   Define the wrapper module with a fixed name derived from the kernel: `module {{ kernel.module_name }}_wrapper #(`.
    *   Declare parameters using Verilog syntax: Iterate through `kernel.parameters`. For each, output `parameter {{ parameter.name }} = {{ parameter.template_param_name }}`. Handle commas correctly.
    *   Close parameter list: `) (`.
3.  **Port Definition:**
    *   Iterate through `kernel.interfaces`, ensuring a consistent order (e.g., Globals, AXI-Stream Inputs, AXI-Stream Outputs, AXI-Lite). You might need a helper function or property in `HWKernel` to provide sorted interfaces.
    *   **Global Ports:** Declare using Verilog syntax: `{{ port.direction.value }} {{ port.name }}`.
    *   **AXI-Stream Ports:**
        *   Use the `interface.name` (e.g., `in0`, `out0`) as the prefix.
        *   Declare standard signals using Verilog syntax and widths from metadata:
            *   `{{ port.direction.value }} [{{ interface.metadata.get('data_width_expr', '0:0') }}] {{ interface.name }}_tdata`
            *   `{{ port.direction.value }} {{ interface.name }}_tvalid`
            *   `{{ port.direction.value }} {{ interface.name }}_tready`
            *   Add other signals (`tlast`, `tkeep`, etc.) if they exist in `interface.ports`, using `port.direction.value` and the appropriate width expression from `port.width` or `interface.metadata`.
    *   **AXI-Lite Ports:**
        *   Use the `interface.name` (e.g., `config`) as the prefix.
        *   Declare standard signals using Verilog syntax and widths from metadata:
            *   `{{ port.direction.value }} [{{ interface.metadata.get('addr_width_expr', '0:0') }}] {{ interface.name }}_awaddr` (and `_araddr`)
            *   `{{ port.direction.value }} [{{ interface.metadata.get('data_width_expr', '0:0') }}] {{ interface.name }}_wdata` (and `_rdata`)
            *   `{{ port.direction.value }} [{{ interface.metadata.get('keep_width_expr', '3:0') }}] {{ interface.name }}_wstrb` (Use appropriate default if metadata missing)
            *   Declare all other required signals (`_awvalid`, `_awready`, `_wvalid`, `_wready`, `_bvalid`, `_bready`, `_bresp`, etc.) based on `interface.ports`, using `port.direction.value`.
    *   Handle commas between port declarations correctly.
    *   Close port list: `);`.
4.  **Kernel Instantiation:**
    *   Instantiate the original kernel: `{{ kernel.module_name }} #(`.
    *   Connect parameters: `.{{ parameter.name }}( {{ parameter.name }} )`. Handle commas.
    *   Close parameter connections: `) dut (`.
    *   Connect ports: Iterate through `kernel.interfaces` and `interface.ports`. Map the original port name (`port.name` from the `Port` object stored in `interface.ports`) to the standardized wrapper port name created in the definition step (e.g., `{{ interface.name }}_tdata`). Output `.{{ port.name }}( {{ wrapper_port_name }} )`. Handle commas.
    *   Close port connections: `);`.
5.  **End Module:** `endmodule // {{ kernel.module_name }}_wrapper`.

**Phase 3: Implement Generator Logic (`generators/rtl_template_generator.py`)**

1.  **Imports:** Add `from jinja2 import Environment, FileSystemLoader, select_autoescape`.
2.  **Setup Jinja Environment:**
    *   `template_dir = Path(__file__).parent.parent / "templates"`
    *   `env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape())`
3.  **Load Template:** `template = env.get_template("rtl_wrapper.v.j2")`
4.  **Prepare Context:**
    *   Ensure `hw_kernel_data` (the `HWKernel` object passed in) has parameters with `template_param_name` correctly set (should happen automatically via `__post_init__`).
    *   Ensure `hw_kernel_data.interfaces` contains the necessary `metadata` (width expressions) added in Phase 1.
    *   Add logic to sort interfaces if needed before passing to the template: `sorted_interfaces = sorted(hw_kernel_data.interfaces, key=lambda i: (i.type.value, i.name))` (adjust sorting key as needed).
    *   `context = {"kernel": hw_kernel_data, "interfaces": sorted_interfaces}` (or pass `hw_kernel_data` directly if sorting is handled within the template or `HWKernel` class).
5.  **Render Template:** `rendered_code = template.render(context)`
6.  **Save Output:**
    *   `output_filename = f"{hw_kernel_data.module_name}_wrapper.v"`
    *   `output_path = output_dir / output_filename`
    *   Write `rendered_code` to `output_path`.
    *   Return `output_path`.
