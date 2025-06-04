We will now further implement the RTL Parser, adding data processing for the information extracted by the parser.

### 3. *Data Processing*

#### *Parameters*
Module parameters in the Kernel are exposed to the compiler as attributes of the generated HWCustomOp instance, and as placeholder variables in the generated wrapper template (formatted like this: $varname$)

#### *Interfaces*
Groups of ports define different interfaces. All interfaces should be identified and labeled appropriately. Any ports that don't fall within an interface definition are considered an error. There are three types of interfaces, largely from the AXI4 standard:

##### 1. *Timing & Global Control Signals*
- The "required" control signals must exist in the module's ports, or this is considered an error.

    | Signal Name | I/O   | Function          | Req./Opt. |
    | ----------- | ----- | ----------------- | --------- |
    | ap\_clk     | Input | Core clock        | Required  |
    | ap\_rst\_n  | Input | Active-low reset  | Required  |
    | ap\_clk2x   | Input | Double-rate clock | Optional  |


#### 2. *Dataflow Signals*
- These are the primary input and output points for the Kernel, streaming data to and from other Kernels. The "required" They must be implemented as AXI-Stream interfaces with this format (for i inputs and j outputs). There is no maximum number of Dataflow Signals, but each kernel must have at least one input or output.

    | Signal Name       | I/O    | Description | Interface | Width          | Req./Opt. |
    | ----------------- | ------ | ----------- | --------- | -------------- | --------- |
    | in{i}\_V\_TDATA   | Input  | Data        | s\_axis   | n (n % 8 == 0) | Required  |
    | in{i}\_V\_TREADY  | Output | Ready       | s\_axis   | 1              | Required  |
    | in{i}\_V\_TVALID  | Input  | Valid       | s\_axis   | 1              | Required  |
    | in{i}\_V\_TLAST   | Input  | Last        | s\_axis   | 1              | Optional  |
    | out{j}\_V\_TDATA  | Output | Data        | m\_axis   | m (m % 8 == 0) | Required  |
    | out{j}\_V\_TREADY | Input  | Ready       | m\_axis   | 1              | Required  |
    | out{j}\_V\_TVALID | Output | Valid       | m\_axis   | 1              | Required  |
    | out{j}\_V\_TLAST  | Output | Last        | m\_axis   | 1              | Optional  |


#### 3. *Runtime Configuration Signals*:
- AXI-Lite signals can be added for validation, debugging, and runtime configuration. It's possible for just the write or read half of the AXI-Lite to be implemented, this is considered a valid interface. Each kernel can only have one full AXI-Lite interface maximum.

    | Signal Name     | I/O    | Description  | AXI Interface | Width          | Required/Optional |
    | --------------- | ------ | ------------ | ------------- | -------------- | ----------------- |
    | config\_AWADDR  | Input  | Write addr   | Writing       | 32 (or 64)     | Required          |
    | config\_AWPROT  | Input  | Prot type    | Writing       | 3              | Required          |
    | config\_AWVALID | Input  | Addr valid   | Writing       | 1              | Required          |
    | config\_AWREADY | Output | Addr ready   | Writing       | 1              | Required          |
    | config\_WDATA   | Input  | Write data   | Writing       | 32 (or 64)     | Required          |
    | config\_WSTRB   | Input  | Byte enables | Writing       | (data-width/8) | Required          |
    | config\_WVALID  | Input  | Data valid   | Writing       | 1              | Required          |
    | config\_WREADY  | Output | Data ready   | Writing       | 1              | Required          |
    | config\_BRESP   | Output | Resp status  | Writing       | 2              | Required          |
    | config\_BVALID  | Output | Resp valid   | Writing       | 1              | Required          |
    | config\_BREADY  | Input  | Resp ready   | Writing       | 1              | Required          |
    | config\_ARADDR  | Input  | Read addr    | Reading       | 32 (or 64)     | Required          |
    | config\_ARPROT  | Input  | Prot type    | Reading       | 3              | Required          |
    | config\_ARVALID | Input  | Addr valid   | Reading       | 1              | Required          |
    | config\_ARREADY | Output | Addr ready   | Reading       | 1              | Required          |
    | config\_RDATA   | Output | Read data    | Reading       | 32 (or 64)     | Required          |
    | config\_RRESP   | Output | Resp status  | Reading       | 2              | Required          |
    | config\_RVALID  | Output | Read valid   | Reading       | 1              | Required          |
    | config\_RREADY  | Input  | Read ready   | Reading       | 1              | Required          |

    
#### *Pragmas*
Compiler data that can't be easily surmised from the code must be specified by the user via Pragmas, comments that match a specific format.

1. *Top Module*: If there are multiple modules in the file, select the top module to be templated:
    ```
    // @brainsmith top <module_name>
    ```
2. *Supported datatype*: Restrict what datatypes each Dataflow or Runtime Configuration Signal supports. FINN will determine the width of the AXI interface's data signal based on these restrictions. The name we use to identify the signal is the prefix shared by all signals in that AXI interface. If not max_size is specified, it is assumed only exactly min_size is supported.
    ```
    // @brainsmith supported_dtype <signal_prefix> <min_size> <max_size>
    ```
    Example:
    ```
    // @brainsmith supported_dtype in0 INT 4 8
    // @brainsmith supported_dtype in0 INT 16
    // @brainsmith supported_dtype in1 FLOAT 16
    // @brainsmith supported_dtype in1 INT 16 32
    ```
    It is the parser's job to determine what datatypes each interface supports, and add that information to the interface data model. Signals can have multiple of these pragmas applied to them, and the compiler will determine the superset of all supported datatypes. In the above example, the supported datatypes should be:
    ```
    in0: [INT: [4-8, 16]] 
    in1: [INT:  [16-32], FLOAT: [16]]
    ```

3. *Derived Parameter*: In some cases, one ONNX parameter in "my_attrs" may correspond to multiple parameters at the RTL level. In this case, those module parameters must linked to some python function defined in the compiler data python input file. 
    ```
    // @brainsmith derived_param <function> <param>
    ```
    Multiple params can be linked with one pragma like so:
    ```
    // @brainsmith derived_param my_python_fn PE SIMD TILE
    ```

3. *Weight*: Marks interface as a weight interface, informing HWCustomOp generation:
    ```
    // @brainsmith weight <signal_prefix>
    ```
    Multiple interfaces can be linked with one pragma like so:
    ```
    // @brainsmith weight in1 in2
    ```

4. *Custom Pragmas*: Give a clear way to add new pragmas, facilitating future extensions of Brainsmith and the HKG.
