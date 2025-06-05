# Interface-Based Dataflow Model

The *Dataflow Model* framework simplifies HW Kernels to a set of input, output, and weight AXI-Stream interfaces, abstracting the implementation to the relationship between these interfaces. This gives a simple but robust layer of abstraction between ONNX/PyTorch operators and their hardware implementation, simplifying design space exploration (DSE) and enabling automation for HW Kernel creation. Once implemented, the Dataflow Model will be integrated into the HW Kernel Generator to automatically generate the `HWCustomOp` class for each kernel, with most methods standardized at the `HWCustomOp` level and the concrete subclass usually only containing its interface information.

## Interface Types

* **Input** – AXI-Stream input, streaming activation data into the kernel.
* **Output** – AXI-Stream output, streaming activation data out of the kernel.
* **Weight** – AXI-Stream input, streaming weight data into the kernel. 
* **Config** – AXI-Lite input/output/inout, streaming configuration data into the kernel.
* **Control** – Input control signals: clk, rst, and (optional) clk2x.

*Config* and *Control* interfaces have no impact on the Dataflow Model, but if they exist their names and information must be added to the HWCustomOp as attributes. The *Input*, *Output*, and *Weight* interfaces stream data in and out of the kernel, and thus constitute the Dataflow Model.

## Data and Execution Hierarchy

The data streamed through each interface is described using the following data hierarchy:

* **Query** – The data of the entire hidden state or weight streamed through the interface. For an Input/Output inteface, the complete processing of a query is called an *inference*. For a Weight interface, the complete processing of a query is called an *execution*.
* **Tensor** – The data streamed through the interface for a single *calculation* in the kernel. This has different connotations depending on the interface type:
    - For inputs, this is the smallest amount of data required to stimulate a calculation in the kernel and produce an output. For kernels with multiple calculations per input tensor, completing all calclulations with a tensor is called an *execution*. The kernel cannot begin processing a new tensor until the previous execution is complete.
    - For weights, this is the vector/slice/tile used in a calculation (e.g. a single vector of the weight Query in the MVAU, or the entire weight for a LayerNorm). 
    - For outputs, this is the amount of data produced by a single calculation, and the smallest amount of data that can be outputted to the next kernel at a time.
* **Stream** – The data streamed through the interface each clock cycle.
* **Element** – A single value, bit width defined by the interface’s datatype.

Constraints and restrictions:
- Each data level's shape must tile into the next (i.e. stream tiles into tensor, tensor tiles into query) for each interface. Some parameters for each interface are universal for all instances of the kernel, but most are defined at runtime based on the ONNX pattern and design space exploration (DSE) results.

## Interface Parameters

For a fully instantiated kernel, each input interface has a set of parameters that define its shape and behavior.

| Symbol  | Name              | Definition                                                                |
|---------|-------------------|---------------------------------------------------------------------------|
| $qDim$  | Query Dimensions  | Shape of the query, or number of tensors streamed per inference/execution |
| $tDim$  | Tensor Dimensions | Shape of the tensor passed through the interface for a calculation        |
| $sDim$  | Stream Dimensions | Shape of the stream passed through the interface each clock cycle         |
| $dtype$ | Data Type         | Data type of each element streamed through the interface                  |

When discussing multiple interfaces, each variable is subscripted as $qDim_{I|W|O,i}$ for the $i^{th}$ interface of type Input, Weight, or Output, respectively.

## Computational Model

When implementing a Kernel, the Dataflow Model abstracts the kernel's implementation to a set of interfaces and their relationships, with the core assumption that each **Input tensor** performs a calculation with a **Weight tensor** from every Weight interface to produce an **Output tensor**. The number of cycles to complete that calculation is the basis for the kernel's computational model:

| Symbol | Name                            | Definition                                                                          |
|--------|---------------------------------|-------------------------------------------------------------------------------------|
| $cII$  | Calculation Initiation Interval | Num cycles for one calculation (processing an Input tensor against a Weight tensor) |
| $eII$  | Execution Initiation Interval   | Num cycles for one execution (processing an Input tensor against a  Weight query)   |
| $L$    | Inference Cycle Latency         | Num cycles for one Inference (processing an Input query)                            |

Although the number of interfaces in a kernel is constant, the parameters of each interface are defined at runtime. The $qDim$, $tDim$, and $dtype$ parameters are defined based on the ONNX pattern of the target model, while the $sDim$ (stream dimensions) is determined based on the results of the compiler's DSE. The compiler explores this design space via *Parllelism Paramaters*:

- **$iPar$**: Input parallelism. This corresponds to *SIMD* in the previous parallelism model. $1 \leq iPar \leq tDim_I$.
- **$wPar$**: Weight parallelism. This corresponds to *PE* in the previous parallelism model. $1 \leq wPar \leq qDim_W$.

Unlike the interface parameters which merely describe the Kernel, the parallelism parameters are used in final code generation to define the architecture of the kernel instance.

$batch$ is another important parameter, but is separated from the abstraction layer due to its influence on the model's architecture, not just the kernel. If the model is batched (and the Kernel supports $batching$), then the following parameters are directly scaled by $batch$:
- $tDim_I = tDim_I * batch$
- $tDim_O = tDim_O * batch$
- $sDim_I = sDim_I * batch$
- $sDim_O = sDim_O * batch$

### Simple case

For a simple kernel with only one of each interface type, the Dataflow Model defines the following relationships:

- $sDim_I = iPar$
- $sDim_W = wPar * iPar * (tDim_W / tDim_I)$
- $sDim_O = sDim_I * (tDim_O / tDim_I)$
- $cII = \prod(tDim_I / sDim_I)$
- $eII = cII * \prod(qDim_W / wPar)$
- $L = eII * \prod(qDim_I)$

### Arbitary number of interfaces

For a kernel with multiple Input interfaces, there exists a different $iPar$ for each $i \in X$ Inputs:

- $sDim_{I,i} = iPar_i$
- $cII_i = * \prod(tDim_{I,i} / sDim_{I,i})$

For a kernel with multiple Weight interfaces, there exists a different $wPar$ for each $j \in Y$ Weights. For each combination of Input $i$ and Weight $j$, $sDim_W$ and $eII$ are calculated and set to the bottleneck calculation:

- $sDim_{W,j} = \max_i^X{(wPar_j * iPar_i * (tDim_{W,j} / tDim_{I,i}))}$
- $eII_i = cII_i * \max_j^Y{(\prod(qDim_{W,j} / wPar_j))}$
- $L = \max_i^X{(eII_i)}$

For a kernel with multiple Inputs, the projected $sDim_O$ for each $k \in Z$ Outputs is calculated based on the Input $b$ with the largest execution initiation interval ($b = \arg\max_i^X{(eII_i)}$):

- $sDim_{O,k} = sDim_{I,b} * (tDim_{O,k} / tDim_{I,b})$

This "worst case calculation" is a simplification that may not be the case for all Kernels, but is sufficiently accurate for this level of abstraction. If for some reason a kernel diverges significantly, the initiation interval functions can all be overwritten.
