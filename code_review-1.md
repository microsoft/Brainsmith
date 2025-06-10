# Code Review

## dataflow_interface.py
**Changes**
- DataflowDataType should use qonnx directly instead
- Some of the terminology gets confused. The original input (of shape qDim) should be referred to as a *query* not a *tensor*. Then, the chunks its split into (of shape tDim) are *tensors* not *tensor chunks*. stream_dims is for a single *stream* of data.
  **✅ RESOLVED**: Complete terminology migration implemented. Now using: tensor_dims (full tensor), block_dims (processing blocks), num_blocks (count).
- Because of the confusion between qDim and the original tensor shape (which are the same thing) reconstruct_tensor_shape and _compute_qDim_from_chunking are redundant.

**Questions**
- Line 148 the proper example would be [1,128,768] for bert with a batch 1 and seqlen of 128. How does this change how you should implement Dataflow Interface and attached logic?
- Line 171 why does qDim/tDim/stream_dims have to be the same length? Usually, qDim will be multi-dimensional, and tDim will be one maybe two dimensions
- Line 186 instead of setting default constraints, if there are no constraints assume *any* datatype is supported.


## dataflow_model.py
**Changes**
- Line 21 specify that L is cycle latency
!!!
- Line 70, 92 config (axi-lite) and control signals shouldn't be included in the dataflow model whatsoever since they run independent of the core datastreams
!!!
- Line 225 _calculate_cII is redundant with get_transfer_cycles in DataflowInterface. This is a responsibility of the DataflowInterface, so should be called from there.
- Line 158: the bottlenecking weight interface (max_weight) for each input should be saved as a part of the bottleneck analysis so human debuggers can view it
- Line 165: Max eII should be multiplied by num_tensors, not qDim. Perform a throrough sweep of the full dataflow model (all of brainsmith/dataflow) to ensure qDim and num_tensors are being used properly in their new definitions.
  **✅ RESOLVED**: Complete terminology migration and mathematical formula updates implemented throughout dataflow module.
- Line 233: _calculate_weight_cycles
- Line 342: The optimization engine this is a placeholder for is being created in parallel by another engineer. Remove this functino, and instead just ensure add a small readme explaining how best to hook DataflowModels into parallelism optimization engine.

**Questions**
- Line 165 why multiply the max eII by the number of elements? eII is already the number of cycles for the entire execution
- Line 233: I don't understand the formula for _calculate_weight_cycles, double check this is correct and explain it to me.


## interface_metadata.py
**Changes**
- Line 15: DataTypeConstraint seems redundant with the one in dataflow_interface.py. Audit these two files for redundancy and try to improve separation of concerns.

**Questions**
- What is the purpose of InterfaceMetadata and InterfaceMetadataCollection. Show me where they're used and justify their existence.

## tensor_chunking.py
**Changes**
- Line 80: _map_interface_to_input_index needs to be able to handle an arbitrary number of inputs, outputs, and weights/biases.
- Line 95: all deprecated code should be removed.
- Let's rework this file to be a framework to define a chunking strategy. Consider all the ways an input query could be split into a list of tensors based on only the input data shape (qDim) and layout (e.g. NCHW, NC, NLC, etc.), considering vectorwise, tiled, and more compute methods.
  **✅ RESOLVED**: Renamed to block_chunking.py with comprehensive strategy framework and updated terminology.

