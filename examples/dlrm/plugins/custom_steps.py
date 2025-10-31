############################################################################
# @author       Joshua Monson <joshmonson@microsoft.com>
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
BERT-Specific Custom Build Steps

Custom steps specifically for BERT model processing, including:
- Head and tail removal for model decomposition
- Metadata extraction for shell integration
- Reference I/O generation for validation

These steps are highly specific to BERT model architecture and
are not general-purpose FINN dataflow compilation steps.
"""

from brainsmith.registry import step
from finn.util import onnxscript_helpers as oxh
from qonnx.core.modelwrapper import ModelWrapper
from onnxscript import ir
from .import reachablefrominputx as reachable
import onnx


@step(name="split_sparse_processing")
def split_sparse_processing(model, cfg):
    """Separate the sparse processing parts of the model into their own graph
       because they will not go on the FPGA.
    """
    transform = reachable.ReachableFromInputTransform()
    transform.apply(model)

    # Get the IR model from the transform (same instance used for analysis)
    model_ir = transform.ir_model

    # Get reachable nodes for each input (these are IR node objects from model_ir)
    dense_x_nodes = transform.get_reachable_nodes("dense_x")
    indices_0_nodes = transform.get_reachable_nodes("indices_0")
    indices_1_nodes = transform.get_reachable_nodes("indices_1")
    indices_2_nodes = transform.get_reachable_nodes("indices_2")
    offsets_nodes = transform.get_reachable_nodes("offsets")

    # All nodes reachable from sparse inputs
    all_sparse = indices_0_nodes | indices_1_nodes | indices_2_nodes | offsets_nodes

    # Dense nodes: reachable from dense_x (includes overlapping nodes)
    dense_nodes = dense_x_nodes

    # Sparse nodes: ONLY reachable from sparse inputs, NOT from dense_x
    sparse_nodes = all_sparse - dense_x_nodes

    # Check for missing nodes (not reachable from any input)
    all_graph_nodes = set(model_ir.graph)
    all_partitioned_nodes = sparse_nodes | dense_nodes
    missing_nodes = all_graph_nodes - all_partitioned_nodes

    # Assign missing nodes to the partition where their consumers are
    # (these are typically constant/empty initializer nodes with no inputs)
    for missing_node in missing_nodes:
        # Check which partition(s) consume this node's outputs
        sparse_consumers = 0
        dense_consumers = 0
        for output in missing_node.outputs:
            for consumer in output.consumers():
                if consumer in sparse_nodes:
                    sparse_consumers += 1
                elif consumer in dense_nodes:
                    dense_consumers += 1

        # Assign to the partition with more consumers (or sparse if tied/no consumers)
        if dense_consumers > sparse_consumers:
            dense_nodes.add(missing_node)
        else:
            sparse_nodes.add(missing_node)

    # Recompute coverage after assigning missing nodes
    all_partitioned_nodes = sparse_nodes | dense_nodes
    remaining_missing = all_graph_nodes - all_partitioned_nodes

    print(f"\n=== Graph Partitioning Summary ===")
    print(f"Total nodes in graph: {len(all_graph_nodes)}")
    print(f"Sparse nodes: {len(sparse_nodes)} (includes {len(missing_nodes)} non-input nodes)")
    print(f"Dense nodes: {len(dense_nodes)}")
    print(f"Overlap nodes (going to FPGA): {len(dense_x_nodes & all_sparse)}")
    print(f"Accounted for: {len(all_partitioned_nodes)}")
    print(f"Remaining missing: {len(remaining_missing)}")

    if missing_nodes:
        print(f"\n=== Originally Missing Nodes (now assigned) ===")
        for node in missing_nodes:
            partition = "SPARSE" if node in sparse_nodes else "DENSE"
            print(f"  - {node.name} ({node.op_type}) -> {partition}")

    if remaining_missing:
        print(f"\n=== Still Missing Nodes (could not assign) ===")
        for node in remaining_missing:
            print(f"  - {node.name} ({node.op_type})")
            print(f"    Inputs: {[inp.name if inp else 'None' for inp in node.inputs]}")
            print(f"    Outputs: {[out.name for out in node.outputs]}")
            # Check if outputs are consumed and which partition consumers are in
            consumers = []
            consumer_partitions = []
            for out in node.outputs:
                for consumer in out.consumers():
                    consumers.append(consumer.name)
                    if consumer in sparse_nodes:
                        consumer_partitions.append(f"{consumer.name} (SPARSE)")
                    elif consumer in dense_nodes:
                        consumer_partitions.append(f"{consumer.name} (DENSE)")
                    else:
                        consumer_partitions.append(f"{consumer.name} (UNKNOWN)")
            print(f"    Consumers: {consumers if consumers else 'None (unused)'}")
            print(f"    Consumer partitions: {consumer_partitions}")

    print(f"\nSample sparse node names: {[n.name for n in list(sparse_nodes)[:5]]}")
    print(f"Sample dense node names: {[n.name for n in list(dense_nodes)[:5]]}")

    model_ir.graph.sort()

    dense_subgraph = oxh.SubGraphView(model_ir.graph, 'dense', dense_nodes, include_initializers=True)
    sparse_subgraph = oxh.SubGraphView(model_ir.graph, 'sparse', sparse_nodes, include_initializers=True)

    dense_model = ir.Model(dense_subgraph, ir_version=model_ir.ir_version)
    sparse_model = ir.Model(sparse_subgraph, ir_version=model_ir.ir_version)



    sparse_proto = ir.serde.serialize_model(sparse_model)
    dense_proto = ir.serde.serialize_model(dense_model)
    onnx.save(sparse_proto, "sparse.onnx")
    onnx.save(dense_proto, "dense.onnx")
    return ModelWrapper(dense_proto)

@step(name="dense_cleanup")
def dense_cleanup(model: ModelWrapper, cfg):

    model = model.cleanup()
    # Optimize the dense model using ONNX Graph Optimization Toolkit
    import onnxsim
    model, check = onnxsim.simplify(model.model)
    if not check:
        raise RuntimeError("Unable to simplify the DLRM datapath")
    return ModelWrapper(model)
