############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for DataflowGraph class"""

import pytest
from brainsmith.core.dataflow.core.types import InterfaceDirection, INT16, INT32
from brainsmith.core.dataflow.core.interface import Interface
from brainsmith.core.dataflow.core.kernel import Kernel
from brainsmith.core.dataflow.core.graph import DataflowGraph, DataflowEdge


class TestDataflowEdge:
    """Test DataflowEdge class"""
    
    def test_edge_creation(self):
        """Test basic edge creation"""
        edge = DataflowEdge(
            producer_kernel="k1",
            producer_intf="out",
            consumer_kernel="k2",
            consumer_intf="in",
            buffer_depth=1024
        )
        
        assert edge.producer_kernel == "k1"
        assert edge.producer_intf == "out"
        assert edge.consumer_kernel == "k2"
        assert edge.consumer_intf == "in"
        assert edge.buffer_depth == 1024
        assert edge.id == "k1.out->k2.in"
    
    def test_self_loop_validation(self):
        """Test that self-loops are rejected"""
        with pytest.raises(ValueError, match="Cannot connect interface to itself"):
            DataflowEdge("k1", "data", "k1", "data")


class TestDataflowGraphConstruction:
    """Test graph construction and kernel management"""
    
    def test_empty_graph(self):
        """Test empty graph creation"""
        graph = DataflowGraph()
        assert len(graph.kernels) == 0
        assert len(graph.edges) == 0
        assert graph.topological_sort() == []
    
    def test_add_kernels(self):
        """Test adding kernels to graph"""
        k1 = Kernel(name="k1")
        k2 = Kernel(name="k2")
        
        graph = DataflowGraph([k1, k2])
        
        assert len(graph.kernels) == 2
        assert "k1" in graph.kernels
        assert "k2" in graph.kernels
        
        # Try to add duplicate
        with pytest.raises(ValueError, match="already exists"):
            graph.add_kernel(k1)
    
    def test_add_edges(self):
        """Test adding edges between kernels"""
        # Create kernels with compatible interfaces
        k1 = Kernel(
            name="producer",
            interfaces=[
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ]
        )
        
        k2 = Kernel(
            name="consumer",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,))
            ]
        )
        
        graph = DataflowGraph([k1, k2])
        
        # Add valid edge
        edge = graph.add_edge("producer", "out", "consumer", "in", buffer_depth=512)
        
        assert edge.id in graph.edges
        assert edge.buffer_depth == 512
        assert len(graph.edges) == 1
    
    def test_edge_validation(self):
        """Test edge validation rules"""
        # Setup kernels
        k1 = Kernel(
            name="k1",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ]
        )
        
        k2 = Kernel(
            name="k2",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ]
        )
        
        graph = DataflowGraph([k1, k2])
        
        # Invalid: non-existent kernel
        with pytest.raises(ValueError, match="Producer kernel 'k3' not found"):
            graph.add_edge("k3", "out", "k2", "in")
        
        # Invalid: non-existent interface
        with pytest.raises(ValueError, match="Interface 'missing' not found"):
            graph.add_edge("k1", "missing", "k2", "in")
        
        # Invalid: wrong direction (input to input)
        with pytest.raises(ValueError, match="Producer interface must be OUTPUT"):
            graph.add_edge("k1", "in", "k2", "in")
        
        # Invalid: wrong direction (output to output)
        with pytest.raises(ValueError, match="Consumer interface must be INPUT or WEIGHT"):
            graph.add_edge("k1", "out", "k2", "out")
        
        # Valid edge
        graph.add_edge("k1", "out", "k2", "in")
        
        # Invalid: duplicate edge
        with pytest.raises(ValueError, match="Edge already exists"):
            graph.add_edge("k1", "out", "k2", "in")
        
        # Invalid: interface already connected
        k3 = Kernel(
            name="k3",
            interfaces=[
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ]
        )
        graph.add_kernel(k3)
        
        with pytest.raises(ValueError, match="already has an incoming connection"):
            graph.add_edge("k3", "out", "k2", "in")


class TestDataflowGraphAnalysis:
    """Test graph analysis methods"""
    
    def setup_method(self):
        """Create test graph: k1 -> k2 -> k3
                              k1 -> k3
        """
        self.k1 = Kernel(
            name="k1",
            interfaces=[
                Interface("out1", InterfaceDirection.OUTPUT, INT16, (64,), (64,)),
                Interface("out2", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ],
            latency_cycles=(100, 80)
        )
        
        self.k2 = Kernel(
            name="k2",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ],
            latency_cycles=(50, 40)
        )
        
        self.k3 = Kernel(
            name="k3",
            interfaces=[
                Interface("in1", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("in2", InterfaceDirection.INPUT, INT16, (64,), (64,))
            ],
            latency_cycles=(75, 60)
        )
        
        self.graph = DataflowGraph([self.k1, self.k2, self.k3])
        self.graph.add_edge("k1", "out1", "k2", "in")
        self.graph.add_edge("k2", "out", "k3", "in1")
        self.graph.add_edge("k1", "out2", "k3", "in2")
    
    def test_topological_sort(self):
        """Test topological sorting"""
        order = self.graph.topological_sort()
        
        # k1 must come before k2 and k3
        assert order.index("k1") < order.index("k2")
        assert order.index("k1") < order.index("k3")
        
        # k2 must come before k3
        assert order.index("k2") < order.index("k3")
    
    def test_sources_and_sinks(self):
        """Test finding source and sink nodes"""
        sources = self.graph.get_sources()
        sinks = self.graph.get_sinks()
        
        assert sources == ["k1"]
        assert sinks == ["k3"]
    
    def test_kernel_edges(self):
        """Test getting edges for a kernel"""
        # k1: only outgoing
        incoming, outgoing = self.graph.get_kernel_edges("k1")
        assert len(incoming) == 0
        assert len(outgoing) == 2
        
        # k2: one incoming, one outgoing
        incoming, outgoing = self.graph.get_kernel_edges("k2")
        assert len(incoming) == 1
        assert len(outgoing) == 1
        assert incoming[0].producer_kernel == "k1"
        assert outgoing[0].consumer_kernel == "k3"
        
        # k3: only incoming
        incoming, outgoing = self.graph.get_kernel_edges("k3")
        assert len(incoming) == 2
        assert len(outgoing) == 0
    
    def test_critical_path(self):
        """Test critical path analysis"""
        path, latency = self.graph.get_critical_path()
        
        # Path should be k1 -> k2 -> k3 (100 + 50 + 75 = 225)
        # Direct path k1 -> k3 is only 100 + 75 = 175
        assert path == ["k1", "k2", "k3"]
        assert latency == 225
    
    def test_subgraph_extraction(self):
        """Test extracting subgraph"""
        # Extract k1 and k2 only
        subgraph = self.graph.get_subgraph(["k1", "k2"])
        
        assert len(subgraph.kernels) == 2
        assert "k1" in subgraph.kernels
        assert "k2" in subgraph.kernels
        assert "k3" not in subgraph.kernels
        
        # Should have only the edge between k1 and k2
        assert len(subgraph.edges) == 1
        edge = list(subgraph.edges.values())[0]
        assert edge.producer_kernel == "k1"
        assert edge.consumer_kernel == "k2"


class TestDataflowGraphValidation:
    """Test graph validation"""
    
    def test_cycle_detection(self):
        """Test detection of cycles in graph"""
        k1 = Kernel(
            name="k1",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ]
        )
        
        k2 = Kernel(
            name="k2",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ]
        )
        
        graph = DataflowGraph([k1, k2])
        graph.add_edge("k1", "out", "k2", "in")
        graph.add_edge("k2", "out", "k1", "in")  # Creates cycle
        
        with pytest.raises(ValueError, match="Graph contains cycles"):
            graph.validate()
    
    def test_missing_connections(self):
        """Test detection of missing required connections"""
        k1 = Kernel(
            name="k1",
            interfaces=[
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ]
        )
        
        k2 = Kernel(
            name="k2",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,)),  # Required
                Interface("optional_in", InterfaceDirection.INPUT, INT16, (64,), (64,), optional=True)
            ]
        )
        
        graph = DataflowGraph([k1, k2])
        # Don't connect required input
        
        with pytest.raises(ValueError, match="Required input interface 'in'.*not connected"):
            graph.validate()
        
        # Connect it and validation should pass
        graph.add_edge("k1", "out", "k2", "in")
        graph.validate()  # Should not raise
    
    def test_interface_compatibility(self):
        """Test interface compatibility checking"""
        k1 = Kernel(
            name="k1",
            interfaces=[
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ]
        )
        
        k2 = Kernel(
            name="k2",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT32, (64,), (64,))  # Different dtype
            ]
        )
        
        graph = DataflowGraph([k1, k2])
        
        with pytest.raises(ValueError, match="Data type mismatch"):
            graph.add_edge("k1", "out", "k2", "in")


class TestDataflowGraphSerialization:
    """Test graph serialization and visualization"""
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        k1 = Kernel(name="k1", latency_cycles=(100, 80))
        k2 = Kernel(name="k2", latency_cycles=(50, 40))
        
        k1.interfaces.append(
            Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
        )
        k2.interfaces.append(
            Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,))
        )
        
        graph = DataflowGraph([k1, k2])
        graph.add_edge("k1", "out", "k2", "in", buffer_depth=256)
        
        data = graph.to_dict()
        
        assert "kernels" in data
        assert "edges" in data
        assert len(data["kernels"]) == 2
        assert len(data["edges"]) == 1
        
        assert data["kernels"]["k1"]["latency"] == (100, 80)
        assert data["edges"][0]["from"] == "k1.out"
        assert data["edges"][0]["to"] == "k2.in"
        assert data["edges"][0]["buffer_depth"] == 256
    
    def test_visualization(self):
        """Test text visualization"""
        k1 = Kernel(name="source")
        k2 = Kernel(name="sink")
        
        k1.interfaces.append(
            Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
        )
        k2.interfaces.append(
            Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,))
        )
        
        graph = DataflowGraph([k1, k2])
        graph.add_edge("source", "out", "sink", "in")
        
        viz = graph.visualize()
        
        assert "Dataflow Graph:" in viz
        assert "Kernels: 2" in viz
        assert "Edges: 1" in viz
        assert "source" in viz
        assert "sink" in viz
        assert "source.out -> sink.in" in viz  # Space around arrow


class TestComplexGraphScenarios:
    """Test complex graph scenarios"""
    
    def test_multi_input_multi_output(self):
        """Test kernel with multiple inputs and outputs"""
        # Merge kernel: takes two inputs, produces one output
        merge = Kernel(
            name="merge",
            interfaces=[
                Interface("in1", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("in2", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (128,), (128,))
            ]
        )
        
        # Split kernel: takes one input, produces two outputs
        split = Kernel(
            name="split",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (128,), (128,)),
                Interface("out1", InterfaceDirection.OUTPUT, INT16, (64,), (64,)),
                Interface("out2", InterfaceDirection.OUTPUT, INT16, (64,), (64,))
            ]
        )
        
        graph = DataflowGraph([merge, split])
        graph.add_edge("merge", "out", "split", "in")
        
        # Create full diamond pattern
        src1 = Kernel(
            name="src1",
            interfaces=[Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))]
        )
        src2 = Kernel(
            name="src2",
            interfaces=[Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (64,))]
        )
        
        graph.add_kernel(src1)
        graph.add_kernel(src2)
        
        graph.add_edge("src1", "out", "merge", "in1")
        graph.add_edge("src2", "out", "merge", "in2")
        
        # Validate structure
        graph.validate()
        
        # Check topology
        order = graph.topological_sort()
        merge_idx = order.index("merge")
        split_idx = order.index("split")
        
        assert order.index("src1") < merge_idx
        assert order.index("src2") < merge_idx
        assert merge_idx < split_idx