############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Dataflow graph representation"""

import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from .kernel import Kernel
from .interface import Interface
from .types import InterfaceDirection


@dataclass
class DataflowEdge:
    """Edge between kernel interfaces
    
    Represents a connection from one kernel's output interface
    to another kernel's input interface.
    """
    producer_kernel: str
    producer_intf: str
    consumer_kernel: str
    consumer_intf: str
    buffer_depth: Optional[int] = None  # Will be computed by scheduler
    
    def __post_init__(self):
        """Validate edge"""
        if self.producer_kernel == self.consumer_kernel:
            if self.producer_intf == self.consumer_intf:
                raise ValueError("Cannot connect interface to itself")
    
    @property
    def id(self) -> str:
        """Unique edge identifier"""
        return f"{self.producer_kernel}.{self.producer_intf}->{self.consumer_kernel}.{self.consumer_intf}"
    
    def __repr__(self) -> str:
        return f"Edge({self.producer_kernel}.{self.producer_intf} -> {self.consumer_kernel}.{self.consumer_intf})"


@dataclass
class DataflowGraph:
    """Graph of connected kernels
    
    Manages a directed acyclic graph (DAG) of kernels connected
    through their interfaces. Provides validation, analysis, and
    traversal capabilities.
    """
    
    def __init__(self, kernels: Optional[List[Kernel]] = None):
        """Initialize graph with kernels
        
        Args:
            kernels: List of kernels to add to graph
        """
        self.graph = nx.DiGraph()
        self.kernels: Dict[str, Kernel] = {}
        self.edges: Dict[str, DataflowEdge] = {}
        
        if kernels:
            for kernel in kernels:
                self.add_kernel(kernel)
    
    def add_kernel(self, kernel: Kernel) -> None:
        """Add kernel to graph
        
        Raises:
            ValueError: If kernel name already exists
        """
        if kernel.name in self.kernels:
            raise ValueError(f"Kernel '{kernel.name}' already exists in graph")
        
        self.kernels[kernel.name] = kernel
        self.graph.add_node(kernel.name, kernel=kernel)
    
    def add_edge(self, producer_kernel: str, producer_intf: str,
                 consumer_kernel: str, consumer_intf: str,
                 buffer_depth: Optional[int] = None) -> DataflowEdge:
        """Add connection between kernels
        
        Args:
            producer_kernel: Name of producing kernel
            producer_intf: Name of output interface on producer
            consumer_kernel: Name of consuming kernel
            consumer_intf: Name of input interface on consumer
            buffer_depth: Optional buffer size (computed later if None)
            
        Returns:
            Created edge
            
        Raises:
            ValueError: If connection is invalid
        """
        # Validate kernels exist
        if producer_kernel not in self.kernels:
            raise ValueError(f"Producer kernel '{producer_kernel}' not found")
        if consumer_kernel not in self.kernels:
            raise ValueError(f"Consumer kernel '{consumer_kernel}' not found")
        
        # Get kernels and interfaces
        prod_kernel = self.kernels[producer_kernel]
        cons_kernel = self.kernels[consumer_kernel]
        
        try:
            prod_intf = prod_kernel.get_interface(producer_intf)
        except KeyError:
            raise ValueError(f"Interface '{producer_intf}' not found in kernel '{producer_kernel}'")
        
        try:
            cons_intf = cons_kernel.get_interface(consumer_intf)
        except KeyError:
            raise ValueError(f"Interface '{consumer_intf}' not found in kernel '{consumer_kernel}'")
        
        # Validate interface directions
        if prod_intf.direction != InterfaceDirection.OUTPUT:
            raise ValueError(f"Producer interface must be OUTPUT, got {prod_intf.direction.value}")
        
        if cons_intf.direction not in [InterfaceDirection.INPUT, InterfaceDirection.WEIGHT]:
            raise ValueError(f"Consumer interface must be INPUT or WEIGHT, got {cons_intf.direction.value}")
        
        # Validate connection compatibility
        prod_intf.validate_connection(cons_intf)
        
        # Check for duplicate edges
        edge = DataflowEdge(producer_kernel, producer_intf,
                           consumer_kernel, consumer_intf, buffer_depth)
        
        if edge.id in self.edges:
            raise ValueError(f"Edge already exists: {edge.id}")
        
        # Check if consumer interface already has a connection
        for existing_edge in self.edges.values():
            if (existing_edge.consumer_kernel == consumer_kernel and
                existing_edge.consumer_intf == consumer_intf):
                raise ValueError(
                    f"Interface '{consumer_intf}' on kernel '{consumer_kernel}' "
                    f"already has an incoming connection"
                )
        
        # Add to graph
        self.edges[edge.id] = edge
        self.graph.add_edge(
            producer_kernel, consumer_kernel,
            edge=edge,
            producer_intf=producer_intf,
            consumer_intf=consumer_intf
        )
        
        return edge
    
    def remove_edge(self, edge_id: str) -> None:
        """Remove edge from graph"""
        if edge_id not in self.edges:
            raise ValueError(f"Edge '{edge_id}' not found")
        
        edge = self.edges[edge_id]
        self.graph.remove_edge(edge.producer_kernel, edge.consumer_kernel)
        del self.edges[edge_id]
    
    def get_edge(self, producer_kernel: str, producer_intf: str,
                 consumer_kernel: str, consumer_intf: str) -> Optional[DataflowEdge]:
        """Get edge by endpoints"""
        edge_id = f"{producer_kernel}.{producer_intf}->{consumer_kernel}.{consumer_intf}"
        return self.edges.get(edge_id)
    
    def get_kernel_edges(self, kernel_name: str) -> Tuple[List[DataflowEdge], List[DataflowEdge]]:
        """Get all edges connected to a kernel
        
        Returns:
            Tuple of (incoming_edges, outgoing_edges)
        """
        incoming = []
        outgoing = []
        
        for edge in self.edges.values():
            if edge.consumer_kernel == kernel_name:
                incoming.append(edge)
            if edge.producer_kernel == kernel_name:
                outgoing.append(edge)
        
        return incoming, outgoing
    
    def validate(self) -> None:
        """Validate graph structure and connections
        
        Checks:
        - Graph is a DAG (no cycles)
        - All interfaces have valid connections
        - All kernel pragmas are satisfied
        
        Raises:
            ValueError: If validation fails
        """
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Graph contains cycles: {cycles}")
        
        # Validate each kernel
        for kernel in self.kernels.values():
            kernel.validate()
        
        # Check all required interfaces are connected
        for kernel_name, kernel in self.kernels.items():
            incoming, outgoing = self.get_kernel_edges(kernel_name)
            
            # Get connected interface names
            connected_inputs = {e.consumer_intf for e in incoming}
            connected_outputs = {e.producer_intf for e in outgoing}
            
            # Check all non-optional inputs are connected
            for intf in kernel.input_interfaces:
                if not intf.optional and intf.name not in connected_inputs:
                    raise ValueError(
                        f"Required input interface '{intf.name}' on kernel '{kernel_name}' "
                        f"is not connected"
                    )
            
            # Weights typically must be connected
            for intf in kernel.weight_interfaces:
                if not intf.optional and intf.name not in connected_inputs:
                    raise ValueError(
                        f"Weight interface '{intf.name}' on kernel '{kernel_name}' "
                        f"is not connected"
                    )
    
    def topological_sort(self) -> List[str]:
        """Get topological ordering of kernels
        
        Returns:
            List of kernel names in topological order
        """
        return list(nx.topological_sort(self.graph))
    
    def get_sources(self) -> List[str]:
        """Get source kernels (no incoming edges)"""
        return [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
    
    def get_sinks(self) -> List[str]:
        """Get sink kernels (no outgoing edges)"""
        return [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
    
    def get_critical_path(self) -> Tuple[List[str], int]:
        """Compute critical path through graph
        
        Returns:
            Tuple of (kernel_names, total_latency)
        """
        # Use worst-case latencies as weights
        for kernel_name, kernel in self.kernels.items():
            self.graph.nodes[kernel_name]['weight'] = kernel.latency_cycles[0]
        
        # Find longest path (critical path)
        if not self.kernels:
            return [], 0
        
        # For each source, find longest path to any sink
        sources = self.get_sources()
        sinks = self.get_sinks()
        
        longest_path = []
        max_length = 0
        
        for source in sources:
            for sink in sinks:
                try:
                    # Find all simple paths
                    paths = nx.all_simple_paths(self.graph, source, sink)
                    for path in paths:
                        # Calculate path length
                        length = sum(self.kernels[node].latency_cycles[0] for node in path)
                        if length > max_length:
                            max_length = length
                            longest_path = path
                except nx.NetworkXNoPath:
                    continue
        
        return longest_path, max_length
    
    def get_subgraph(self, kernel_names: List[str]) -> "DataflowGraph":
        """Extract subgraph containing specified kernels"""
        subgraph = DataflowGraph()
        
        # Add kernels
        for name in kernel_names:
            if name in self.kernels:
                subgraph.add_kernel(self.kernels[name])
        
        # Add edges between selected kernels
        for edge in self.edges.values():
            if (edge.producer_kernel in kernel_names and 
                edge.consumer_kernel in kernel_names):
                subgraph.edges[edge.id] = edge
                subgraph.graph.add_edge(
                    edge.producer_kernel, edge.consumer_kernel,
                    edge=edge,
                    producer_intf=edge.producer_intf,
                    consumer_intf=edge.consumer_intf
                )
        
        return subgraph
    
    def to_dict(self) -> dict:
        """Convert graph to dictionary representation"""
        return {
            "kernels": {name: {
                "hw_module": k.hw_module,
                "interfaces": len(k.interfaces),
                "latency": k.latency_cycles,
                "pragmas": len(k.pragmas)
            } for name, k in self.kernels.items()},
            "edges": [{
                "from": f"{e.producer_kernel}.{e.producer_intf}",
                "to": f"{e.consumer_kernel}.{e.consumer_intf}",
                "buffer_depth": e.buffer_depth
            } for e in self.edges.values()]
        }
    
    def visualize(self) -> str:
        """Generate simple text visualization of graph"""
        lines = ["Dataflow Graph:"]
        lines.append(f"  Kernels: {len(self.kernels)}")
        lines.append(f"  Edges: {len(self.edges)}")
        lines.append("")
        
        # Show kernels
        lines.append("Kernels:")
        for name, kernel in self.kernels.items():
            lines.append(f"  {name}: {kernel}")
        
        lines.append("")
        
        # Show edges
        lines.append("Connections:")
        for edge in self.edges.values():
            depth_str = f" [buffer={edge.buffer_depth}]" if edge.buffer_depth else ""
            lines.append(f"  {edge}{depth_str}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"DataflowGraph(kernels={len(self.kernels)}, edges={len(self.edges)})"