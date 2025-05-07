import onnx
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np


class PerfAnalysis():
    
    def __init__(self, model_path:str)->None:
        self.model = onnx.load(model_path)
        self.matmul_nodes = [node for  node in self.model.graph.node if node.op_type == "MatMul"]

    def balance(self, balance_weight=1.5, max_total_resources=8643):    
        """    
        Optimize paths using single variable P per node, where P = PE * SIMD, prioritizing minimal cycle values with adjustable balance emphasis.    
        """          
        nodes_info = self.mmnode_info    
        node_names = list(nodes_info.keys())    
    
        # Single Variables    
        P = {n: cp.Variable(pos=True) for n in node_names}    
        resource = {n: cp.Variable(pos=True) for n in node_names}    
        cycle = {n: cp.Variable(pos=True) for n in node_names}    
        cycle_target = cp.Variable(pos=True)    
    
        constraints = []    
    
        for n in node_names:    
            constraints.append(P[n] >= 1)    
            constraints.append(P[n] <= 8192)  # Example upper bound    
            constraints.append(resource[n] >= 1)    
            constraints.append(cycle[n] >= 1)    
    
            # Cycle modeling    
            a_n, i_n, j_n = nodes_info[n]    
            constraints.append(P[n]*cycle[n] >= (a_n * i_n * j_n))    
    
            # Resource modeling (relax ceil)    
            constraints.append(resource[n] >= P[n]/3)    
    
        # Prioritize minimizing cycle times  
        cycle_minimization = cp.sum(list(cycle.values()))  
  
        mean_cycle = cp.sum(list(cycle.values())) / len(cycle)    
        cycle_penalty = cp.sum([    
            cp.abs(cycle[n] - mean_cycle)    
            for n in node_names    
        ]) * balance_weight  
    
        # Total resources  
         
        total_resources = cp.sum(list(resource.values()))    
        resource_penalty = max_total_resources - total_resources    
    
        constraints.append(total_resources <= max_total_resources)    
    
        # Objective: Prioritize cycle minimization
        objective = cp.Minimize(cycle_minimization + cycle_penalty)        
        problem = cp.Problem(objective, constraints)    
        problem.solve(solver=cp.SCIP, verbose=True, qcp=True)    
       
        results = {}    
        for n in node_names:    
            r = {}    
            r['DSPs'] = None    
            if P[n].value is not None:    
                r['DSPs'] = int(round(P[n].value/3))    
            r['cycles'] = None    
            if cycle[n].value is not None:    
                r['cycles'] = int(round(cycle[n].value))    
            results[n] = r    
        return results  


    def get_tensor_shape(self, tensor_name):
        for tensor in self.model.graph.initializer:
            if tensor.name == tensor_name:
                return list(tensor.dims)
        
        return None
    
    def get_tensor_shape_from_value_info(self, tensor_name):  
        """  
        Gets the shape of a tensor from the value information in inputs or outputs.  
        """  
        graph = self.model.graph
        for value_info in graph.input:  
            if value_info.name == tensor_name:  
                return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]  
      
        for value_info in graph.output:  
            if value_info.name == tensor_name:  
                return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]  
      
        for value_info in graph.value_info:  
            if value_info.name == tensor_name:  
                return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]  
      
        return 'Unknown'  
    
    def is_constant_input(self, node, input_name):  
        """  
        Check if an input is a constant from an initializer or from Constant nodes.  
        """  
        for tensor in self.model.graph.initializer:  
            if tensor.name == input_name:  
                return True  
      
        for n in self.model.graph.node:  
            if n.op_type == 'Constant' and n.output[0] == input_name:  
                return True  
        return False  
    
    def is_dynamic_matmul(self, mmnode)->bool:
        for input_name in mmnode.input:
            if self.is_constant_input(mmnode, input_name):
                return False
        return True
    
    
    def get_tensor_shape(self, tensor_name):
        for tensor in self.model.graph.initializer:
            if tensor.name == tensor_name:
                return list(tensor.dims)
        return None
    
        for input_name in mmnode.input:
            if is_constant_input(model, mmnode, input_name):
                return False
        return True


    @property
    def mmnode_info(self)->dict:
        """ 
            Get the a, i, j configuration values for each matmul to use with the constraint solver.
        """
        mmnode_info = {}
        for mm in self.matmul_nodes:
            if not self.is_dynamic_matmul(mm):
                a = 1
                i = 1
                j = 1
                for input_name in mm.input:
                    if self.is_constant_input(mm,input_name):
                        i,j = self.get_tensor_shape_from_value_info(input_name)
                    else:
                        input_s = self.get_tensor_shape_from_value_info(input_name)
                        a = int(np.prod(input_s[:-1]))
       
            else:
                shape0 = self.get_tensor_shape_from_value_info(mm.input[0])
                shape1 = self.get_tensor_shape_from_value_info(mm.input[1])
        
                i,j = (shape0[-2], shape1[-1])
                a = int(np.prod(shape1[:-1]))
            mmnode_info[mm.name] = (a,i,j)
        return mmnode_info
    
    @property
    def adj_list(self)->dict:
        graph = self.model.graph

        # Maps
        tensor_producer = {}   # tensor_name -> node_name
        node_outputs = {}      # node_name -> list of output tensor names
        full_graph = defaultdict(list)

        # Record outputs and producers
        for node in graph.node:
            for output in node.output:
                tensor_producer[output] = node.name
            node_outputs[node.name] = list(node.output)

        matmul_nodes = {node.name for node in graph.node if node.op_type == 'MatMul'}
        graph_inputs = {input.name for input in graph.input}
        graph_outputs = {output.name for output in graph.output}

        # Build full DAG across all nodes
        for node in graph.node:
            for input_tensor in node.input:
                producer = tensor_producer.get(input_tensor)
                if producer:
                    full_graph[producer].append(node.name)
                elif input_tensor in graph_inputs:
                    full_graph[input_tensor].append(node.name)

        # Build MatMul + Input + Output adjacency list
        relevant_nodes = matmul_nodes.union(graph_inputs)
        matmul_adjacency = defaultdict(list)

        for node in relevant_nodes:
            visited = set()
            queue = deque(full_graph.get(node, []))

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                if current in matmul_nodes:
                    matmul_adjacency[node].append(current)
                else:
                    queue.extend(full_graph.get(current, []))

        # --- ADD OUTPUT NODE ---
        OUTPUT_NODE = "__OUTPUT__"

        # Find MatMul nodes that feed into graph outputs
        for output_tensor in graph_outputs:
            producer_node = tensor_producer.get(output_tensor)
            if producer_node and producer_node in matmul_nodes:
                matmul_adjacency[producer_node].append(OUTPUT_NODE)

        return dict(matmul_adjacency)


    
    @property
    def print_adj_list(self):
        for src, dsts in self.adj_list.items():
            print(f"{src} -> {dsts}")
    
    def get_roots_and_leaves(self):
        all_nodes = set(self.adj_list.keys())
        all_targets = {target for targets in self.adj_list.values() for target in targets}
    
        roots = list(all_nodes - all_targets)
        leaves = [node for node in all_nodes if not self.adj_list.get(node)]
    
        return roots, leaves

    def dfs_paths(self, graph, current, path, all_paths):
        path = path + [current]
        if not graph.get(current):  # Leaf node
            all_paths.append(path)
        else:
            for neighbor in graph[current]:
                self.dfs_paths(graph, neighbor, path, all_paths)

    @property
    def paths(self)->list:
        roots, _ = self.get_roots_and_leaves()
        all_paths = []
        for root in roots:
            self.dfs_paths(self.adj_list, root, [], all_paths)
        return all_paths
    

    def draw_graph(self, start_node="global_in", figsize=(12, 8)):
        G = nx.DiGraph()

        # Add edges
        for src, dsts in self.adj_list.items():
            for dst in dsts:
                G.add_edge(src, dst)
                
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
        except ImportError:
            pos = nx.spring_layout(G)

        if start_node in pos:
            pos[start_node][0] = -1.0
            pos[start_node][1] = 0.0 
            
        # Draw the graph
        plt.figure(figsize=figsize)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
                node_size=2000, font_size=10, arrows=True)
        plt.show()



        
