
from typing import List, Dict, Set
from .node import Node, Data
import torch

class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.nodes_by_name: Dict[str, List[Node]] = {}  # Maps base names to list of nodes
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        
    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        # Extract base name without ID
        base_name = node.name.rsplit('_', 1)[0]
        if base_name not in self.nodes_by_name:
            self.nodes_by_name[base_name] = []
        self.nodes_by_name[base_name].append(node)
        
    def connect(self, from_node: Node, to_node: Node) -> None:
        from_node.outputs.append(to_node)
        to_node.inputs.append(from_node)
    
    def clear_tensors(self) -> None:
        """Clear all stored tensors in the graph"""
        for node in self.nodes:
            node.clear_tensor()

    def clear_shapes(self) -> None:
        """Clear all stored shapes in the graph"""
        for node in self.nodes:
            node.clear_shape()
            
    def infer_shapes(self) -> None:
        """Infer shapes for all nodes in the graph"""
        
        # Clear all existing shapes first
        self.clear_shapes()
        
        # Start from input nodes
        current_nodes = self.input_nodes.copy()
        
        # Keep track of nodes we've processed
        processed: Set[Node] = set()
        
        while current_nodes:
            # Get next node to process
            node = current_nodes.pop(0)
            
            if node in processed:
                continue
            
            # Infer shape for this node
            node.get_shape()
            
            processed.add(node)
            
            # add all it's inputs to processed
            processed.update(node.inputs)
            
            # Add outputs to the list for processing
            current_nodes.extend(node.outputs)
            
            
        # Update nodes list and nodes_by_name to only include processed nodes
        # As there might be redundant nodes in the graph that we don't need for
        # computation
        self.nodes = list(processed)
        
        # Rebuild nodes_by_name dictionary
        self.nodes_by_name.clear()
        for node in self.nodes:
            base_name = node.name.rsplit('_', 1)[0]
            if base_name not in self.nodes_by_name:
                self.nodes_by_name[base_name] = []
            self.nodes_by_name[base_name].append(node)
            
    def forward(self, inputs: Dict[str, torch.Tensor] = {}) -> Dict[str, torch.Tensor]:
        """
        Perform forward pass through the graph.
        """
        # Clear previous tensors
        self.clear_tensors()
        
        # Set input tensors if provided
        if inputs:
            for name, tensor in inputs.items():
                # Find the input node with this name
                input_node = next((node for node in self.input_nodes if node.name == name), None)
                if input_node is None:
                    raise ValueError(f"Input node '{name}' not found")
                
                # Update the node's tensor value
                if isinstance(input_node.kind, Data):
                    input_node.kind.value = tensor
                else:
                    raise ValueError(f"Node '{name}' is not a Data node")
        else:
            raise ValueError("No input tensors provided")
        
        # Start from input nodes
        current_nodes = self.input_nodes.copy()
        
        # Keep track of processed nodes
        processed: Set[Node] = set()

        
        # Process nodes in topological order
        while current_nodes:
            node = current_nodes.pop(0)
            
            if node in processed:
                continue
            
            # Calculate tensor for this node
            node.get_tensor()
            
            processed.add(node)
            
            # Add output nodes to the queue
            current_nodes.extend(node.outputs)
            
                # Collect outputs
        outputs = {}
        for node in self.output_nodes:
            outputs[node.name] = node.get_tensor()
        
        return outputs
