
from typing import List, Dict, Any, Optional, Union
from .node import Node

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
        # This method will be implemented after topological sort
        # as we need to process nodes in the correct order
        pass