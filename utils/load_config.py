import json
from typing import Dict, Any, Union
import torch

from graph import Graph, Node, Data, Operation, DataType
from ops import op_map


def create_operation(op_config: Dict[str, Any]) -> Operation:
    """Create Operation instance based on operation configuration"""
    op_name = op_config["name"]
    op_type = op_config["type"]
    args = op_config.get("args", {})
        
    if op_name not in op_map:
        raise ValueError(f"Unknown operation: {op_name}")
        
    op_class, op_type = op_map[op_name]
    return op_class(name=op_name, op_type=op_type, args=args)


def create_data(data_config: Dict[str, Any]) -> Data:
    """Create Data instance based on data configuration"""
    type = DataType[data_config["type"]]

    if "value" in data_config:
        # Convert list to tensor
        return Data(type=type, value=torch.tensor(data_config["value"]))
    else:
        raise ValueError("Data node must have a value")


def create_node(node_config: Dict[str, Any], input_values: Dict[str, torch.Tensor] = {}) -> Node:
    """Create Node instance based on node configuration"""
    name = node_config["name"]
    kind_config = node_config["kind"]
    
    kind: Union[Data, Operation]
    
    if kind_config["kind"] == "OP":
        kind = create_operation(kind_config)
    elif kind_config["kind"] == "DATA":
        kind = create_data(kind_config)
    else:
        raise ValueError(f"Unknown kind: {kind_config['kind']}")
    
    return Node(name=name, kind=kind)


def load_config(config_path: str, input_sample: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Graph:
    """Load graph configuration from JSON file
    
    Args:
        config_path: Path to the JSON configuration file
        input_sample: Either a single tensor (for single input) or dict of tensors (for multiple inputs)
    """
    # Create graph instance
    graph = Graph()
    
    # Prepare input values dictionary
    if isinstance(input_sample, torch.Tensor):
        # Single input case - assume input_0 as name
        input_values = {"input_0": input_sample}
    else:
        input_values = input_sample
        
    # Read configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # First pass: Create all nodes
    nodes_map: Dict[str, Node] = {}
    
    # Create all explicit input nodes with their values
    for name, value in input_values.items():
        if name not in nodes_map:
            input_node = Node(
                name=name,
                kind=Data(type=DataType.INPUT, value=value)
            )
            nodes_map[name] = input_node
            graph.add_node(input_node)
            graph.input_nodes.append(input_node)
            
    # Then create all nodes from config
    for node_config in config:
        if node_config["name"] not in nodes_map:  # Skip if already created
            node = create_node(node_config, input_values)
            nodes_map[node_config["name"]] = node
            graph.add_node(node)
            
            
    # Second pass: Connect nodes
    for node_config in config:
        node = nodes_map[node_config["name"]]
        
        # Connect inputs
        for input_name in node_config["inputs"]:
            if input_name not in nodes_map:
                raise ValueError(f"Input node {input_name} not found for node {node.name}")
            input_node = nodes_map[input_name]
            graph.connect(input_node, node)
            
    # Validate that all input nodes have values
    for input_node in graph.input_nodes:
        if isinstance(input_node.kind, Data) and input_node.kind.value is None:
            raise ValueError(f"Input node {input_node.name} has no value")
        
    
    return graph
