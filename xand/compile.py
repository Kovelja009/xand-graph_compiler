from typing import Union, List
import torch
from .graph import Graph
from .optimization_passes import sum_identity, matmul_identity, transpose_cancelation, consteval
from .utils import load_config


class XandModule():
    def __init__(self, graph: Graph):
        self.graph = graph
        
    def __call__(self, *inputs: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Make the module callable like a PyTorch module for inference.
        
        Args:
            *inputs: Input tensors for the graph
            
        Returns:
            Single output tensor or list of output tensors
        """
        
        # Validate number of inputs
        if len(inputs) != len(self.graph.input_nodes):
            raise ValueError(f"Expected {len(self.graph.input_nodes)} inputs, got {len(inputs)}")
        
        # Create input dictionary
        input_dict = {}
        for i, (node, tensor) in enumerate(zip(self.graph.input_nodes, inputs)):
            input_dict[node.name] = tensor
        
        # Run the graph forward pass
        outputs = self.graph.forward(input_dict)
        
        # Return result(s)
        if len(outputs) == 1:
            # Return single tensor if there's only one output
            return next(iter(outputs.values()))
        else:
            # Return list of tensors in a consistent order
            return [outputs[node.name] for node in self.graph.output_nodes]
    
    
def compile(config_path: str, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> XandModule:
    """
    Compile a graph from a configuration file and optimize it.
    
    Args:
        config_path: Path to the graph configuration file
        *inputs: Input tensor(s) for shape inference
        
    Returns:
        Compiled XandModule ready for inference
    """
    
    # If single input tensor, convert to list
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
        
    # Convert list of tensors to dictionary
    inputs = {f"input_{i}": tensor for i, tensor in enumerate(inputs)}
    
    # Load graph configuration
    graph = load_config(config_path, inputs)
    
    # Infer shapes
    graph.infer_shapes()
    
    # Run optimization passes
    graph = optimize(graph)
    
    # Compile graph into a module
    module = XandModule(graph)
    
    return module


def optimize(graph: Graph) -> Graph:
    graph = consteval(graph)
    graph = sum_identity(graph)
    graph = matmul_identity(graph)
    graph = transpose_cancelation(graph)
    
    return graph
