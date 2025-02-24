import torch
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

class DataType(Enum):
    CONSTANT = auto()
    VARIABLE = auto()
    PARAMETER = auto()
    INPUT = auto()
    
class OperationType(Enum):
    UNARY = auto()
    BINARY = auto()
    TENSOR_MANIPULATION = auto()
    
class Data:
    def __init__(self, type: DataType, value: torch.Tensor):
        # dtype not yet supported :(
        self.type = type
        self.value = value
        self.shape = list(value.shape) if value is not None else None

        
class Operation(ABC):
    def __init__(self, name: str, op_type: OperationType, args: Dict[str, Any] = {}):
        self.name = name
        self.type = op_type
        self.args = args
    
    @abstractmethod
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def infer_shape(self, input_shapes: List[List[int]]) -> List[int]:
        pass

class Node:
    def __init__(self, name: str, kind: Union[Data, Operation]):
        # Extract ID from name (e.g., 'matmul_9' -> 9)
        self.name = name
        self.id = int(name.split('_')[-1]) if '_' in name else -1

        self.kind = kind
        self.inputs: List[Node] = []
        self.outputs: List[Node] = []
        self.tensor: Optional[torch.Tensor] = None
        self.shape: Optional[List[int]] = None
        
    def get_tensor(self) -> torch.Tensor:
        # If tensor is already computed, return it
        if self.tensor is not None:
            return self.tensor
        
        # If node is Data kind, get tensor from value
        if isinstance(self.kind, Data):
            self.tensor = self.kind.value
            return self.tensor
        
        # Node is Operation kind, need to compute forward
        if isinstance(self.kind, Operation):
            # Get input tensors recursively
            input_tensors = [input_node.get_tensor() for input_node in self.inputs]
            # Compute result using operation's forward method
            self.tensor = self.kind.forward(input_tensors)
            return self.tensor
        
        raise ValueError(f"Invalid kind type for node {self.name}")

    def get_shape(self) -> List[int]:
        """Get the shape of this node's output tensor"""
        # If shape is already computed, return it
        if self.shape is not None:
            return self.shape
        
        # If node is Data kind, get shape from value
        if isinstance(self.kind, Data):
            self.shape = self.kind.shape
            if self.shape is None:
                raise ValueError(f"Data node {self.name} has no shape information")
            return self.shape
        
        # Node is Operation kind, need to infer shape
        if isinstance(self.kind, Operation):
            # Get input shapes recursively
            input_shapes = [input_node.get_shape() for input_node in self.inputs]
            # Infer shape using operation's shape inference method
            self.shape = self.kind.infer_shape(input_shapes)
            return self.shape
        
        raise ValueError(f"Invalid kind type for node {self.name}")

    def clear_tensor(self) -> None:
        """Clear stored tensor to free memory or recompute with new inputs"""
        self.tensor = None

    def clear_shape(self) -> None:
        """Clear stored shape to allow recomputation"""
        self.shape = None
