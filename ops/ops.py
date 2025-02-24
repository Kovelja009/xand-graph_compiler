import torch

from graph import Operation, OperationType
from typing import Dict, Any, List, Tuple, Type, TypeVar

class Add(Operation):
    def __init__(self, name: str, op_type: OperationType, args: Dict[str, Any] = {}):
        super().__init__(name, op_type, args)
        
    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        assert len(tensors) == 2
        return tensors[0] + tensors[1]
    
    # we assume there is no implicit broadcasting
    def infer_shape(self, input_shapes: List[List[int]]) -> List[int]:
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) == len(input_shapes[1])
        return input_shapes[0]
    
    
class Unsqueeze(Operation):
    def __init__(self, name: str, op_type: OperationType, args: Dict[str, Any] = {}):
        if "dim" not in args:
            raise ValueError("Unsqueeze operation requires 'dim' argument")
        super().__init__(name, op_type, args)
        
    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        assert len(tensors) == 1
        return tensors[0].unsqueeze(dim=self.args["dim"])
    
    def infer_shape(self, input_shapes: List[List[int]]) -> List[int]:
        assert len(input_shapes) == 1
        dim = self.args["dim"]
        
        # Allow negative indexing
        if dim < 0:
            dim += len(input_shapes[0]) + 1
        
        return input_shapes[0][:dim] + [1] + (input_shapes[0][dim:] if dim < len(input_shapes[0]) else [])
    

T = TypeVar("T", bound=Operation)
    
# Add more operations as needed
op_map: Dict[str, Tuple[Type[Operation], OperationType]] = {
    "add": (Add, OperationType.BINARY),
    "unsqueeze": (Unsqueeze, OperationType.TENSOR_MANIPULATION),   
}