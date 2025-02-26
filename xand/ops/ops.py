from typing import Dict, Any, List, Tuple, Type, TypeVar
import torch
from ..graph import Operation, OperationType

class Add(Operation):
    def __init__(self, name: str, op_type: OperationType, args: Dict[str, Any] = {}):
        super().__init__(name, op_type, args)
        
    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        assert len(tensors) == 2
        return tensors[0] + tensors[1]
    
    # we assume there is no implicit broadcasting
    def infer_shape(self, input_shapes: List[List[int]]) -> List[int]:
        assert len(input_shapes) == 2, "Add requires exactly 2 input shapes"
        assert len(input_shapes[0]) == len(input_shapes[1]), "Add requires inputs of the same shape"
        return input_shapes[0]
    
    
class Matmul(Operation):
    def __init__(self, name: str, op_type: OperationType, args: Dict[str, Any] = {}):
        super().__init__(name, op_type, args)
        
    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        assert len(tensors) == 2, "Matmul requires exactly 2 input tensors"
        return torch.matmul(tensors[0], tensors[1])
    
    def infer_shape(self, input_shapes: List[List[int]]) -> List[int]:
        assert len(input_shapes) == 2, "Matmul requires exactly 2 input shapes"
        
        # Get shapes
        shape_a = input_shapes[0]
        shape_b = input_shapes[1]
        
        # Check if shapes are compatible for matmul
        if len(shape_a) == 0 or len(shape_b) == 0:
            raise ValueError("Matmul inputs cannot be scalars")
            
        # Handle vector-vector case
        if len(shape_a) == 1 and len(shape_b) == 1:
            # Vector dot product: [n] × [n] -> scalar
            if shape_a[0] != shape_b[0]:
                raise ValueError(f"Incompatible dimensions for vector-vector matmul: {shape_a} and {shape_b}")
            return []  # Scalar output
            
        # Handle matrix-vector case
        elif len(shape_a) == 2 and len(shape_b) == 1:
            # Matrix-vector product: [m,n] × [n] -> [m]
            if shape_a[1] != shape_b[0]:
                raise ValueError(f"Incompatible dimensions for matrix-vector matmul: {shape_a} and {shape_b}")
            return [shape_a[0]]
            
        # Handle vector-matrix case
        elif len(shape_a) == 1 and len(shape_b) == 2:
            # Vector-matrix product: [m] × [m,n] -> [n]
            if shape_a[0] != shape_b[0]:
                raise ValueError(f"Incompatible dimensions for vector-matrix matmul: {shape_a} and {shape_b}")
            return [shape_b[1]]
            
        # Handle matrix-matrix case
        elif len(shape_a) == 2 and len(shape_b) == 2:
            # Matrix-matrix product: [m,n] × [n,p] -> [m,p]
            if shape_a[1] != shape_b[0]:
                raise ValueError(f"Incompatible dimensions for matrix-matrix matmul: {shape_a} and {shape_b}")
            return [shape_a[0], shape_b[1]]
            
        # Handle batched matmul (tensors with more than 2 dimensions)
        else:
            # For batched matmul, we need to check broadcasting compatibility
            # The last two dimensions work like the matrix-matrix case
            # The batch dimensions will be broadcast
            
            # Check the matrix dimensions
            if len(shape_a) >= 2 and len(shape_b) >= 2:
                if shape_a[-1] != shape_b[-2]:
                    raise ValueError(
                        f"Incompatible dimensions for batched matmul: {shape_a} and {shape_b}. "
                        f"Last dimension of first tensor ({shape_a[-1]}) must match second-to-last "
                        f"dimension of second tensor ({shape_b[-2]})."
                    )
            
            # Calculate batch dimensions after broadcasting
            batch_dims_a = shape_a[:-2] if len(shape_a) > 2 else []
            batch_dims_b = shape_b[:-2] if len(shape_b) > 2 else []
            
            # Calculate output batch shape through broadcasting rules
            batch_shape = []
            
            # Pad shorter batch dims with 1s for broadcasting
            max_batch_dims = max(len(batch_dims_a), len(batch_dims_b))
            padded_a = [1] * (max_batch_dims - len(batch_dims_a)) + batch_dims_a
            padded_b = [1] * (max_batch_dims - len(batch_dims_b)) + batch_dims_b
            
            # Apply broadcasting rules
            for dim_a, dim_b in zip(padded_a, padded_b):
                if dim_a == 1:
                    batch_shape.append(dim_b)
                elif dim_b == 1:
                    batch_shape.append(dim_a)
                elif dim_a == dim_b:
                    batch_shape.append(dim_a)
                else:
                    raise ValueError(
                        f"Incompatible batch dimensions for matmul: {shape_a} and {shape_b}. "
                        f"Cannot broadcast batch dimensions."
                    )
            
            # Construct final output shape
            output_shape = batch_shape + [shape_a[-2] if len(shape_a) >= 2 else 1, 
                                          shape_b[-1] if len(shape_b) >= 2 else 1]
            
            return output_shape
        
    
    
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
    
    
class Transpose(Operation):
    def __init__(self, name: str, op_type: OperationType, args: Dict[str, Any] = {}):
        # Transpose typically requires two dimensions to swap
        if "dim0" not in args or "dim1" not in args:
            raise ValueError("Transpose operation requires 'dim0' and 'dim1' arguments")
        super().__init__(name, op_type, args)
        
    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        assert len(tensors) == 1, "Transpose operation requires exactly 1 input tensor"
        return tensors[0].transpose(dim0=self.args["dim0"], dim1=self.args["dim1"])
    
    def infer_shape(self, input_shapes: List[List[int]]) -> List[int]:
        assert len(input_shapes) == 1, "Transpose operation requires exactly 1 input shape"
        
        dim0 = self.args["dim0"]
        dim1 = self.args["dim1"]
        input_shape = input_shapes[0]
        
        # Handle negative indexing
        if dim0 < 0:
            dim0 += len(input_shape)
        if dim1 < 0:
            dim1 += len(input_shape)
            
        # Validate dimensions
        if dim0 >= len(input_shape) or dim1 >= len(input_shape):
            raise ValueError(f"Transpose dimensions {dim0} and {dim1} out of bounds for shape {input_shape}")
            
        # Create the output shape by swapping the specified dimensions
        output_shape = input_shape.copy()
        output_shape[dim0], output_shape[dim1] = output_shape[dim1], output_shape[dim0]
        
        return output_shape
    

T = TypeVar("T", bound=Operation)
    
# Add more operations as needed
op_map: Dict[str, Tuple[Type[Operation], OperationType]] = {
    "add": (Add, OperationType.BINARY),
    "unsqueeze": (Unsqueeze, OperationType.TENSOR_MANIPULATION),   
    "matmul": (Matmul, OperationType.BINARY),
    "transpose": (Transpose, OperationType.TENSOR_MANIPULATION)
}