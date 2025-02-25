from xand.utils import load_config
from xand.optimization_passes import sum_identity, matmul_identity, transpose_cancelation
import torch

# Create input tensor
input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
graph = load_config('examples/example_2.json', input_tensor)

graph.infer_shapes()
graph = sum_identity(graph)
graph = matmul_identity(graph)
graph = transpose_cancelation(graph)

outputs = graph.forward({'input_0': input_tensor})
print(outputs)

for node in graph.nodes:
    print(node.name, node.get_shape())