from xand.utils import load_config
from xand.optimization_passes import remove_sums_with_identity
import torch

# Create input tensor
input_tensor = torch.tensor([1, 1, 1, 1])
graph = load_config('examples/example_0.json', input_tensor)

graph.infer_shapes()
graph = remove_sums_with_identity(graph)
# print dict of node 

outputs = graph.forward({'input_0': input_tensor})
print(outputs)