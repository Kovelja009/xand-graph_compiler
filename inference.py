from utils import load_config
import torch

# Create input tensor
input_tensor = torch.tensor([1, 1, 1, 1])
graph = load_config('examples/example_0.json', input_tensor)