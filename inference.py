import torch
import xand


######## Example 0 ########

# Create input tensor (4)
input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
compiled_module = xand.compile('examples/example_0.json', input_tensor)
outputs = compiled_module(input_tensor)

print(f"\nOutput of the 0th example: \n{outputs}")
print("##############################################")


######## Example 1 ########

# Create input tensor (2,2)
input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
compiled_module = xand.compile('examples/example_1.json', input_tensor)
outputs = compiled_module(input_tensor)

print(f"\nOutput of the 1st example: \n{outputs}")
print("##############################################")


######## Example 2 ########

# Create input tensor (2,2)
input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
compiled_module = xand.compile('examples/example_2.json', input_tensor)
outputs = compiled_module(input_tensor)

print(f"\nOutput of the 2nd example: \n{outputs}")
print("##############################################")


######## Example 3 ########

# Create input tensor (3,3)
input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
compiled_module = xand.compile('examples/example_3.json', input_tensor)
outputs = compiled_module(input_tensor)

print(f"\nOutput of the 3rd example: \n{outputs}")
print("##############################################")