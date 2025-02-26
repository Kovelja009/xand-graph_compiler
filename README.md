# XAND: A Lightweight Neural Network Graph Compiler

## Project Summary

XAND is a simple yet powerful graph compiler for neural networks. It provides a streamlined way to optimize computational graphs and execute them efficiently. The project was designed to demonstrate fundamental compiler techniques like graph optimization, shape inference, and constant folding in a neural network context.

## Key Features

### Core Architecture
- **Node-based computation graph**: Represents neural networks as directed graphs with data nodes and operation nodes
- **Shape inference**: Automatically infers tensor shapes throughout the graph
- **Lazy evaluation**: Computes tensors only when needed using a demand-driven evaluation model
- **PyTorch integration**: Uses PyTorch tensors for all operations with a familiar API

### Optimization Passes
- **Constant folding (consteval)**: Pre-computes expressions with constant inputs at compile time
- **Identity elimination**:
  - `add_zero`: Removes unnecessary additions with zero tensors
  - `matmul_identity`: Eliminates matrix multiplications with identity matrices
- **Transpose cancellation**: Removes pairs of consecutive transpose operations that cancel each other out
- **Dead node elimination**: Automatically removes redundant nodes during optimization

### Usage Interface
- **Simple compilation**: Compile models from JSON configuration files with a single call
- **PyTorch-like inference**: Use compiled models with familiar PyTorch-style syntax
- **Multiple input support**: Handle models with any number of input tensors

## Example Usage

```python
import torch
import xand

# Compile model from configuration
model = xand.compile('model_config.json', torch.randn(3, 3))

# Run inference (PyTorch-style)
output = model(torch.randn(3, 3))

# For multiple inputs
output = model(torch.randn(3, 3), torch.randn(4, 4))
```

## Implementation Details
The compiler follows these key steps:
1. Load graph configuration from JSON
2. Perform shape inference on the entire graph
3. Apply optimization passes to simplify the computation
4. Compile into an executable module
5. Execute the optimized graph with the provided inputs

## Project Structure
- `graph.py`: Core graph and node data structures
- `operations/`: Different operation implementations (Add, MatMul, Transpose, etc.)
- `optimization_passes/`: Graph optimization techniques
- `utils.py`: Configuration loading and utility functions
- `xand.py`: Main compiler interface

## Future Directions
- Gradient computation support
- More advanced optimization passes
- Memory planning for optimal tensor allocation
- Code generation for different hardware targets
- Quantization support
