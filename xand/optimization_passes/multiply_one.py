import torch
from typing import List, Optional, Set
from ..graph import Graph, Node, Data, Operation, DataType



def is_one_tensor(node: Node) -> bool:
    """Check if a node represents an identity matrix or a tensor of all ones."""
    if not isinstance(node.kind, Data) or node.kind.type != DataType.CONSTANT:
        return False
    
    tensor = node.get_tensor()
    if tensor is None:
        return False
    
    # For scalar case
    if tensor.dim() == 0:
        return tensor.item() == 1
    
    # For vector case (all ones)
    if tensor.dim() == 1:
        return bool(torch.all(tensor == 1).item())
    
    # For matrix case, check if it's an identity matrix
    if tensor.dim() == 2:
        # Check if matrix is square
        if tensor.shape[0] != tensor.shape[1]:
            # Non-square matrices can't be identity matrices
            return False
            
        # Use torch.eye to create an identity matrix and compare
        # This is a much cleaner approach than the previous implementation
        identity = torch.eye(tensor.shape[0])
        return torch.equal(tensor, identity)
    
    # For higher dimensions, check element-wise
    return bool(torch.all(tensor == 1).item())


def matmul_identity(graph: Graph) -> Graph:
    '''
    Removes unnecessary matrix multiplications with identity matrices.
    
    This optimization pass looks for matmul operations where one of the operands
    is an identity matrix. In such cases, the matmul operation is redundant and
    can be replaced with the non-identity operand.
    
    Example:
        Input graph:
            input → matmul → output
                      ↑
                  identity
            
        After optimization:
            input → output
    '''

    # Continue optimizing until no more changes can be made
    changes_made = True
    while changes_made:
        changes_made = False
        
        # Get all matmul operations
        matmul_ops = []
        if "matmul" in graph.nodes_by_name:
            matmul_ops.extend(graph.nodes_by_name["matmul"].copy())
        
        if not matmul_ops:
            break  # No matmul operations to optimize
        
        for matmul_node in matmul_ops:
            # Check if operation is binary (has exactly 2 inputs)
            if len(matmul_node.inputs) != 2:
                continue
            
            # Check if any of the inputs are identity matrices
            identity_inputs = []
            non_identity_inputs = []
            
            for i, input_node in enumerate(matmul_node.inputs):
                if is_one_tensor(input_node):
                    identity_inputs.append((i, input_node))
                else:
                    non_identity_inputs.append((i, input_node))
            
            # If at least one input is an identity matrix, we can optimize
            if identity_inputs:
                # Determine which node to keep
                # If we have non-identity inputs, use the first one
                # Otherwise, use the first identity input
                if non_identity_inputs:
                    _, node_to_keep = non_identity_inputs[0]
                else:
                    # If both are identity matrices, just pick the first one
                    _, node_to_keep = identity_inputs[0]
                    
                    # IMPORTANT: Check if the shapes are compatible for removal
                    # We can only remove the matmul node if the shape of the output
                    # would be the same as the shape of the non-identity input

                    # If the shapes wouldn't match, skip this optimization
                    if node_to_keep.get_shape() != matmul_node.get_shape():
                        continue
                
                # Collect nodes to potentially remove (all inputs except the one we're keeping)
                nodes_to_remove = []
                for input_node in matmul_node.inputs:
                    if input_node != node_to_keep:
                        nodes_to_remove.append(input_node)
                
                # Check if matmul_node was an output node
                was_output = matmul_node in graph.output_nodes
                
                # Redirect all outputs of the matmul node to use the node we're keeping
                for output_node in matmul_node.outputs.copy():
                    matmul_node.outputs.remove(output_node)
                    output_node.inputs.remove(matmul_node)
                    graph.connect(node_to_keep, output_node)
                
                # If the matmul node was an output, mark the kept node as output too
                if was_output and node_to_keep not in graph.output_nodes:
                    graph.output_nodes.append(node_to_keep)
                
                # Remove matmul node from graph and output_nodes if needed
                if matmul_node in graph.output_nodes:
                    graph.output_nodes.remove(matmul_node)
                graph.nodes.remove(matmul_node)
                
                # Remove from nodes_by_name
                if "matmul" in graph.nodes_by_name and matmul_node in graph.nodes_by_name["matmul"]:
                    graph.nodes_by_name["matmul"].remove(matmul_node)
                    if not graph.nodes_by_name["matmul"]:
                        del graph.nodes_by_name["matmul"]
                
                # Remove identity inputs if they're not used by any other node
                for node in nodes_to_remove:
                    # First remove matmul_node from outputs
                    if matmul_node in node.outputs:
                        node.outputs.remove(matmul_node)
                    if not node.outputs:
                        if node in graph.nodes:
                            graph.nodes.remove(node)
                        
                        # Remove from nodes_by_name
                        base_name = node.name.rsplit('_', 1)[0]
                        if base_name in graph.nodes_by_name and node in graph.nodes_by_name[base_name]:
                            graph.nodes_by_name[base_name].remove(node)
                            if not graph.nodes_by_name[base_name]:
                                del graph.nodes_by_name[base_name]
                
                # Mark that we made a change and break to restart with fresh node lists
                changes_made = True
                break

    # No need to recalculate output nodes since we maintain them during optimization
    return graph