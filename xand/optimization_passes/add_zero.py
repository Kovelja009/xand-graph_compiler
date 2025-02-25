import torch
from ..graph import Graph, Node, Data, DataType


def is_zero_tensor(node: Node) -> bool:
    """Check if a node represents a constant zero tensor."""
    if not isinstance(node.kind, Data) or node.kind.type != DataType.CONSTANT:
        return False
    
    tensor = node.get_tensor()
    if tensor is None:
        return False
    
    # Check if all elements are zero
    return bool(torch.all(tensor == 0).item())

def sum_identity(graph: Graph) -> Graph:
    '''
    Removes unnecessary additions with zero.
    
    This optimization pass looks for addition operations where one of the operands
    is a constant tensor filled with zeros. In such cases, the addition operation
    is redundant and can be replaced with the non-zero operand.
    
    Example:
        Input graph:
            input → add → output
                     ↑
                   zeros
            
        After optimization:
            input → output
    '''
    
    # Continue optimizing until no more changes can be made
    changes_made = True
    while changes_made:
        changes_made = False
        
        # Get all add operations
        if 'add' not in graph.nodes_by_name:
            break  # No add operations to optimize
        
        add_nodes = graph.nodes_by_name['add'].copy()
        
        for add_node in add_nodes:
            # Check if operation is binary (has exactly 2 inputs)
            if len(add_node.inputs) != 2:
                continue
            
            # Check if any of the inputs are zero constants
            zero_inputs = []
            non_zero_inputs = []
            
            for input_node in add_node.inputs:
                if is_zero_tensor(input_node):
                    zero_inputs.append(input_node)
                else:
                    non_zero_inputs.append(input_node)

            # If at least one input is zero, we can optimize
            if zero_inputs:
                # Determine which node to keep (non-zero input or first zero input)
                node_to_keep = non_zero_inputs[0] if non_zero_inputs else zero_inputs[0]
                nodes_to_remove = [node for node in add_node.inputs if node != node_to_keep]
                
                # Check if add_node was an output node
                was_output = add_node in graph.output_nodes
                
                # Redirect all outputs of the add node to use the node we're keeping
                for output_node in add_node.outputs.copy():
                    add_node.outputs.remove(output_node)
                    output_node.inputs.remove(add_node)
                    graph.connect(node_to_keep, output_node)
                
                # If the add node was an output, mark the kept node as output too
                if was_output and node_to_keep not in graph.output_nodes:
                    graph.output_nodes.append(node_to_keep)
                
                # Remove add node from graph and output_nodes if needed
                if add_node in graph.output_nodes:
                    graph.output_nodes.remove(add_node)
                graph.nodes.remove(add_node)
                if 'add' in graph.nodes_by_name:
                    graph.nodes_by_name['add'].remove(add_node)
                    if not graph.nodes_by_name['add']:
                        del graph.nodes_by_name['add']
                
                # Remove other nodes if they're not used by any other node
                for node in nodes_to_remove:
                    # first remove add_node from outputs
                    if add_node in node.outputs:
                        node.outputs.remove(add_node)
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