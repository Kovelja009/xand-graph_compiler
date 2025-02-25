from typing import List, Dict
import torch
from ..graph import Graph, Node, Data, Operation, DataType


def consteval(graph: Graph) -> Graph:
    '''
    Constant Evaluation Optimization Pass
    
    This optimization pass identifies operations where all inputs are constants,
    computes the result at compile time, and replaces the operation with a new
    constant node containing the pre-computed result.
    
    Example:
        Input graph:
            const_1 → add → output
              ↑
            const_2
            
        After optimization:
            computed_const → output
    '''
    
    # Continue optimizing until no more changes can be made
    changes_made = True
    while changes_made:
        changes_made = False
        
        # Check all nodes in the graph
        for node in list(graph.nodes):  # Create a copy to safely modify during iteration
            # Skip if not an operation node
            if not isinstance(node.kind, Operation):
                continue
                
            # Check if all inputs are constant nodes
            all_inputs_constant = True
            for input_node in node.inputs:
                if not isinstance(input_node.kind, Data) or input_node.kind.type != DataType.CONSTANT:
                    all_inputs_constant = False
                    break
            
            # If all inputs are constants, we can evaluate this operation at compile time
            if all_inputs_constant and node.inputs:
                # Gather input tensors
                input_tensors = [input_node.get_tensor() for input_node in node.inputs]
                
                # Compute the result
                result_tensor = node.kind.forward(input_tensors)
                
                # Create a new constant node with the computed result
                const_name = f"{node.name.split('_')[0]}_const_{node.id}"
                
                # Create new constant data
                const_data = Data(type=DataType.CONSTANT, value=result_tensor)
                
                # Create new constant node
                const_node = Node(name=const_name, kind=const_data)
                
                # Add new node to graph
                graph.add_node(const_node)
                
                # Check if the original node was an output node
                was_output = node in graph.output_nodes

                # If the original node was an output, mark the new constant as output
                if was_output:
                    if node in graph.output_nodes:
                        graph.output_nodes.remove(node)
                    if const_node not in graph.output_nodes:
                        graph.output_nodes.append(const_node)
                
                # Connect the new constant node to all outputs of the original node
                for output_node in node.outputs.copy():
                    # Disconnect from the original node
                    node.outputs.remove(output_node)
                    output_node.inputs.remove(node)
                    
                    # Connect to the new constant node
                    graph.connect(const_node, output_node)
                
                
                # Remove the original operation node
                if node in graph.nodes:
                    graph.nodes.remove(node)
                
                # Remove from nodes_by_name
                base_name = node.name.rsplit('_', 1)[0]
                if base_name in graph.nodes_by_name and node in graph.nodes_by_name[base_name]:
                    graph.nodes_by_name[base_name].remove(node)
                    if not graph.nodes_by_name[base_name]:
                        del graph.nodes_by_name[base_name]
                
                # Remove input nodes if they're not used by any other nodes
                for input_node in node.inputs:
                    input_node.outputs.remove(node)  # Remove connection to the removed node
                    
                    if not input_node.outputs:  # If no more outputs, we can remove this node
                        if input_node in graph.nodes:
                            graph.nodes.remove(input_node)
                            
                        # Remove from nodes_by_name
                        base_name = input_node.name.rsplit('_', 1)[0]
                        if base_name in graph.nodes_by_name and input_node in graph.nodes_by_name[base_name]:
                            graph.nodes_by_name[base_name].remove(input_node)
                            if not graph.nodes_by_name[base_name]:
                                del graph.nodes_by_name[base_name]
                
                # Mark that we made a change
                changes_made = True
                break  # Break to restart with fresh nodes list
                    
    
    return graph