from ..graph import Graph

def transpose_cancelation(graph: Graph) -> Graph:
    '''
    Removes consecutive transpose operations.
    
    This optimization pass looks for patterns where two transpose operations
    are performed consecutively and cancel each other out. In such cases, both
    transpose operations are eliminated, and the original input is connected
    directly to the final output.
    
    Example:
        Input graph:
            input → transpose_1 → transpose_2 → output
            
        After optimization:
            input → output
    '''
    
    # Continue optimizing until no more changes can be made
    changes_made = True
    while changes_made:
        changes_made = False
        
        # Get all transpose operations
        if 'transpose' not in graph.nodes_by_name:
            break  # No transpose operations to optimize
        
        transpose_nodes = graph.nodes_by_name['transpose'].copy()
        
        for transpose_node in transpose_nodes:
                
            # Check all inputs to see if any are also transpose nodes
            for input_node in transpose_node.inputs:
                #  input is a transpose node
                if ('transpose' in graph.nodes_by_name and 
                    input_node in graph.nodes_by_name['transpose']):
                    
                    # We found a consecutive pair of transpose operations
                    first_transpose = input_node
                    second_transpose = transpose_node
                    
                    # IMPORTANT: Check if the first transpose has exactly one output
                    # and that output is the second transpose
                    if len(first_transpose.outputs) != 1 or first_transpose.outputs[0] != second_transpose:
                        continue  # Skip if this condition isn't met
                    
                    # Check if the double transpose preserves the original shape
                    
                    # Get the original input shape (input to first transpose)                        
                    original_input = first_transpose.inputs[0]
                    original_shape = original_input.get_shape()
                    
                    # Get the final output shape (output of second transpose)
                    final_shape = second_transpose.get_shape()
                    
                    # If shapes don't match, these transposes don't cancel out
                    if original_shape != final_shape:
                        continue

                    
                    # Check if second_transpose was an output node
                    was_output = second_transpose in graph.output_nodes
                    
                    # Connect the original input directly to all outputs of the second transpose
                    for output_node in second_transpose.outputs.copy():
                        second_transpose.outputs.remove(output_node)
                        output_node.inputs.remove(second_transpose)
                        graph.connect(original_input, output_node)
                        
                    # Remove the first transpose from the outputs of the original input
                    if first_transpose in original_input.outputs:
                        original_input.outputs.remove(first_transpose)
                    
                    # If the second transpose was an output, mark the original input as output too
                    if was_output and original_input not in graph.output_nodes:
                        graph.output_nodes.append(original_input)
                    
                    # Remove both transpose nodes from output_nodes if needed
                    if first_transpose in graph.output_nodes:
                        graph.output_nodes.remove(first_transpose)
                    if second_transpose in graph.output_nodes:
                        graph.output_nodes.remove(second_transpose)
                    
                    # Remove both nodes from the graph
                    if first_transpose in graph.nodes:
                        graph.nodes.remove(first_transpose)
                    if second_transpose in graph.nodes:
                        graph.nodes.remove(second_transpose)
                    
                    # Remove from nodes_by_name
                    if 'transpose' in graph.nodes_by_name:
                        if first_transpose in graph.nodes_by_name['transpose']:
                            graph.nodes_by_name['transpose'].remove(first_transpose)
                        if second_transpose in graph.nodes_by_name['transpose']:
                            graph.nodes_by_name['transpose'].remove(second_transpose)
                        if not graph.nodes_by_name['transpose']:
                            del graph.nodes_by_name['transpose']
                    
                    # Mark that we made a change and break to restart with fresh node lists
                    changes_made = True
                    break
            
            if changes_made:
                break
    
    return graph