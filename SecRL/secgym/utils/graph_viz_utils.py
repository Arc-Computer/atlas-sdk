# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#Graph Viz time
import codecs
import networkx as nx
import pandas as pd
from pyvis.network import Network
import os
from IPython.display import display, HTML
from pyvis import network as net
import pickle
import json

def node_2_nl(node):
    nl = str(node[1]['node_type']+" "+node[0])
    #nl2skg_dict[nl] = node
    return nl

def print_bfs_graph(g, node_id, radius=1):
    """
    Print ego graph for given node
    """
    ego_graph = nx.ego_graph(g, node_id, radius=radius)
    nodes_str = ""

    #print("getting graph context for node: ", node_id)

    for neighbor in ego_graph.nodes(data=True):
        
        neighbor_id = neighbor[0]
        #print("Neighbor Node: ", neighbor_id)
        
        if neighbor_id == node_id:
            continue

        neighbor_nl = node_2_nl(neighbor)
        neighbor_type = neighbor[1]['node_type']
        neighbor_desc = json.dumps(neighbor[1])
        edge_info = g.get_edge_data(node_id, neighbor_id)

        nodes_str += str("Neighbor Node: "+neighbor_nl+", Node Type:"+neighbor_type+", Relation to main node: "+edge_info['Relationship']+", ") #+", Node Description: "+neighbor_desc+";"), TODO: Add node description to testing

    #print(nodes_str)    
    return nodes_str

#TODO: Implement DFS graph print
def print_dfs_graph(g, node_id):
    return

def get_graph_context(g, node_list, mode="bfs"):
    """
    Get graph context for given nodes
    """
    graph_context = ""

    for node_id in node_list:
        #print(node_id)
        node_desc = g.nodes(data=True)[node_id]
        node = [node_id, node_desc]
        node_nl = node_2_nl(node)
        node_str = str("Node: "+node_nl+", Node Description: "+json.dumps(node_desc))

        if mode == "bfs":
            node_str += str(";"+print_bfs_graph(g, node_id))
        elif mode == "dfs":
            node_str += str(";"+print_dfs_graph(g, node_id))
        else:   
            #start_node_str += str(";"+get_nearby_nodes(g, node_id))
            return 
        
        graph_context += str(", "+node_str)

    #print(graph_context)
    return graph_context
        
#Graph Viz Functions
def get_hover_attributes(node_id, node_dict):

    def format_attributes(attribute_list):
        attribute_list = ['node_type'] + attribute_list
        return '\n'.join([f'ID: {node_id}'] + [f'{key}:{format_values(node_dict[key])}' for key in attribute_list])
    
    def format_hover_attribs(node_dict):

        return '\n'.join(f'{key}:{format_values(str(value))}' for key,value in node_dict.items())
    
    def format_values(content):
        max_length_per_line = 100
        lines = []
        for i in range(len(content)//max_length_per_line + 1):
            line_start = i  * max_length_per_line
            line_end = (i + 1) * max_length_per_line
            lines.append(content[line_start:line_end])
        
        return '\n'.join(lines)
    
    
    return format_hover_attribs(node_dict)
def generate_pyvis(subgraph, outputfile="graph.html"):
    """
    Generates an interactive visualization of the graph using PyVis.
    - Alert nodes: Larger size, red color, show 'id' for label, and 'id' + 'name' on hover.
    - Entity nodes: Smaller size, blue color, show 'id' for label, and 'id' + 'value' on hover.
    """
    g = net.Network(notebook=True, filter_menu=True, select_menu=True)
    g.force_atlas_2based()

    # Add nodes to the PyVis graph
    for node in subgraph.nodes(data=True):
        node_id = node[0]  # Node ID
        node_attrs = node[1]  # Node attributes

        if node_attrs['type'] == "alert":
            # Alert node: Larger size, red color
            label = f"{node_id}"
            hover_info = f"ID: {node_id}\nType: {node_attrs.get('type', 'N/A')}\nName: {node_attrs.get('name', 'N/A')}"
            g.add_node(
                node_id, 
                label=label, 
                node_type="alert", 
                title=hover_info, 
                shape="square", 
                color="#FF7F7F", 
                value=22  # Larger size for alerts
            )
        
        elif node_attrs['type'] == "entity":
            # Entity node: Smaller size, blue color
            label = f"{node_id}"
            hover_info = f"ID: {node_id}\nType: {node_attrs.get('type', 'N/A')}\nValue: {node_attrs.get('value', 'N/A')}"
            g.add_node(
                node_id, 
                label=label, 
                node_type="entity", 
                title=hover_info, 
                color="#87CEEB", 
                value=20  # Smaller size for entities
            )
        
        else:
            # Default node visualization (if no type is defined)
            label = f"{node_id}"
            hover_info = f"ID: {node_id}\nDetails: {json.dumps(node_attrs)}"
            g.add_node(
                node_id, 
                label=label, 
                group="default", 
                title=hover_info, 
                shape="circle", 
                color="gray", 
                value=10  # Default node size
            )

    # Add edges to the PyVis graph
    for edge in subgraph.edges(data=True):
        src, dst, edge_attrs = edge
        edge_title = json.dumps(edge_attrs) if edge_attrs else "N/A"
        g.add_edge(src, dst, title=edge_title)

    # Enable physics for better layout
    g.toggle_physics(True)
    g.show_buttons(filter_=['physics'])
    g.show(outputfile)
    display(outputfile)

def save_graph_as_pickle(g,output_file="test.pkl"):
    with open(output_file, 'wb') as f:
        pickle.dump(g, f)

def load_graph_as_pickle(input_file="test.pkl"):
    with open(input_file, 'rb') as f:
        g = pickle.load(f)
        return g
    


if __name__ == "__main__":
    from secgym.qagen.alert_graph import AlertGraph
    import os
    alert_graph = AlertGraph()
    alert_graph.load_graph_from_graphml(filepath=f"./qagen/graph_files/incident_322.graphml")
        
    # Generate PyVis visualization
    generate_pyvis(alert_graph.graph, outputfile="sample_graph.html")