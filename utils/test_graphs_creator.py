# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:06:56 2020

@author: olive
"""


import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

os.chdir("../data/square")

# Creates graph
def create_graph():
    G = nx.Graph()
    
    G.add_node(1, x=10, y=10)
    G.add_node(2, x=10, y=20)
    G.add_node(3, x=20, y=10)

    
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)    

    return G

G = create_graph()

G.nodes(data=True)

# Gets positions of a graph
def get_pos(G):
    
    nodes = G.nodes
    
    pos = {}
    
    for n in nodes:
        pos[n] = [G.nodes(data=True)[n]["x"], G.nodes(data=True)[n]["y"]]
        
    return pos


# Plots a graph with correct axes     
def plot_graphs(G):
    pos = get_pos(G)
        
    x_pos = []
    y_pos = []
    
    for k in pos.keys():
        x_pos.append(pos[k][0])        
        y_pos.append(pos[k][1])   
    
    x_min = min(x_pos)
    x_max = max(x_pos)

    y_min = min(y_pos)
    y_max = max(y_pos)
    
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos)
    plt.xlim(x_min-x_min*0.1, x_max+x_max*0.1)
    plt.ylim(y_min-y_min*0.1, y_max+y_max*0.1)
    plt.show()

G =  nx.geographical_threshold_graph(20, 15)

def pos_to_xy(G):
    for n in range(len(G.nodes(data=True))):
        G.nodes(data=True)[n]["x"] = G.nodes(data=True)[n]["pos"][0]
        G.nodes(data=True)[n]["y"] = G.nodes(data=True)[n]["pos"][1]

pos_to_xy(G)
    
plot_graphs(G)

# Create x number of graphs and saves them
def save_graphs(graph, num_graphs):
    for n in range(num_graphs):
        G = graph
        with open(f'train_square{n}.pickle', 'wb') as handle:
            pickle.dump(G, handle)
            

# Create x number of graphs and saves them
def create_graphs(num_graphs):
    for n in range(num_graphs):
        G = create_graph()
        with open(f'train_square{n}.pickle', 'wb') as handle:
            pickle.dump(G, handle)
            
create_graphs(10)
            
G = nx.to_numpy_matrix(G)

pos_test = np.array([[0, 1, 2], [0, 10, 0]])

        
def get_graph(adj, node_pos):
  """ get a graph from zero-padded adj """
  # remove all zeros rows and columns
  #adj = adj[~np.all(adj == 0, axis=1)]
  #adj = adj[:, ~np.all(adj == 0, axis=0)]
  adj = np.asmatrix(adj)
  G = nx.from_numpy_matrix(adj)
  
  # //Oliver converts the node list to a dict
  dict_pos = pos_list_to_dict(node_pos)
  # // Oliver Combines the graph with it's positions 
  nx.set_node_attributes(G, dict_pos)
  
  return G

def pos_list_to_dict(pos):
    """converts pos list to dict """
    pos = np.transpose(pos)
    dict_pos = {}
    for i, p in enumerate(pos):
        dict_pos[i] = {"x": p[0], "y": p[1]}
    return dict_pos

def create_triangles():
    G = nx.Graph()
    
    G.add_node(1, x=10, y=10)
    G.add_node(2, x=10, y=20)
    G.add_node(3, x=20, y=10)
    
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)    

    return G