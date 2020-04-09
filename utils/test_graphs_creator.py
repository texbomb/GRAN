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

os.chdir("../data/triangles")

# Creates graph
def create_graph():
    G = nx.Graph()
    
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=10)
    G.add_node(3, x=2, y=0)
    
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    
    return G

G = create_graph()

G.nodes(data=True)

plot_graphs(G)

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
    plt.xlim(x_min-1, x_max+1)
    plt.ylim(y_min-1, y_max+1)
    plt.show()

plot_graphs(create_graph())

# Create x number of graphs and saves them
def create_graphs(num_graphs):
    for n in range(num_graphs):
        G = create_graph()
        with open(f'train{n}.pickle', 'wb') as handle:
            pickle.dump(G, handle)
            
            
G = nx.to_numpy_matrix(G)
G = nx.from_numpy_matrix(G)
        
def get_graph(adj, node_pos):
  """ get a graph from zero-padded adj """
  # remove all zeros rows and columns
  #adj = adj[~np.all(adj == 0, axis=1)]
  #adj = adj[:, ~np.all(adj == 0, axis=0)]
  adj = np.asmatrix(adj)
  G = nx.from_numpy_matrix(adj)
  return G

G = get_graph(G)

pos = {
       0: {"x": 0, "y": 0},
       1: {"x": 1, "y": 10},
       2: {"x": 2, "y": 0},
       }

nx.set_node_attributes(G, pos)

G.nodes(data=True)

def pos_list_to_dict(pos):
    pos = np.transpose(pos)
    dict_pos = {}
    for i, p in enumerate(pos):
        dict_pos[i] = {"x": p[0], "y": p[1]}
    return dict_pos

pos_test = np.array([[0, 1, 2], [0, 10, 0]])

pos_list_to_dict(pos_test)
