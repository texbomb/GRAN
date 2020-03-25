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


# Create x number of graphs and saves them
def create_graphs(num_graphs):
    for n in range(num_graphs):
        G = create_graph()
        with open(f'train{n}.pickle', 'wb') as handle:
            pickle.dump(G, handle)
        
t = create_graph()

plot_graphs(t)
