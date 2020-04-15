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
import math
import copy

os.chdir("../data/triangles")

# Creates graph
def create_graph():
    G = nx.Graph()
    
    G.add_node(1, x=100, y=300)
    G.add_node(2, x=300, y=100)
    G.add_node(3, x=100, y=100)
    G.add_node(4, x=300, y=300)
    
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 4)
    G.add_edge(4,3)
    
    return G

G = create_graph()

def rotate_graph(G,angle, point = 'in_place'):
    """
    Rotates a networkx graph object around a point with a given angle in degrees
    Rotates clockwise around the specified point, if nothing is given, rotate in place.
    Returns a new networkX graph with the rotated object
    """
    angle *= math.pi / 180
    dic = copy.deepcopy( dict(G.nodes(data=True)) )

    if point == 'in_place':
        point = ( sum(d['x'] for d in dic.values() if d) / len(dic.keys()), sum(d['y'] for d in dic.values() if d) / len(dic.keys())  )

    for key in dic:    
        dic[key]['x'] =  round(math.cos(angle) * (dic[key]['x']-point[0]) -  math.sin(angle) * (dic[key]['y']-point[1]) + point[0] )

        dic[key]['y'] =  round(math.sin(angle) * (dic[key]['x']-point[0]) + math.cos(angle) * (dic[key]['y']-point[1]) + point[1] )

    H = copy.deepcopy(G)
    nx.set_node_attributes(H,dic)
    return H.nodes(data=True)

def scale_graph(G, scale, in_place = True):
    """
    Scales a networkx graph object by scaling the distance from each node to the center of the graph by 'scale'.
    If in_place is set to true, scales the graph in place
    """
    dic = copy.deepcopy( dict(G.nodes(data=True)) )

    if in_place:
         point = ( sum(d['x'] for d in dic.values() if d) / len(dic.keys()), sum(d['y'] for d in dic.values() if d) / len(dic.keys())  )
    else:
        point = (0,0)
    for key in dic:    
        dic[key]['x'] =  ( dic[key]['x'] - point[0] ) * scale + point[0]

        dic[key]['y'] =  ( dic[key]['y'] - point[1] ) * scale + point[1]

    H = copy.deepcopy(G)
    nx.set_node_attributes(H,dic)
    return H


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
        with open(f'train_big{n}.pickle', 'wb') as handle:
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
