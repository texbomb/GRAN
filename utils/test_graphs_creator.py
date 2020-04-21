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

os.chdir(r"C:\Users\olive\OneDrive\Dokumenter\GitHub\GRAN\data\random")

# Creates graph
def create_graph():
    G = nx.Graph()
    
    G.add_node(1, x=10, y=10)
    G.add_node(2, x=20, y=20)
    G.add_node(3, x=10, y=20)
 
    
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)

    
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(1, 2)    

    return G

G = create_graph() 

def rotate_graph(G, degrees=0, point= 'in_place'):
    """
    Rotates a networkx graph object around a point with a given angle in degrees
    Rotates clockwise around the specified point, if nothing is given, rotate in place.
    Returns a new networkX graph with the rotated object
    """
    points = []
    dic = copy.deepcopy( dict(G.nodes(data=True)) )
    for key in dic:  
        points.append((dic[key]['x'],dic[key]['y']))

    if point == 'in_place':
        point = ( sum(d['x'] for d in dic.values() if d) / len(dic.keys()), sum(d['y'] for d in dic.values() if d) / len(dic.keys())  )
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(point)
    p = np.atleast_2d(points)
    new_points = np.squeeze((R @ (p.T-o.T) + o.T).T)
    for key in dic:  
        dic[key]['x'] = round(new_points[key-1][0])
        dic[key]['y'] = round(new_points[key-1][1])
    H = copy.deepcopy(G)
    nx.set_node_attributes(H,dic)
    return H
    #def Rotate2D(pts,cnt,ang=pi/4):
    #'''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    #return dot(pts-cnt,ar([[cos(ang),sin(ang)],[-sin(ang),cos(ang)]]))+cnt


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
    plt.xlim(x_min-x_min*0.1, x_max+x_max*0.1)
    plt.ylim(y_min-y_min*0.1, y_max+y_max*0.1)
    plt.show()

G =  nx.geographical_threshold_graph(50, 10)

def pos_to_xy(G):
    new_g = nx.Graph()
    for n in range(len(G.nodes(data=True))):
        x = G.nodes(data=True)[n]["pos"][0]
        y = G.nodes(data=True)[n]["pos"][1]
        new_g.add_node(n, x = x, y = y)
    for e in G.edges:
        new_g.add_edge(e[0], e[1])
    
    return new_g

G = pos_to_xy(G)
    
plot_graphs(G)

# Create x number of graphs and saves them
def save_graphs(graph, num_graphs):
    for n in range(num_graphs):
        G = graph
        with open(f'train_random_20{n}.pickle', 'wb') as handle:
            pickle.dump(G, handle)
            
save_graphs(G, 10)

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