import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

def get_city(point, distance):
    # Imports the a given city from point and distance around the point. 
    G_ox = ox.graph_from_point(point, distance=distance, network_type='drive')

    # Creates a nx Graph and inputs the node with positional data and edge weigts 
    G = nx.Graph()
    for n in G_ox.nodes:
        x = G_ox.nodes(data=True)[n]["x"] - point[1]
        y = G_ox.nodes(data=True)[n]["y"] - point[0]
        G.add_node(n, pos=(x, y))
    for e in G_ox.edges:
        G.add_edge(e[0], e[1])
    for e in G_ox.edges:
        G[e[0]][e[1]]['weight'] = G_ox.get_edge_data(e[0], e[1])[0]["length"]
        
    # returns the nx graph
    return G


G = get_city((55.6867243, 12.5700724), 1000)
