import math 
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
import networkx as nx
ox.config(log_console=True, use_cache=True)

# Source https://github.com/gboeing/osmnx/blob/a58aab5f8f73a5620247a849c755becfe770c065/osmnx/bearing.py


def get_bearing(origin_point, destination_point):
    """
    Calculate the bearing between two lat-lng points.
    Each tuple should represent (lat, lng) as decimal degrees.
    Parameters
    ----------
    origin_point : tuple
        (lat, lng)
    destination_point : tuple
        (lat, lng)
    Returns
    -------
    bearing : float
        the compass bearing in decimal degrees from the origin point
        to the destination point
    """
    if not (isinstance(origin_point, tuple) and isinstance(destination_point, tuple)):
        raise TypeError("origin_point and destination_point must be (lat, lng) tuples")

    # get latitudes and the difference in longitude, as radians
    lat1 = math.radians(origin_point[0])
    lat2 = math.radians(destination_point[0])
    diff_lng = math.radians(destination_point[1] - origin_point[1])

    # calculate initial bearing from -180 degrees to +180 degrees
    x = math.sin(diff_lng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_lng))
    initial_bearing = math.atan2(x, y)

    # normalize initial bearing to 0 degrees to 360 degrees to get compass bearing
    initial_bearing = math.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360

    return bearing


def add_edge_bearings(G):
    """
    Add each bearing attributes to all graph edges.
    Calculate the compass bearing from origin node to destination
    node for each edge in the directed graph then add each bearing
    as a new edge attribute.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    Returns
    -------
    G : networkx.MultiDiGraph
        graph with edge bearing attributes
    """
    for u, v, data in G.edges(keys=False, data=True):

        if u == v:
            # a self-loop has an undefined compass bearing
            data["bearing"] = np.nan

        else:
            # calculate bearing from edge's origin to its destination
            origin_point = (G.nodes[u]["y"], G.nodes[u]["x"])
            destination_point = (G.nodes[v]["y"], G.nodes[v]["x"])
            bearing = get_bearing(origin_point, destination_point)

            # round to thousandth of a degree
            data["bearing"] = round(bearing, 3)

    return G

# Source https://github.com/gboeing/osmnx-examples/blob/v0.11/notebooks/17-street-network-orientations.ipynb


def reverse_bearing(x):
    return x + 180 if x < 180 else x - 180

def count_and_merge(n, bearings):
    # make twice as many bins as desired, then merge them in pairs
    # prevents bin-edge effects around common values like 0° and 90°
    n = n * 2
    bins = np.arange(n + 1) * 360 / n
    count, _ = np.histogram(bearings, bins=bins)
    
    # move the last bin to the front, so eg 0.01° and 359.99° will be binned together
    count = np.roll(count, 1)
    return count[::2] + count[1::2]

# function to draw a polar histogram for a set of edge bearings
def polar_plot(ax, bearings, n=72, title=''):

    bins = np.arange(n + 1) * 360 / n
    count = count_and_merge(n, bearings)
    _, division = np.histogram(bearings, bins=bins)
    frequency = count / count.sum()
    division = division[0:-1]
    width =  2 * np.pi / n

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')

    x = division * np.pi / 180
    bars = ax.bar(x, height=frequency, width=width, align='center', bottom=0, zorder=2,
                  color='#003366', edgecolor='k', linewidth=0.5, alpha=0.7)
    
    ax.set_ylim(top=frequency.max())
    
    title_font = {'family':'Century Gothic', 'size':24, 'weight':'bold'}
    xtick_font = {'family':'Century Gothic', 'size':10, 'weight':'bold', 'alpha':1.0, 'zorder':3}
    ytick_font = {'family':'Century Gothic', 'size': 9, 'weight':'bold', 'alpha':0.2, 'zorder':3}
    
    ax.set_title(title.upper(), y=1.05, fontdict=title_font)
    
    ax.set_yticks(np.linspace(0, max(ax.get_ylim()), 5))
    yticklabels = ['{:.2f}'.format(y) for y in ax.get_yticks()]
    yticklabels[0] = ''
    ax.set_yticklabels(labels=yticklabels, fontdict=ytick_font)
    
    xticklabels = ['N', '', 'E', '', 'S', '', 'W', '']
    ax.set_xticklabels(labels=xticklabels, fontdict=xtick_font)
    ax.tick_params(axis='x', which='major', pad=-2)

def calculate_length(G):
    for i, edge in enumerate(G.edges(data=True)):
        edge0 = edge[0]
        x0 = G.nodes(data=True)[edge0]["x"]
        y0 = G.nodes(data=True)[edge0]["y"]
        
        edge1 = edge[1]
        x1 = G.nodes(data=True)[edge1]["x"]
        y1 = G.nodes(data=True)[edge1]["y"]    
        
        length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        G[edge0][edge1][0]['length'] = round(length, 3) * 10
    return G 
    
def plot_bearing(G):
    G = nx.MultiDiGraph(G)
    G = calculate_length(G)
    Gu = add_edge_bearings(G)
    bearings = {}
    city_bearings = []
    for u, v, k, d in Gu.edges(keys=True, data=True):
        city_bearings.extend([d['bearing']] * int(d['length']))
    b = pd.Series(city_bearings)
    bearings['plot'] = pd.concat([b, b.map(reverse_bearing)]).reset_index(drop='True')
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection':'polar'})
    polar_plot(ax, bearings['plot'].dropna(), title="place")