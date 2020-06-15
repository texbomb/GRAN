import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_graph_list(G_list,
                    row,
                    col,
                    fname='exp/gen_graph.png',
                    layout='spring',
                    is_single=False,
                    k=1,
                    node_size=55,
                    alpha=1,
                    width=1.3):
  plt.switch_backend('agg')
  for i, G in enumerate(G_list):
    plt.subplot(row, col, i + 1)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # plt.axis("off")

    # turn off axis label
    plt.xticks([])
    plt.yticks([])

    if layout == 'spring':
      pos = nx.spring_layout(
          G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
    elif layout == 'spectral':
      pos = nx.spectral_layout(G)

    if is_single:
      # node_size default 60, edge_width default 1.5
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=node_size,
          node_color='#336699',
          alpha=1,
          linewidths=0,
          font_size=0)
      nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
    else:
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=1.5,
          node_color='#336699',
          alpha=1,
          linewidths=0.2,
          font_size=1.5)
      nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

  plt.tight_layout()
  plt.savefig(fname, dpi=300)
  plt.close()


def draw_graph_list_separate(G_list,
                    fname='exp/gen_graph',
                    layout='spring',
                    is_single=False,
                    k=1,
                    node_size=55,
                    alpha=1,
                    width=1.3):
  
  for i, G in enumerate(G_list):
    plt.switch_backend('agg')
    
    plt.axis("off")

    # turn off axis label
    # plt.xticks([])
    # plt.yticks([])

    if layout == 'spring':
      pos = nx.spring_layout(
          G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
    elif layout == 'spectral':
      pos = nx.spectral_layout(G)
    elif layout == 'position':
      pos = get_pos(G)


    if is_single:
      #color map
      node_cat = list(nx.get_node_attributes(G, 'alpha').values())

      # node_size default 60, edge_width default 1.5
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=node_size,
          node_color=node_cat,
          cmap=plt.cm.tab20c,
          alpha=1,
          linewidths=0,
          font_size=0)      
      nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
    else:
      node_cat = list(nx.get_node_attributes(G, 'alpha').values())
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=1.5,
          node_color=node_cat,
          cmap=plt.cm.tab20c,
          alpha=1,
          linewidths=0.2,
          font_size=1.5
          )
      nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    # Added to set the axis properly 
    x_pos = []
    y_pos = []
    
    for k in pos.keys():
        x_pos.append(pos[k][0])        
        y_pos.append(pos[k][1])   
    
    x_min = min(x_pos)
    x_max = max(x_pos)

    y_min = min(y_pos)
    y_max = max(y_pos)

    plt.draw()
    plt.xlim(x_min*0.6, x_max*1.2)
    plt.ylim(y_min*0.6, y_max*1.2)
    plt.tight_layout()
    plt.savefig(fname+'_{:03d}.png'.format(i), dpi=300)
    plt.close()

def get_pos(G):
    
    nodes = G.nodes
    
    pos = {}
    
    for n in nodes:
        pos[n] = [G.nodes(data=True)[n]["x"], G.nodes(data=True)[n]["y"]]
        
    return pos
