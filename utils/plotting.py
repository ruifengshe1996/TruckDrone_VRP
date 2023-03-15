import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx


# ================== plain graph ==============================================

def plot_graph(G,ax):
    pos_dict = {node:(G.nodes[node]['x'],G.nodes[node]['y']) for node in G.nodes}
    
    nx.drawing.draw_networkx(G,pos = pos_dict,ax = ax,
                                    node_size = 0,
                                    node_color = 'r',
                                    edge_color = 'grey',
                                    with_labels= False,)
    return ax.figure, ax

# ================== Demand ===================================================
from classes.demand import Demand

def plot_demand(demand:Demand, ax:plt.Axes):
    
    
    ax.scatter(demand.data[:,0],demand.data[:,1],
                c = 'black',
                s = 1,
                zorder = 3)
    return ax.figure, ax

def plot_polygon(polygon,ax:plt.Axes,fill = False):
    
    if fill:
        pass
    else:
        x,y = polygon.exterior.xy
        ax.plot(x,y,c = 'black',zorder = 2)
    
    return ax.figure,ax




# ================== DeliveryRegion ===========================================
from classes.delivery_region import DeliveryRegion

def plot_dr_graph(dr:DeliveryRegion, ax:plt.Axes):    
    ''' a wrapper of the ox.plot.plot_graph function
    '''
    node_gdf = ox.graph_to_gdfs(dr.G, edges = False)
    num_nodes = len(node_gdf.index)

    node_size = [0] * num_nodes

    for i in range(num_nodes):
        if node_gdf.index[i] in dr.boundary:
            node_size[i] = 15

    pos_dict = {node:(dr.G.nodes[node]['x'],dr.G.nodes[node]['y']) for node in dr.G.nodes}

    nx.drawing.draw_networkx(dr.G,pos = pos_dict,ax = ax,
                                    node_size = node_size,
                                    node_color = 'r',
                                    edge_color = 'grey',
                                    with_labels= False)
   
    return ax


def plot_dr_graph_nodes(dr:DeliveryRegion, ax:plt.Axes, nodelist:list, node_color = 'g'):    
    ''' a wrapper of the ox.plot.plot_graph function
    '''

    pos_dict = {node:(dr.G.nodes[node]['x'],dr.G.nodes[node]['y']) for node in dr.G.nodes}

    nx.drawing.draw_networkx_nodes(dr.G,pos = pos_dict,ax = ax,
                                    nodelist = nodelist,
                                    node_size = 15,
                                    node_color = node_color)
   
    return ax




def plot_path(G:nx.Graph,path:list,ax:plt.Axes):

    
    X = [G.nodes[node]['x'] for node in path if node not in [None,'dummy']]
    Y = [G.nodes[node]['y'] for node in path if node not in [None,'dummy']]


    ax.plot(X,Y)
    return ax.figure, ax


