import osmnx as ox
import networkx as nx
import pandas as pd
import os
import importlib
import numpy as np

def run(casename:str,**kwargs):

    new = kwargs['new']
    grid = kwargs['grid']

    # ==================================================

    savepath = os.path.join('cases',casename,'data','network.pkl')
    
    if (not new) and os.path.exists(savepath):
        print(savepath,' exists, skipping this task: ', __name__)
        return


    ox.settings.use_cache=True
    ox.settings.log_console = True

    config = importlib.import_module('.config','cases.'+casename)

    north, south, east, west = config.bounding_box

    # retrieve from openstreetmap the raodway map
    # can take a long time
    if grid:
        step = config.road_spacing
        G = nx.grid_2d_graph(np.arange(west, east + step,step),
                             np.arange(south, north + step,step),
                             periodic = False,
                             create_using = nx.Graph)
        from utils.network import prepare_grid_network
        G = prepare_grid_network(G)       
        
    else:
        G = ox.graph_from_bbox(north,south,east,west,
                            network_type='drive',
                            simplify=True)
        from utils.network import prepare_network
        G = prepare_network(G)
    
    # plotting
    print('saving plot')
    import matplotlib.pyplot as plt
    from utils.plotting import plot_graph
    fig,ax = plt.subplots()
    fig,ax = plot_graph(G,ax)
    fig_savepath = os.path.join('cases',casename,'output','network.png')
    plt.savefig(fig_savepath)
    
    
    # saving
    print('saving network')
    import pickle
    with open(savepath,mode = 'wb') as f:
        pickle.dump(G,f)

    return

