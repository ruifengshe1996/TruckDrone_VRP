import importlib
import numpy as np
import os
import osmnx as ox
import matplotlib.pyplot as plt
import pickle


def run(casename, **kwargs):

    new = kwargs['new']
    grid = kwargs['grid']

    # ==================================================

    ox.settings.use_cache=True
    ox.settings.log_console = False
    
    savepath = os.path.join('cases',casename,'data','demand.pkl')

    if not new and os.path.exists(savepath):

        print(savepath,' exists, skipping this task: ', __name__)
        return

    config = importlib.import_module('.config','cases.'+casename)



    print('loading in configures')
    from shapely.geometry import Point
    centers_coord = config.demand_centers
    if grid:
        # coords in m
        centers = np.array(centers_coord)
    else:
        # coords in long-lat
        # project the centers from longitude-lattitude into m-based coordinates
        centers = []
        for coord in centers_coord:
            p,_ = ox.projection.project_geometry(Point(coord))
            centers.append((p.x,p.y))
        centers = np.array(centers)
    
    print('loading in graph')
    graph_file = os.path.join('cases',casename,'data','network.pkl')
    with open(graph_file,mode = 'rb') as f:
        G = pickle.load(f)
    
    print('generate demand')
    from sklearn.datasets import make_blobs    
    data,labels = make_blobs(n_samples = config.number_of_demand, 
                             n_features = 2, 
                             centers = centers,
                             cluster_std = config.demand_std,
                             random_state=np.random.RandomState(config.random_seed)
                             )    

    from classes.demand import Demand
    demand = Demand(data,labels)
    
    if demand.clustered:
        print('demand is clustered')
    else:
        print('clustering demand')
        demand.cluter()
    


    print('validating clusters and creating partitioning polygons')
    demand.cluster2shape()
    demand.validate_clusters(prune=True)





    # plot the output
    print('plotting the generated demand and cluster/polygon')
    from utils.plotting import plot_demand,plot_polygon,plot_graph
    
    fig,ax = plt.subplots()


    fig,ax = plot_graph(G,ax)

    fig,ax = plot_demand(demand,ax)
    for key,polygon in demand.polygons.items():
        fig,ax = plot_polygon(polygon,ax)
    
    fig_savefile = os.path.join('cases',casename,'output','demand_polygon.png')
    fig.savefig(fig_savefile)



    # save the class
    print('saving model')
    
    with open(savepath,mode = 'wb') as f:
        pickle.dump(demand,f)
    
    return





    
    
