import numpy as np
import osmnx as ox
import networkx as nx
import os
import importlib
import pickle



def run(casename,new,**kwargs):
    
    heuristics = kwargs['heuristics']
    # ==================================================

    ox.settings.use_cache=True
    ox.settings.log_console = False

    vrp_file = os.path.join('cases',casename,'data','vrp.pkl')
    
    if not new and os.path.exists(vrp_file):

        print(vrp_file,' exists, skipping this task: ', __name__)
        return


    config = importlib.import_module('.config','cases.'+casename)

    graph_file = os.path.join('cases',casename,'data','network.pkl')
    with open(graph_file,mode = 'rb') as f:
        G = pickle.load(f)
        
    # read in demand data
    print('loading in data from file')
    demand_file = os.path.join('cases',casename,'data','demand.pkl')
    with open(demand_file,mode = 'rb') as f:
        demand = pickle.load(f)
        



    # ======== initialize vrp and register parameters ===========================
    print('initializing VRP')
    from classes.tdvrp import TDVRP

    vrp = TDVRP(demand,G)
    vrp.register_depot(depot_coord=config.depot_coord)
    vrp.make_from_polygon(heuristics)
    # =========== evaluate cost ==============================

    print('\n evaluating depot-dr cost for all delivery regions')
    
    vrp.eval_depot_dr_cost()

    print('\n evaluating inter-dr costs between all dr pairs')
    vrp.eval_inter_dr_cost()


    print('\n evaluating intra-dr costs for all delivery regions')
    vrp.eval_intra_dr_cost()

    # ========== save model ===============================
    print('\n saving delivery regions')
    with open(vrp_file,mode = 'wb') as f:
        pickle.dump(vrp,f)

    
    
    return