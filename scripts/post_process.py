import numpy as np
import pickle
import os
import importlib
from itertools import product


def run(casename,**kwargs):
    
    heuristics = kwargs['heuristics']

    # ================== load in pre-computed cost data =======================
    # load in cost matrices as pd.DataFrame

    config = importlib.import_module('.config','cases.'+casename)

    print('\n ========= loading in cost data ==============')
    vrp_file = os.path.join('cases',casename,'data','vrp.pkl')
    with open(vrp_file,mode = 'rb') as f:
        vrp = pickle.load(f)

    # update some vrp-level pameters
    vrp.register_params(num_trucks = config.num_trucks,
                        truck_capacity = config.truck_capacity,
                        truck_cost = config.truck_cost)



    from utils.network import complete_path
    import matplotlib.pyplot as plt
    import utils.plotting



    # refactor region-level solutions


    # plot and save global solutions
    if heuristics:        
        # %% solve using savings heuristics
        import modules.set_vrp.solve_savings as savings

        with open(os.path.join('cases',casename,'data','savings_raw.pkl'),'rb') as f:
            s = pickle.load(f)

        print('\n retriving path solution')
        paths_dict = savings.retrieve_path(s)

        print('\n retriving complete path on original network')
        complete_paths_dict = {}
        for key,path in paths_dict.items():
            complete_paths_dict[key] = complete_path(vrp,path)
        total_cost_savings = savings.get_total_cost(s)

        print('\n saving solution to pkl')
        with open(os.path.join('cases',casename,'data','savings.pkl'),'wb') as f:
            pickle.dump({'cost':total_cost_savings,'solution':complete_paths_dict},f)

        print('\n plotting solution')
        fig,ax = plt.subplots()
        fig,ax = utils.plotting.plot_graph(vrp.G, ax)
        for path in complete_paths_dict.values():
            fig,ax = utils.plotting.plot_path(vrp.G, path, ax)
        fig.savefig(os.path.join('cases',casename,'output','savings.png'))
        print('\n savings cost: ', total_cost_savings)
    
    else:
        # %% solve using gurobi
        import modules.set_vrp.solve_gurobi as gurobi
        
        with open(os.path.join('cases',casename,'data','gurobi_raw.pkl'),'rb') as f:
            g = pickle.load(f)

        print('\n retriving path solution')
        paths_dict = gurobi.retrieve_path(g)
        print(paths_dict)
        print('\n retriving complete path on original network')
        complete_paths_dict = {}
        for key,path in paths_dict.items():
            complete_paths_dict[key] = complete_path(vrp,path)
        # print(complete_paths_dict)


        print('\n plotting solution')
        fig,ax = plt.subplots()
        fig,ax = utils.plotting.plot_graph(vrp.G, ax)
        for path in complete_paths_dict.values():
            fig,ax = utils.plotting.plot_path(vrp.G, path, ax)
        fig.savefig(os.path.join('cases',casename,'output','gurobi.png'))
        
        total_cost_gurobi = g.obj
        print('\n saving solution to pkl')
        print('\n gurobi cost: ', total_cost_gurobi)
        with open(os.path.join('cases',casename,'data','gurobi.pkl'),'wb') as f:
            pickle.dump({'cost':total_cost_gurobi,'solution':complete_paths_dict},f)

