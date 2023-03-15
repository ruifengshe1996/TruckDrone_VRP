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

    from modules.set_vrp.make_abstract_graph import make_abstract_graph
    print('\n making abstract graph')
    abstract_G = make_abstract_graph(vrp)

    with open(os.path.join('cases',casename,'data','abstract_graph.pkl'),'wb') as f:
            pickle.dump(abstract_G,f)

    if heuristics:        
        # %% solve using savings heuristics
        import modules.set_vrp.solve_savings as savings

        print('\n solving using savings heurstics')
        s = savings.solve(vrp, abstract_G)

        print('\n saving solution to pkl')
        with open(os.path.join('cases',casename,'data','savings_raw.pkl'),'wb') as f:
            pickle.dump(s,f)
    
    else:
        # %% solve using gurobi
        import modules.set_vrp.solve_gurobi as gurobi
        print('\n solving using gurobi')
        g = gurobi.solve(vrp,abstract_G,
                         init_solution=None,
                         time_limit= 30)

        print('\n saving gurobi solution')
        # save the gurobi model
        # gurobi_sol_path = os.path.join('cases',casename,'data','gurobi_raw.sol')
        # g.model.write(gurobi_sol_path)
            
        # save the wrapper seperately, because somehow model cannot be pickled
        # g.model = None
        with open(os.path.join('cases',casename,'data','gurobi_raw.pkl'),'wb') as f:
            pickle.dump(g,f)