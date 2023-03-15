# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:18:17 2022

@author: ruife
"""
import numpy as np
import pickle
import os
import importlib
from itertools import product

casename = 'Rantoul'
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
abstract_G = make_abstract_graph(vrp)

from utils.network import complete_path
import matplotlib.pyplot as plt
import utils.plotting
    
# %% solve using savings heuristics

import modules.set_vrp.solve_savings as savings
s = savings.solve(vrp, abstract_G)
paths_dict = savings.retrieve_path(s)
print(paths_dict)

complete_paths_dict = {}
for key,path in paths_dict.items():
    complete_paths_dict[key] = complete_path(vrp.G,path)
total_cost_savings = savings.get_total_cost(s)
with open(os.path.join('cases',casename,'data','savings.pkl'),'wb') as f:
    pickle.dump({'cost':total_cost_savings,'solution':complete_paths_dict},f)

fig,ax = plt.subplots()
fig,ax = utils.plotting.plot_graph(vrp.G, ax)
for path in complete_paths_dict.values():
    fig,ax = utils.plotting.plot_path(vrp.G, path, ax)
fig.savefig(os.path.join('cases',casename,'output','savings.png'))

# %% solve using gurobi
import modules.set_vrp.solve_gurobi as gurobi
g = gurobi.solve(vrp,abstract_G,init_solution=None)
paths_dict = gurobi.retrieve_path(g)

complete_paths_dict = {}
for key,path in paths_dict.items():
    complete_paths_dict[key] = complete_path(vrp.G,path)
total_cost_gurobi = gurobi.get_total_cost(g)
with open(os.path.join('cases',casename,'data','gurobi.pkl'),'wb') as f:
    pickle.dump({'cost':total_cost_gurobi,'solution':complete_paths_dict},f)

fig,ax = plt.subplots()
fig,ax = utils.plotting.plot_graph(vrp.G, ax)
for path in complete_paths_dict.values():
    fig,ax = utils.plotting.plot_path(vrp.G, path, ax)
fig.savefig(os.path.join('cases',casename,'output','gurobi.png'))

print('\n')
print('savings solution: ', total_cost_savings)
print('gurobi solution: ', total_cost_gurobi)

