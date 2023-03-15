# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:52:28 2022

@author: ruife
"""

import networkx as nx

from itertools import product

from classes.tdvrp import TDVRP

def make_abstract_graph(vrp:TDVRP) -> nx.DiGraph:
    
    
    depot_dr_cost = vrp.depot_dr_cost
    inter_dr_cost = vrp.inter_dr_cost
    # intra_dr_cost = vrp.intra_dr_cost
    
    # ============= create a abstract graph for easy edge query ==============
    print('\n ========= creating abstract graph for VRP ==============')

    G = nx.DiGraph()
    
    def add_node(node):
        G.add_node(node,x = vrp.G.nodes[node]['x'],y = vrp.G.nodes[node]['y'])
    
    # add depot node    
    depot = vrp.depot
    add_node(depot)
    for region_key,region in vrp.regions.items():
        
        # add boundary nodes
        for node in region.boundary:
            add_node(node)
        
        # add dummy node
        G.add_node('dummy_'+str(region_key),
                   x = region.polygon.centroid.x,
                   y = region.polygon.centroid.y)
    
    
    # add depot-dr edges
    for region_key,cost in depot_dr_cost.items():
        for node in cost.index:
            G.add_edge(depot,node,weight = cost.loc[node,depot])
            G.add_edge(node,depot,weight = cost.loc[node,depot])
            
    # add dr-dummy edges
    for region_key,region in vrp.regions.items():
        for node in region.boundary:
            G.add_edge(node, 'dummy_'+str(region_key), weight = 1e-6)
            G.add_edge('dummy_'+str(region_key), node, weight = 1e-6)
            
    
    # add inter-dr edges
    for region_key,cost in inter_dr_cost.items():
        for u,v in product(cost.index, cost.columns):
            G.add_edge(u,v,weight = cost.loc[u,v])
            G.add_edge(v,u,weight = cost.loc[u,v])
            
    
    # convert to directed graph
    
    print('total nodes : ', G.number_of_nodes())
    print('total directed edges: ', G.number_of_edges())
    
    return G