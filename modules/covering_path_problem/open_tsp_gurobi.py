# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 01:26:13 2022

@author: ruife
"""

import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
import numpy as np


def solve(nodes:list, dist:dict):
    

    #  ========== special cases ==============

    assert len(nodes) >= 3, 'most have more than three nodes, otherwise will be infeasible'

    # tested with Python 3.7 & Gurobi 9.0.0
    
    m = gp.Model()
    m.setParam('OutputFlag',0)

    # Variables: is city 'i' adjacent to city 'j' on the tour?
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')
    
    # Symmetric direction: Copy the object
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]  # edge in opposite direction
    
    # Constraints: two edges incident to each city
    cons = m.addConstrs(vars.sum(c, '*') == 2 for c in nodes)
    
    # Callback - use lazy constraints to eliminate sub-tours
    
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                                 if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = longestsubtour(selected)
            if len(tour) < len(nodes):
                # add subtour elimination constr. for every pair of cities in subtour
                model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                             <= len(tour)-1)
    
    # Given a tuplelist of edges, find the shortest subtour
    
    def longestsubtour(edges):
        unvisited = nodes[:]
        cycle = nodes[:] # Dummy - guaranteed to be replaced
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*')
                             if j in unvisited]
            if len(thiscycle) <= len(cycle):
                cycle = thiscycle # New shortest subtour
        return cycle
    
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim)
    
    # Retrieve solution
    
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    
    tour = longestsubtour(selected)
    
    if len(tour) == len(nodes):
        val = m.getObjective().getValue()
        return val, tour
    else:
        return np.inf, None