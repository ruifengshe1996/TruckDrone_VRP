'''
current version: greedily fill trucks with ascending or efficiency or capacity
consider dynamic programming
'''

# %%
# -*- coding: utf-8 -*-

"""
Created on Mon Dec 19 12:09:03 2022
@author: ruife
"""

import networkx as nx
import gurobipy as gp
import numpy as np
import pandas as pd
import copy 

import itertools
import heapq

from classes.tdvrp import TDVRP
from utils.pandas import find_df_min

class SavingsHeuristics(object):

    ''' savings heuristics for set vrp
    
    @param vrp: TDVRP class, master problem class
    @param G: nx.DiGraph, abstract graph
    
    A optimal path, giving depot-first-last-depot, is a dictionary:
    {(first_cluster, last_cluster): {'path': [i_first, ..., i_last],
                                      'out' : [j_first, ..., j_last]
                                      'demand': val,
                                      'cost': val}}
    '''
    
    def __init__(self,vrp:TDVRP, G:nx.DiGraph) -> None:
        
        self.vrp = vrp
        self.G = G
        # initial single-region paths
        # computes depot - region -depot cost via the best pair of access points
        depot_dr_cost = vrp.depot_dr_cost
        intra_dr_cost = vrp.intra_dr_cost

        self.region_assigend = dict.fromkeys(vrp.regions.keys(),False)

        paths_dict = dict()
        for region_key in vrp.regions:
            
            cost = pd.DataFrame(intra_dr_cost[region_key],copy = True)
            
            for i,j in itertools.product(cost.index, repeat = 2):
                cost.loc[i,j] += depot_dr_cost[region_key].loc[i,self.vrp.depot] + \
                                    depot_dr_cost[region_key].loc[j,self.vrp.depot]
            
            val,i,j = find_df_min(cost)

            path = {'demand':vrp.regions[region_key].num_demand,
                    'region':[region_key],
                    'in':[i],
                    'out':[j],
                    'cost':val}
            
            paths_dict[(region_key,region_key)] = path
        self.paths_dict = paths_dict


        return 


    def solve(self):
        ''' main iterations of the savings heuristics
        In each iteration:
            
            1. update the savings dictionary with the current merged paths
            2. if two paths can be merged, do the merge
            3. if no paths can be merged, assign the path with largest demand to the current vehicle
            4. if some clusters failed to be assigned, do some clean-ups

        '''
        
        truck_list = self._get_truck_queue('efficiency')
        self.truck_assignment = dict()
        savings_dict = dict()

        # terminate code: 0-unfinished; 1-finished; 2-leftover
        terminate = 0

        cur_truck = truck_list.pop(0)
        cur_capacity = self.vrp.truck_capacity[cur_truck]
        print('total clusters to assign: ', len(self.paths_dict))

        while terminate == 0:
            print('remaining paths: ', list(self.paths_dict.keys()))
            # update savings
            print('update savings')
            for p1,p2 in itertools.permutations(self.paths_dict.keys(), r = 2):
                self._update_savings(savings_dict,p1,p2)
            
            # update solution
            merged = False
            savings_list = [[val['val'], key] for key,val in savings_dict.items()]
            heapq._heapify_max(savings_list)
            
            while savings_list:

                val, merge_pair = heapq._heappop_max(savings_list)
                p1, p2 = find_path_pair(self.paths_dict, merge_pair)
                if not p1:
                    continue
                
                demand = self.paths_dict[p1]['demand'] + self.paths_dict[p2]['demand']
                # check capacity constraint
                if demand > cur_capacity:
                    continue

                # valid merge, merge p1 and p2. size of self.paths_dict will decrease by 1
                print('merging paths', merge_pair)
                self._merge_path(p1,p2,savings_dict)
                merged = True
                break

            # a merge is done in ths iteration, skip the rest
            if merged:
                continue

            # no merge is done
            # assign the path with most demand to the current truck
            demand_list = list(map(lambda x: self.paths_dict[x]['demand'],self.paths_dict))
            max_path_key = list(self.paths_dict.keys())[demand_list.index(max(demand_list))]
            max_demand = self.paths_dict[max_path_key]['demand']

            self.truck_assignment[cur_truck] = self.paths_dict[max_path_key]
            self.paths_dict.pop(max_path_key)

            print('assigning demand {} to truck {} with capacity {}'.format(max_demand, cur_truck, cur_capacity))
            # check if all paths are assigned
            if not self.paths_dict:
                terminate = 1
                break

            # check if there are more available trucks
            if truck_list:
                # mark new current truck
                cur_truck = truck_list.pop(0)
                cur_capacity = self.vrp.truck_capacity[cur_truck]
            else:
                terminate = 2
                break
        

        # some regions remain but ran out of trucks
        if terminate == 2:
            #TODO: clean up the leftovers
            pass
        

        if terminate == 1:
            print('all clusters assigned')

            # ============= validate solution =========================

            for truck,sol in self.truck_assignment.items():
                load = sum([self.vrp.regions[region].num_demand for region in sol['region']])
                assert load <= self.vrp.truck_capacity[truck], 'capacity violated for truck {}'.format(truck)

            return
        


        return

    # ======================= body functions ========================      

    def _get_truck_queue(self,order:str = 'efficiency') -> list:
        ''' returns a list of trucks, sorted with the given order
        '''
        if order == 'efficiency':
            sort_order = {key:self.vrp.truck_capacity[key]/self.vrp.truck_cost[key]  
                                    for key in self.vrp.truck_capacity.keys()}
        if order == 'capacity':
            sort_order = self.vrp.truck_capacity

        truck_order = sorted(list(self.vrp.truck_capacity.keys()),
                             key = lambda x: sort_order[x], 
                             reverse= True)
        return truck_order

    def _update_savings(self,savings_dict:dict, p1:tuple, p2:tuple) -> None:
        '''
        compute savings for each pair of paths (start1,end1) (start2,end2)
        as the saving to merge into (start1,end2)
        '''

        start_1,end_1 = p1; start_2,end_2 = p2

        i1 = self.paths_dict[p1]['in'][-1]
        j1 = self.paths_dict[p1]['out'][-1]
        i2 = self.paths_dict[p2]['in'][0]
        j2 = self.paths_dict[p2]['out'][0]
                    
        # 
        # if the same i1,j2 has been evaluated, skip
        if (end_1,start_2) in savings_dict:
            record = savings_dict[(end_1,start_2)]
            if record['i1'] == i1 and record['j2'] == j2:
                return
        
        # evaluate saving for the current i1, j2 of the paths
        saving_df = pd.DataFrame(0,
                                index = self.vrp.regions[end_1].boundary,
                                columns = self.vrp.regions[start_2].boundary)
        
        fixed = self.vrp.depot_dr_cost[end_1].loc[j1,self.vrp.depot] + \
                self.vrp.depot_dr_cost[start_2].loc[i2,self.vrp.depot] + \
                self.vrp.intra_dr_cost[end_1].loc[i1,j1] + \
                self.vrp.intra_dr_cost[start_2].loc[i2,j2]
                                    
        for j1_,i2_ in itertools.product(saving_df.index, saving_df.columns):
            saving_df.loc[j1_,i2_] = fixed - \
                                        self.vrp.intra_dr_cost[end_1].loc[i1,j1_] - \
                                        self.vrp.intra_dr_cost[start_2].loc[i2_,j2] - \
                                        self.vrp.inter_dr_cost[(end_1,start_2)].loc[j1_,i2_]

        # reuse name of j1, i2 as the ought-to-be points if merged                             
        val,j1,i2 = find_df_min(saving_df)
        savings_dict[(end_1,start_2)] = {'val':val,'i1':i1,'j1':j1,'i2':i2,'j2':j2}

        return 

    def _merge_path(self,p1:tuple,p2:tuple,savings_dict:dict):
        ''' merge the given two paths and update the relavent variabels
        '''

        saving = savings_dict[(p1[1],p2[0])]
        path1 = self.paths_dict[p1]; path2 = self.paths_dict[p2]

        path = {'demand':path1['demand'] + path2['demand'],
                'region': list(set(path1['region'] + path2['region'])),
                'in':path1['in'] + [saving['i2']] + path2['in'][1:],
                'out':path1['out'][:-1] +[saving['j1']] + path2['out'],
                'cost':path1['cost'] + path2['cost'] - saving['val']}

        # add the new path into dictionary
        self.paths_dict[(p1[0],p2[1])] = path
        # remove the pre-merge paths
        self.paths_dict.pop(p1)
        self.paths_dict.pop(p2)
        # remove corresponding savings
        remove_keys = [key for key in savings_dict.keys() if key[0] == p1[1] or key[1] == p2[0]]
        for key in remove_keys: savings_dict.pop(key)

        return

    
# %% utility functions ==========================

def find_path_pair(paths_dict:dict,merge_pair:tuple):
    '''
    find the pair of paths from the paths_dict dictionary that ends and starts with the given region pair
    '''
    for p1,p2 in itertools.permutations(paths_dict.keys(), r = 2):
        # the solution should be unique
        if (p1[1],p2[0]) == merge_pair:
            return (p1,p2)
    # no matching path pair

    return (None,None)

def retrieve_path(s:SavingsHeuristics) -> dict[list]:
    truck_assignment = s.truck_assignment
    # reform paths
    depot = s.vrp.depot
    paths_dict = dict()
    for key,sol in truck_assignment.items():

        path = [depot]
        for start,end in zip(sol['in'],sol['out']):
            path.append(start); path.append(end)
        path.append(depot)

        paths_dict[key] = path
    return paths_dict

def get_total_cost(s:SavingsHeuristics):
    truck_assignment = s.truck_assignment
    total_cost = 0
    for key,sol in truck_assignment.items():
        total_cost += sol['cost']
        total_cost += s.vrp.truck_cost[key]
    return total_cost

# %% external call function
def solve(vrp:TDVRP, G:nx.DiGraph) -> dict:
    ''' a wrapper function
    '''
    s = SavingsHeuristics(vrp,G)
    s.solve()
 
    return s

