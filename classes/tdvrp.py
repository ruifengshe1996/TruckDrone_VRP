import networkx as nx
import numpy as np
import pandas as pd


from tqdm import tqdm 
from itertools import combinations,product
from collections import Counter,defaultdict

from .demand import Demand
from .delivery_region import DeliveryRegion

from utils.network import nearest_nodes


class TDVRP(object):
    ''' set class of all delivery regions 
    '''
    def __init__(self, demand:Demand, G:nx.Graph) -> None:
        '''
        partition the graph G into disjoint subgraphs given demand data

        @input demand

        demand is assumed to have .labels, .chulls and .polygons, as disjoint shapely polygons
        '''

        self.demand = demand
        self.G = G
        
        self.regions: dict[int:DeliveryRegion]
        
        return

    def make_from_polygon(self,heuristics = True) -> None:
        ''' initialize all region classes for each cluster
        '''

        regions = dict()

        counter = Counter(self.demand.labels)

        for key,polygon in tqdm(self.demand.polygons.items()):
            
            region = DeliveryRegion()
            if region.make_from_polygon(self.G, polygon):
                region.register_id(key)
                region.set_using_heuristics(heuristics)
                region.register_density(counter[key])
                regions[key] = region
    
        self.regions = regions
        
        # prepare a node-region mapping for convinient reference
        node_affil = dict()
        for region_id, region in self.regions.items():
            for node in region.boundary:
                node_affil[node] = region_id
                
        self.boundary_node_affil = node_affil

        return
            

    def eval_intra_dr_cost(self, method = 'TSP'):
        ''' evaluate covering path cost for all start-end pairs
        '''
        
        intra_dr_cost = dict()
        intra_dr_path = dict()
        for region_id, region in tqdm(self.regions.items()):
            cost,paths = region.eval_cost(method)
            intra_dr_cost[region_id] = cost
            intra_dr_path[region_id] = paths
        self.intra_dr_cost = intra_dr_cost
        self.intra_dr_path = intra_dr_path
        return
            

    def eval_inter_dr_cost(self):
        
        inter_dr_cost = dict()
        
        
        for c1,c2 in tqdm(list(combinations(self.regions.keys(),r=2))):
            dr1,dr2 = self.regions[c1],self.regions[c2]
            cost = pd.DataFrame(np.inf,
                                index = dr1.boundary,
                                columns = dr2.boundary)
            for p1,p2 in product(dr1.boundary,dr2.boundary):
                cost.loc[p1,p2] = nx.shortest_path_length(self.G,p1,p2,weight = 'length')
                
        
            inter_dr_cost[(c1,c2)] = cost
            inter_dr_cost[(c2,c1)] = cost.transpose()
            
        self.inter_dr_cost = inter_dr_cost
        
        return
    
    def eval_depot_dr_cost(self):
        depot = self.depot
        
        depot_dr_cost = dict()
        for key,region in tqdm(self.regions.items()):
            cost = pd.DataFrame(data = np.inf,
                             index = region.boundary,
                             columns = [depot])
            for b in cost.index:
                cost.loc[b,depot] = nx.shortest_path_length(self.G,depot,b,weight = 'length')
            
            depot_dr_cost[key] = cost
            
        self.depot_dr_cost = depot_dr_cost
        
        return
        
    def register_params(self,**kwargs):
        for name,val in kwargs.items():
            self.__setattr__(name, val)
            
        return
    
    def register_depot(self,depot_coord):
        depot,_ = nearest_nodes(self.G, x = depot_coord[0], y = depot_coord[1])
        self.depot = depot