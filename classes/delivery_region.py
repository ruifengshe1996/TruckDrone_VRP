import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
import itertools
import copy


from shapely.geometry.polygon import Polygon
from itertools import combinations,product

from utils.network import bbox_from_graph, prepare_network
from classes.drone_traffic import yield_optimal_drone_config
from utils.network import nearest_nodes

class DeliveryRegion(object):
    ''' master class of delivevery region

    solves all region-level problems

    @attr interior_points
    @attr boundary_points
    @attr graph

    @method solve_open_TSP(start,end)
        solves the open-tsp problem with given start-end
        creates a dummy node and solve as standard tsp
        use nx.christofide

        @return solution: list[node]
        @return cost: float
    
    @method solve_swath(start,end)
        solves the open-tsp problem with back-forth heuristics

        @return solution: list[node]
        @return cost: float
        

    '''

    def __init__(self, 
                 polygon:Polygon = None, 
                 density:float = 1
                 ) -> None:
        

        self.polygon = polygon
        self.density = density

        self.heuristics = True

        self.G : nx.Graph
        self.boundary : list
        self.interior : list

        if polygon is not None:
            self.make_from_polygon(polygon)
        return

    def set_using_heuristics(self,heuristics:bool):
        self.heuristics = heuristics
        return

    def register_id(self,id):
        self.id = id
        return

    def register_density(self,num_demand):
        self.num_demand = num_demand
        self.density = num_demand / self.polygon.area
        return



    def make_from_polygon(self,G:nx.Graph, polygon:Polygon) -> bool:
        ''' extract the subgraph, boundary and interior points from given polygon
        '''
        
        self.polygon = polygon
        from utils.network import nodes_from_polygon,get_boundary_nodes
        interior = nodes_from_polygon(G,polygon)

        if not interior: # trivial region, skip
            return False

        # if interior subgraph is not connected, yield the largest connected component instead
        G_interior = nx.induced_subgraph(G,interior)
        if nx.is_connected(G_interior):
            self.G_interior = G_interior.copy()
        else:
            largest_component_nodes = sorted(nx.connected_components(G_interior), key=len, reverse=True)[0]
            self.G_interior = nx.induced_subgraph(G,largest_component_nodes).copy()


        self.interior = list(self.G_interior.nodes)

        self.boundary = get_boundary_nodes(G,self.G_interior)
        self.G = nx.induced_subgraph(G,self.boundary + self.interior).copy()

        # if the subgraph is not connected, skip it as there is no way to deal with it
        if not nx.is_connected(self.G):
            print('\n something is wrong with this polygon with centroid {}, skip'.format(polygon.centroid))
            return False


        return True

    def _get_config(self, modify = False):
        
        
        # get service width from the surrogate CA model
        config = yield_optimal_drone_config(self.polygon.area, self.density)
        
        if modify:
            W = config['W']
            vertices = np.array(self.polygon.boundary.coords)
            _,dist_list = nearest_nodes(self.G_interior,x = vertices[:,0],y = vertices[:,1])
            W = np.min([W,np.min(dist_list)])
            config['W'] = W

            # TODO: regain vt based on W

        self.config = config
        return


    def eval_cost(self, method = 'TSP'):
        ''' evaluate covering path cost for all start-end pairs
        '''

        self._get_config()

        if method == 'TSP':
            from modules.covering_path_problem.open_tsp import OpenTSP
            cpp = OpenTSP(G = self.G,
                          G_interior = self.G_interior,
                          boundaries = self.boundary,
                          swath_width = self.config['W'],
                          heuristics = self.heuristics)
        else:
            from modules.covering_path_problem.swath import Swath
            cpp = Swath(swath_width = self.config['W'])

        self.cpp = cpp
        
        cost = pd.DataFrame(np.inf, 
                            index = self.boundary,
                            columns = self.boundary)
        paths = dict()

        for start,end in itertools.product(cost.index,cost.columns):
            if (start,end) in paths or (end,start) in paths: continue

            val,path = cpp.solve_simple(start,end)

            if path is not None:

                cost.loc[start,end] = val
                cost.loc[end,start] = val
                
                paths[(start,end)] = path



        return cost, paths












    # def get_path(self,start,end):
    #     try:
    #         return self.paths[(start,end)]
    #     except AttributeError:
    #         raise AttributeError('no paths yet stored, run eval_cost first')
    #     except KeyError:
    #         try:
    #             return self.paths[(end,start)]
    #         except KeyError:
    #             raise KeyError('the requested o-d pair does not exist. Check again')



        #  def make_from_polygon(self,polygon:Polygon,crs) -> bool:
        # ''' extract the subgraph, boundary and interior points from given polygon
        # '''
        # if crs:
        #     # crs is provided, try to retrieve graph from openstreetmap
        #     try: 
        #         # convert to long lat
        #         polygon_latlong,_ = ox.projection.project_geometry(polygon,crs,to_crs = 'epsg:4326')
                
        #         G_dr_enclosure = ox.graph_from_polygon(polygon_latlong,
        #                                             network_type='drive',
        #                                             simplify=True,
        #                                             truncate_by_edge=True)

        #         G_dr_interior = ox.graph_from_polygon(polygon_latlong,
        #                                             network_type='drive',
        #                                             simplify=True,
        #                                             truncate_by_edge=False)
        #     except:
        #         print('this region is somehow not valid, skipped')
        #         return False
        # else:
        #     # crs is not provided, make subgraph from polygon directly
        #     polygon = None

        # self.polygon = polygon
        

        # nodeset_enclosure = set(G_dr_enclosure.nodes)
        # nodeset_interior = set(G_dr_interior.nodes)
        # nodeset_boundary = nodeset_enclosure.difference(nodeset_interior)

        # self.G = prepare_network(G_dr_enclosure)
        # self.G_interior = prepare_network(G_dr_interior)
        # self.boundary = list(nodeset_boundary)
        # self.interior = list(nodeset_interior)

        # return True       





    # def _solve_open_TSP(self,start,end) -> tuple:

    #     G = copy.deepcopy(self.G)

    #     # add a dummy node
    #     G.add_node('dummy',x=0,y=0)
        
    #     G.add_edge('dummy',start, length = 1e-6)
    #     G.add_edge(end,'dummy', length = 1e-6)

    #     visited_nodes = self.visited_nodes + [start,end,'dummy']
    #     path = nx.approximation.traveling_salesman_problem(G, weight = 'length',nodes = visited_nodes)
        
    #     if path.count('dummy') > 1: 
    #         print('dummy node visited multiple times, something is wrong')
    #         return np.inf, None
        
    #     # remove dummy node and pivot path to start with start and end with end
    #     d_ind = path.index('dummy')
    #     path = path[d_ind:] + path[:d_ind]
    #     path.remove('dummy')

    #     if path[0] == start and path[-1] == end:
    #         pass
    #     elif path[0] == start and path[-1] == end:
    #         path = path[::-1]
    #     else:
    #         print('path not starting and ending properly, something is wrong')
    #         return np.inf, None

    #     val = eval_path_cost(self.G, path)

    #     return val,path

    