import networkx as nx
from numbers import Real
import itertools
import copy
import numpy as np
from utils.network import bbox_from_graph, nearest_nodes
        
from modules.covering_path_problem import open_tsp_gurobi,open_tsp_networkx

class OpenTSP(object):
    ''' the open tsp problem
    '''

    def __init__(self,
                 G:nx.Graph,
                 G_interior:nx.Graph,
                 boundaries:list,
                 swath_width:float,
                 heuristics:bool = True,
                 simple:bool = True) -> None:
        '''
        @param G: the master graph
        @param boundaries: boundary nodes
        @swath_width :
        @heuristics: whether to use heuristics or gurobi
        '''
        self.G = G
        self.G_interior = G_interior
        self.boundaries = boundaries
        self.swath_width = swath_width
        self.heuristics = heuristics
        self.simple = simple
        # identify center
        self.center = nx.center(self.G,weight = 'length')[0]
        
        # identify visited nodes and pre-compute spd
        self.visited_nodes = self._get_visited_nodes()
        self.inner_dist = {(c1, c2): nx.shortest_path_length(self.G,c1,c2,weight = 'length') \
                for c1, c2 in itertools.combinations(self.visited_nodes, 2)}
        pass

    



    def _get_visited_nodes(self) -> list:
        # ==================== identify node subset to visit ======
        north,south,east,west = bbox_from_graph(self.G)

        x = np.arange(west,east,step = self.swath_width)
        y = np.arange(south,north,step = self.swath_width)
        grid_nodes = np.stack(np.meshgrid(x,y),axis = 2).reshape(-1,2)


        nn,_ = nearest_nodes(self.G_interior,x= grid_nodes[:,0],y = grid_nodes[:,1])
        visited_nodes = list(set(nn))
        
        return visited_nodes


    def _pre_process(self,start,end) -> tuple[list,dict]:
        """yields the nodes and edges of the abstract graph
        only used to solve the open-tsp as a augmented tsp
        

        Args:
            start (_type_): _description_
            end (_type_): _description_

        Returns:
            tuple[list,dict]: _description_
            nodes: visited nodes plus start, end plus dummy, if needed
            dist: shortest path distance between each pair of nodes
        """

        dist = copy.deepcopy(self.inner_dist)
        diff_access_flag = start != end        


        # make it an open-tsp
        if diff_access_flag:
            # different start-end case
            for c1,c2 in itertools.product(self.visited_nodes,[start,end]):
                dist[(c1,c2)] = nx.shortest_path_length(self.G,c1,c2,weight = 'length')
                
            for c1,c2 in itertools.product(self.visited_nodes,['dummy']):
                dist[(c1,c2)] = 1e6
            dist[(start,end)] = 1e6
            nodes = self.visited_nodes + [start,end,'dummy']
            dist[(start,'dummy')] = 0.1
            dist[(end,'dummy')] = 0.1
        else:
            # same start-end
            for c1,c2 in itertools.product(self.visited_nodes,[start]):
                dist[(c1,c2)] = nx.shortest_path_length(self.G,c1,c2,weight = 'length')
            nodes = self.visited_nodes + [start]

        return nodes, dist
    
    def _post_process(self,path:list,start,end) -> list:
        """ format the solution to the open-tsp to a valid path
        removes the dummy node if necessary
        properly include start and end
        
        only used to solve the open-tsp as a augmented tsp

        Args:
            path (list): raw solution
            start (_type_): node label
            end (_type_): node label

        Returns:
            list: factored path
        """
        
        # format the path from the open tsp solution
        diff_access_flag = start != end     
        # remove dummy node and pivot path to start with start and end with end
        if diff_access_flag:
            d_ind = path.index('dummy')
            path = path[d_ind:] + path[:d_ind]
            path.pop(0)
    
            if path[0] == start and path[-1] == end:
                pass
            elif path[0] == end and path[-1] == start:
                path = path[::-1]
            else:
                # print('path not starting and ending properly, something is wrong')
                return np.inf, None
        else:
            s_ind = path.index(start)
            path = path[s_ind:] + path[:s_ind]
            path.append(start)
        # print(path)
        return path

    def solve(self,start,end) -> tuple:
        ''' solve the open tsp for the given start and end
        '''
        
        nodes,dist = self._pre_process(start,end)

        # special cases:
        if len(self.visited_nodes) == 0:
            return 0, (start,end)
        elif len(self.visited_nodes) == 1:
            node = self.visited_nodes[0]
            return dist[(node,start)] + dist[(node,end)], [start,node,end]
        # call gurobi function to solve
        if self.heuristics:
            val,path = open_tsp_networkx.solve(nodes,dist)
        else:
            val,path = open_tsp_gurobi.solve(nodes,dist)
            
        if not path:
            # print('start: {}, end: {} failed, skipping this'.format(start,end))
            return np.inf, None
        
        path = self._post_process(path,start,end)
        
        return val,path
    

    def solve_simple(self,start,end) -> Real:
        """ yield the simple start-center-end cost

        Args:
            start (_type_): _description_
            end (_type_): _description_

        Returns:
            Real: _description_
        """
        
        val = nx.shortest_path_length(self.G, start,self.center,weight = 'length') + \
                nx.shortest_path_length(self.G, self.center,end,weight = 'length')
        path = [start,self.center,end]
        return val, path