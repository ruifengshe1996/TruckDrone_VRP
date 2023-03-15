import numpy as np
import networkx as nx
import itertools

class Swath(object):
    
    def __init__(self,
                 G:nx.Graph,
                 G_interior:nx.Graph,
                 boundaries:list,
                 swath_width:float,) -> None:
        
        self.G = G
        self.G_interior = G_interior
        self.boundaries = boundaries
        self.swath_width = swath_width
        
        self._get_visited_nodes()
        self.inner_dist = {(c1, c2): nx.shortest_path_length(self.G,c1,c2,weight = 'length') \
                for c1, c2 in itertools.combinations(self.visited_nodes, 2)}
        
        return

    def _get_visited_nodes(self):
        
        # ==================== identify node subset to visit =======================
        
        from utils.network import bbox_from_graph, nearest_nodes
        north,south,east,west = bbox_from_graph(self.G)

        x = np.arange(west,east,step = self.swath_width)
        y = np.arange(south,north,step = self.swath_width)

        grid_nodes = np.stack(np.meshgrid(x,y),axis = 2).reshape(-1,2)

        nn,dist = nearest_nodes(self.G_interior,x= grid_nodes[:,0],y = grid_nodes[:,1])
        mask = [n if d>np.sqrt(2)/2*self.swath_width else None for (n,d) in zip(nn,dist) ]
        
        self.board = np.array(mask).reshape(grid_nodes.shape[:2])
        self.visited_nodes = list(set(nn))
        
        return
    
    def reset(self):
        self.sol = dict()
        
    def solve(self, 
              board:np.ndarray,
              start:int,
              end:int) -> tuple:
        
        case_num = self.map_case(start,end)    
        # case-by-case
        # choose D1 and D2 to be 0 (row) or 1 (col)
        
        if case_num in [1,7]:
            # odd num_row, odd num_col, diag
            # standard: (1,4)
            # choose D1 as row, D2 as column, the cost will be the same
            D1 = 0; D2 = 1
            path = []
            for i in range(self.board.shape[D1]):
                pass
                
        
        return
                
            
            
            
    def map_case(self,start,end):

        """solves the swath with given start and end

        position encoding: top-left: 1, top-right: 2, bottom-left: 3, bottom-right: 4
        1   2
        3   4

        Args:
            start (int): starting position
            end (int): ending position

        Returns:
            (list: path as list of coordinates, float: path cost)
        
        Choose D1 as the main axis to scan, and D2 as the minor axis
        
        A total of 7 cases may call for different solution
        1. odd num_row, odd num_col, diag: 
        2. odd num_row, odd num_col, same-side:
        
        3. odd num_row, even num_col, diag:
        4. odd num_row, even num_col, same-side on row (odd):
        5. odd num_row, even num_col, same-side on col (even):
        
        6. even num_row, even num_col, diag:
        7. even num_row, even num_col, same-side:
        
        """
        num_row,num_col = self.board.shape
        
        # differentiate case
        if self.isodd(num_row) and self.isodd(num_col):
            if (start,end) in [(1,4),(4,1),(2,3),(3,2)]:
                case_num = 1
            else:
                case_num = 2
        
        if self.isodd(num_row) and not self.isodd(num_col):
            if (start,end) in [(1,4),(4,1),(2,3),(3,2)]:
                case_num = 3
            elif (start,end) in [(1,3),(3,1),(2,4),(4,2)]:
                case_num = 4
            else:
                case_num = 5
        
        if not self.isodd(num_row) and self.isodd(num_col):
            if (start,end) in [(1,4),(4,1),(2,3),(3,2)]:
                case_num = 3
            elif (start,end) in [(1,3),(3,1),(2,4),(4,2)]:
                case_num = 5
            else:
                case_num = 4
                
        if not self.isodd(num_row) and not self.isodd(num_col):
            if (start,end) in [(1,4),(4,1),(2,3),(3,2)]:
                case_num = 6
            else:
                case_num = 7
                
        return case_num
    
    @staticmethod
    def isodd(a:int):
        return (a%2==1)
