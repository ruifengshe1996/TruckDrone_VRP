import networkx as nx
from utils.network import eval_path_cost

def solve(nodes:list,dist:dict) -> tuple:

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(dist)
    nx.set_edge_attributes(G,
                           values = {edge:{'length':length} for edge,length in dist.items()},
                           )

    path = nx.approximation.traveling_salesman_problem(G,weight='length')
    val = eval_path_cost(G,path)

    return val,path