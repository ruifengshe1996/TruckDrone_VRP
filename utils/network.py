import networkx as nx
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
import osmnx as ox
import numpy as np

def nearest_nodes(G:nx.Graph, x, y):
    """ find the nearest nodes in G from the given coordinate

    Args:
        G (nx.Graph): the graph to find nodes in
        x (_type_): x coordinate(s), value or iterator
        y (_type_): y coordinate(s), value or iterator

    Returns:
        (nn,dist): if input is iterator, return the list of nodes and corresponding distance with the same length
                   if input is a single value, return the node and the distance
    """

    if hasattr(x,'__iter__'):
        num_points = len(x)
        nn = []
        dists = []
        for i in range(num_points):
            node,dist = nearest_nodes(G,x[i],y[i])
            nn.append(node)
            dists.append(dist)
        return nn, dists

    else:

        node_coord = np.array([(node['x'],node['y']) for node in G.nodes.values()])
        dist = np.linalg.norm(node_coord - np.array((x,y)),axis = 1)
        min_ind = np.argmin(dist);min_dist = dist[min_ind]
        
        return list(G.nodes)[min_ind], min_dist


def prepare_network(G:nx.MultiDiGraph) -> nx.Graph:
    ''' prepare a freshely downloaded nx.MultiDiGraph by callying ox.graph_from_...()
    1. convert into undirected graph nx.Graph
    2. remove self-loop edges
    '''
    # convert into nx.Graph
    G = ox.project_graph(G)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G

def prepare_grid_network(G:nx.Graph) -> nx.Graph:
    ''' preprocess the grid_2d_network 
    '''
    # assign node coordinates
    node_coord = {node:{'x':node[0],'y':node[1]} for node in G.nodes}
    nx.set_node_attributes(G,node_coord)
    # assign edge lengths
    edge_length = {edge:{'length':np.linalg.norm(np.array(edge[0])-np.array(edge[1]))} for edge in G.edges}
    nx.set_edge_attributes(G,edge_length)
    # rename nodes 
    node_rename = dict(zip(node_coord.keys(),range(len(node_coord))))
    nx.relabel_nodes(G,node_rename,copy = False)
    
    return G

def bbox_from_polygon(polygon:Polygon):
    west,south,east,north = polygon.bounds
    return [north,south,east,west]

def bbox_from_graph(G:nx.Graph):
    x = nx.get_node_attributes(G,'x').values()
    y = nx.get_node_attributes(G,'y').values()
    return [max(y),min(y),max(x),min(x)]


def complete_path(vrp,path:list) -> list:
    
    G = vrp.G
    c_path = [path[0]]
    for i in range(len(path)-1):
        cur_node, next_node = path[i], path[i+1]
        
        is_intra_region = False
        if cur_node in vrp.boundary_node_affil and next_node in vrp.boundary_node_affil:
            if vrp.boundary_node_affil[cur_node] == vrp.boundary_node_affil[next_node]:
                region_id = vrp.boundary_node_affil[cur_node]
                is_intra_region = True
        
        if is_intra_region:
            # intra-region
            # resolve to get a path
            cost,sub_path = vrp.regions[region_id].cpp.solve(cur_node,next_node)
            # sub_path = nx.shortest_path(G, cur_node, next_node, weight = 'length')
            
            
        else:
            # inter-region or region-depot        
            sub_path = nx.shortest_path(G, cur_node, next_node, weight = 'length')

        c_path += sub_path[1:]
        
    return c_path

def nodes_from_polygon(G:nx.Graph, polygon:Polygon) -> list:
    interior_nodes = [node_name for node_name,node in G.nodes.items()
                        if polygon.contains(Point((node['x'],node['y'])))]
    
    return interior_nodes

def get_boundary_nodes(G:nx.Graph,sub_G:nx.Graph) -> list:
    boundary_nodes = set()
    for node_name,node in sub_G.nodes.items():
        for neighbor in G.adj[node_name]:
            if neighbor not in sub_G.nodes:
                boundary_nodes.add(neighbor)
    return list(boundary_nodes)

def eval_path_cost(G:nx.Graph,
                   path:list,
                   autocomplete:bool = False)->float:
    ''' evaluates the cost of a path in graph G
    '''
    assert nx.is_weighted(G,weight = 'length'), 'graph has no edge length, must be set'

    path_cost = 0
    for i in range(len(path)-1):
        s,t = path[i],path[i+1]
        if (s,t) in G.edges:
            path_cost += G.edges[(s,t)]['length']
        else:
            if autocomplete:
                path_cost += nx.shortest_path_length(G,s,t,weight = 'length')
            else:
                raise KeyError('({},{}) not connected in graph'.format(s,t))

    return path_cost


def eval_path_cost_directed(G:nx.DiGraph,
                   path:list)->float:
    ''' evaluates the cost of a path in graph G
    '''
    assert nx.is_weighted(G,weight = 'length'), 'graph has no edge length, must be set'

    path_cost = 0
    for i in range(len(path)-1):
        s,t = path[i],path[i+1]
        if (s,t) in G.edges:
            path_cost += G.edges[(s,t)]['length']
        elif (t,s) in G.edges:
            print('??')
            path_cost += G.edges[(t,s)]['length']
        else:
            print('???')
            path_cost += nx.shortest_path_length(G,s,t,weight = 'length')

    return path_cost