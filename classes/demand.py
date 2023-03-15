import numpy as np


# =================== base class of demand ====================
class Demand(object):
    def __init__(self,data:np.ndarray, labels: np.ndarray = None) -> None:
        self.data = data
        self.labels = labels
        
        self.clustered = True if labels is not None else False
        
        return

    
    def reset_cluster(self):
        
        self.labels = None
        self.clustered = False
        
        return
    
    def cluster(self, method = 'HDCA',**kwargs):
        
        if self.clustered:
            print('demand is already clustered. If want to re-cluster, reset first')
            return
        
        from importlib import import_module
        cluster_mod = import_module('.'+method,'cluster_alg')
        cluster_class = getattr(cluster_mod,method)

        cluster = cluster_class(self.demand.data)
        cluster.set_params(**kwargs)
        cluster.process()
        
        self.labels = cluster.labels

        return

    def cluster2shape(self):
        ''' covnerts labeld clusters into shapes
        1. scipy.spatial.ConvexHull
        2. shapely.geometry.polygon.Polygon
        '''
        from scipy.spatial import ConvexHull
        from collections import defaultdict
        from shapely.geometry.polygon import Polygon

        clusters = defaultdict(list)
        
        data = self.data
        labels = self.labels

        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(data[i])

        chulls = dict()
        polygons = dict()
        for label,cluster in clusters.items():
            if len(cluster) >= 3:
                cluster = np.array(cluster)
                
                c = ConvexHull(cluster)
                p = Polygon(cluster[c.vertices])

                if not p.is_valid:
                    print('Warning: cluster {} does not generate a valid shapely polygon'.format(label))

                chulls[label] = c
                polygons[label] = p
        self.chulls = chulls
        self.polygons = polygons
        
        return

    def validate_clusters(self, prune = False) -> bool:


        from itertools import product
        
        violation = set()
        for key1,key2 in product(self.polygons.keys(),repeat = 2):
            if key1 != key2:
                p1 = self.polygons[key1]
                p2 = self.polygons[key2]
                if not ((key1,key2) in violation or (key2,key1) in violation):
                    if p1.intersects(p2): 
                        violation.add((key1,key2))
                        

        if prune:
            for key1,key2 in violation:
                if key1 in self.polygons:
                    self.polygons.pop(key1)
                    self.chulls.pop(key1)
                if key2 in self.polygons:
                    self.polygons.pop(key2)
                    self.chulls.pop(key2)

        if not violation:
            print('clusters validated as mutually disjoint')
            
            return True
        else:
            print('the following cluster pairs intersect: ', violation)
            if prune:
                print('intersecting clusters pruned, numbers of clusters left: ', len(self.polygons.keys()))
            return False




# =========== demand generator functions ======================================

def random_sub_bbox(bbox):
    ''' generate a random bbox within the given bbox
    '''
    north, south, east, west = bbox

    x1 = np.random.rand()*(east-west) + west
    x2 = np.random.rand()*(east-west) + west
    y1 = np.random.rand()*(north-south) + south
    y2 = np.random.rand()*(north-south) + south

    return ([max([y1,y2]),min([y1,y2]),max([x1,x2]),min([x1,x2])])

def generate_bbox(N:int, bbox:list = None, mode = 'haltons'):
    
    if bbox is None:
        bbox = [1,0,1,0]
    
    north,south,east,west = bbox
    width = east - west
    height = north - south
    anchor = (west,south)

    if mode == 'uniform':
        X = np.random.rand(N,2)
    elif mode == 'haltons':
        from scipy.stats.qmc import Halton
        sampler = Halton(d=2,scramble = False)
        X = sampler.random(n=N)
    elif mode == 'grid':
        x,y = np.meshgrid(np.linspace(0,1,int(np.sqrt(N))),np.linspace(0,1,int(np.sqrt(N))))
        X = np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis = 1)

    # project to proper bbox
    X[:,0] = width * X[:,0]  + anchor[0]
    X[:,1] = height * X[:,1] + anchor[1]

    return X



# Junkyard


# # ========================== demand set calss ===================
# class DemandSet(object):
#     ''' demand points input/generation and clustering
#     '''

#     def __init__(self, **kwargs) -> None:

#         self.clustered = False
#         self.demand_list = []

#         self.demand = Demand(None,None)
#         self.labels = None
#         return

#     def register_demand(self,demand:Demand,merge = False):

#         self.demand_list.append(demand)
#         if merge: self._merge_demand()

#         return


#     def generate_demand_bbox(self,number_of_demand:int,bbox:list):
#         generator = DemandGenerator(bbox)
#         demand_points = generator.generate_unif(number_of_demand)

#         self.demand_list.append(Demand(demand_points,bbox))

#         return
    


            
#     def _merge_demand(self):
#         ''' merge all demand classes in to one 
#         '''
#         def merge_bbox(bbox_list):

#             north = np.max([bbox[0] for bbox in bbox_list])
#             south = np.min([bbox[1] for bbox in bbox_list])
#             east = np.max([bbox[2] for bbox in bbox_list])
#             west = np.min([bbox[3] for bbox in bbox_list])

#             return [north,south,east,west]

#         demand = np.concatenate([d.data for d in self.demand_list])
#         bbox = merge_bbox([d.bbox for d in self.demand_list])

#         self.demand = Demand(demand,bbox)

#         return







