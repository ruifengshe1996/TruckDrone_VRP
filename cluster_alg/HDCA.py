import numpy as np
from tqdm import tqdm

class HDCA():
    def __init__(self, dataset:np.ndarray):

        self.dataset = dataset
        self.N = len(dataset)
        self.reset()


    def process(self):
        ''' convenient call to a cluster task
        '''
        self.reset()
        self.compute_distance()
        self.cluster()



    def set_params(self,**kwargs):
        for key,val in kwargs.items():
            self.__setattr__(key,val)

        return


    def cluster(self):


        print('start clustering with maxpts = {}, minpts = {}'.format(self.maxpts,self.minpts))

        clus_ind = 0
        
        for i in tqdm(range(self.N)):
            if not self.labels[i]:
                clustered = self.expand_cluster(i,clus_ind)
                if clustered: clus_ind += 1

        
        print('identified {} clusters'.format(clus_ind))
        print('remaining {} unclassified points'.format(self.labels.count(None)))
        
        return

    def reset(self):
        self.labels = [None for _ in range(self.N)]
        return

    def expand_cluster(self,init_ind:int,clus_ind:int) -> bool:

        

        _,eps = self.get_k_neighbors(init_ind,self.maxpts)
        seed = self.get_neighbors(init_ind,eps)
        
        if len(seed)<self.minpts-1:
            self.labels[init_ind] = -1
            return False

        # assign clusters
        self.labels[init_ind] = clus_ind
        for i in seed: self.labels[i] = clus_ind

        # add new neighbors (BFS)
        while seed:

            new_init_ind = seed.pop()
            neighbors = self.get_neighbors(new_init_ind,eps)
            new_seed = [i for i in neighbors if self.labels[i] is None]
            if len(new_seed)>=self.minpts-1:
                for i in new_seed: 
                    self.labels[i] = clus_ind
                    seed.append(i)
        
        return True



    

    def get_neighbors(self,ind,eps) -> list[int]:
        sorted_dist = self.dist_mat[ind,self.sorted_ind[ind]]
        
        for k in range(self.N):
            if sorted_dist[k]>eps:
                break
        
        return self.sorted_ind[ind][:k]



    def get_k_neighbors(self,ind,k) -> tuple[list[int],float]:
        knearest_ind = self.sorted_ind[ind][:k]
        knearest_dist = np.mean(self.dist_mat[ind,knearest_ind])
        return knearest_ind,knearest_dist

    def compute_distance(self) -> None:

        print('computing point-wise distance')

        from scipy.spatial.distance import pdist, squareform

        dist_mat = squareform(pdist(self.dataset,metric = 'euclidean'))
        
        sorted_ind = [list(np.argsort(dist_mat[i,:])) for i in range(self.N)]

        self.dist_mat = dist_mat
        self.sorted_ind = sorted_ind
        return