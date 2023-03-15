from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from demand_clustering.ddbscan import HDCA
import numpy as np
import scipy

# x,y = make_blobs(n_samples=300, 
#             n_features=2, 
#             centers=5, 
#             cluster_std=0.2, 
#             center_box=(-10.0, 10.0), 
#             shuffle=True, 
#             random_state=2, 
#             return_centers=False)


def gen_unif(N,width,height,mode = 'haltons'):

    if mode == 'uniform':
        X = np.random.rand(N,2)
    elif mode == 'haltons':
        sampler = scipy.stats.qmc.Halton(d=2,scramble = False)
        X = sampler.random(n=N)
    elif mode == 'grid':
        x,y = np.meshgrid(np.linspace(0,1,int(np.sqrt(N))),np.linspace(0,1,int(np.sqrt(N))))
        X = np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis = 1)

    X[:,0] *= width
    X[:,1] *= height
    return X

x = gen_unif(100,1.1,1.1)
x_ = gen_unif(100,2.1,2.1); x = np.concatenate([x,x_])
x_ = gen_unif(100,3.1,3.1); x = np.concatenate([x,x_])

hdca = HDCA(x,2,4)
hdca.compute_distance()
hdca.cluster()
labels = hdca.labels

# clustering = DBSCAN(eps=0.1, min_samples=3).fit(x)
# labels = clustering.labels_


print(labels)

clustered_labels = [i for i in range(len(labels)) if labels[i] != -1]
clustered_color = [labels[i] for i in range(len(labels)) if labels[i] != -1]
unclustered_labels = [i for i in range(len(labels)) if labels[i] == -1]


fig,ax = plt.subplots()
ax.set_aspect('equal')
ax.scatter(x[clustered_labels,0],x[clustered_labels,1],
           c = clustered_color,
           marker='o')
ax.scatter(x[unclustered_labels,0],x[unclustered_labels,1],
           c = 'black',
           marker='x')
plt.show()

