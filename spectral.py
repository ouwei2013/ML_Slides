
import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.cluster import  KMeans
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

x=mpimg.imread('/Users/ouwei/Desktop/b.jpg')
print(x.shape)
x= sp.misc.imresize(x, 0.10) / 255
#plt.imshow(x, cmap=plt.cm.gray)
graph=image.img_to_graph(x)
beta = 5
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
N_REGIONS = 6
for assign_labels in ('kmeans', 'discretize'):
    t0 = time.time()
    labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_labels, random_state=1)
    t1 = time.time()
    labels = labels.reshape(x.shape)
    print(labels.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(x, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l, contours=1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS))])
    plt.xticks(())
    plt.yticks(())
    title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
    plt.title(title)

y=x.reshape(x.size,1)
t0 = time.time()
kmeans =KMeans(n_clusters=N_REGIONS, random_state=0).fit(y)
t1 = time.time()
z=kmeans.labels_.reshape(x.shape)
plt.figure(figsize=(5, 5))
plt.imshow(x, cmap=plt.cm.gray)
for l in range(N_REGIONS):
        plt.contour(z == l, contours=1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS))])
plt.xticks(())
plt.yticks(())
title = 'K-means clustering:%.2fs' %(t1 - t0)
plt.title(title)
plt.show()
