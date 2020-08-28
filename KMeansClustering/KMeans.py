import random
import numpy as np
import scipy
from scipy.stats.mstats import gmean
import sys

class KMeans:
    def __init__(self, n_clusters, n_iter=10, random_state=None):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, data):
        """
        input, 2D numpy array
        output, 1
        """
        if self.random_state is not None:
            random.seed(self.random_state)

        self.centroids = [tuple(random.choice(data)) for i in range(self.n_clusters)]
        dist = {}
        for centroid in self.centroids:
            distances = [np.linalg.norm(value - centroid) for value in data]
            dist[centroid] = distances
        
        self.clusters = []
        for i in range(len(data)):
            comparison = []
            for j in range(len(self.centroids)):
                comparison.append(dist[tuple(self.centroids)[j]][i])

            cluster = comparison.index(min(comparison))
            self.clusters.append(cluster)

        self.clusters = np.array(self.clusters)

        self.avgs = []
        self.gmeans = []
        self.cluster_dict = {}
        for cluster in set(np.array(self.clusters)):
            indicies = np.where(self.clusters == cluster)
            dist_list = [list(dist.values())[cluster][i] for i in indicies[0] if i in indicies[0]]
            self.avgs.append(sum(dist_list) / len(dist_list))

            cluster_list = np.array([data[i] for i in indicies[0] if i in indicies[0]])
            self.gmeans.append(gmean(cluster_list))

            self.cluster_dict[cluster] = cluster_list

        # Though it would have been preferable to use recursion for
        # conciseness, the need to initialize random values at the
        # top of the method necessitated using a while loop.
        while self.n_iter >= 1:
            self.n_iter -= 1

            self.centroids = [tuple(gmean) for gmean in self.gmeans]
            dist = {}
            for centroid in self.centroids:
                distances = [np.linalg.norm(value - centroid) for value in data]
                dist[centroid] = distances
                
            self.clusters = []
            for i in range(len(data)):
                comparison = []
                for j in range(len(self.centroids)):
                    comparison.append(dist[tuple(self.centroids)[j]][i])

                cluster = comparison.index(min(comparison))
                self.clusters.append(cluster)
                    
            self.avgs = []
            self.gmeans = []
            self.cluster_dict = {}
            for cluster in set(np.array(self.clusters)):
                indicies = np.where(self.clusters == cluster)
                dist_list = [list(dist.values())[cluster][i] for i in indicies[0] if i in indicies[0]]
                self.avgs.append(sum(dist_list) / len(dist_list))
                cluster_list = np.array([data[i] for i in indicies[0] if i in indicies[0]])
                self.gmeans.append(gmean(cluster_list))
                self.cluster_dict[cluster] = cluster_list

            self.clusters = np.array(self.clusters)

    def predict(self, data):
        distances = [np.linalg.norm(data - centroid) for centroid in self.centroids]
        cluster = distances.index(min(distances))
        return cluster