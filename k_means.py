import numpy as np 
import pandas as pd 
import random

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, K=None, metric='distortion'):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.K = K
        self.centroids = None
        self.mu = None
        self.metric = metric
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # TODO: Implement
        test, N = 10, 10
        
        # list scores will collect relevent information for each different initialization
        self.scores = []
        
        # loop on differents initialization 
        for _ in range(test):
            self.centroids = np.zeros((len(X)), dtype=int)
            self.mu = X.iloc[[random.randint(0,len(X)-1) for _ in range(self.K)]].to_numpy()
           
            # Repeat until convergence
            for _ in range(N):

                # setting on which cluster each points belong
                for i in range(len(X)):
                    self.centroids[i] = np.argmin([(euclidean_distance(X.iloc[i], self.mu[k])) for k in range(self.K)])

                # update the center of each centroids
                for j in range(self.K):
                    if len(X[self.centroids == j]) != 0:
                        self.mu[j] = X[self.centroids == j].mean(axis=0)
                    else:
                        pass
            
            # collect information for this simulation
            self.scores.append([euclidean_distortion(X,self.centroids),
                                euclidean_silhouette(X,self.centroids),
                                self.centroids,
                                self.mu])
        
        self.scores = np.array(self.scores, dtype=object)
        
        # best simulation depends on the metric
        if self.metric == 'silhouette':
            # for silhouette : higher is the best 
            best_ = list(self.scores[:,1]).index(max(self.scores[:,1]))
        else:
            # for distortion : lower is the best
            best_ = list(self.scores[:,0]).index(min(self.scores[:,0]))
        self.centroids = self.scores[best_, 2]
        self.mu = self.scores[best_, 3]
        
        
        
        
        
        
    def fit_noK(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # TODO: Implement
        test, N, K_max = 2, 5, 10
  
        # list scores will collect relevent information for each different initialization
        self.scores = []
        self.K_scores = []
        
        for self.K in range(2, K_max+1):
            
            # loop on differents initialization 
            for _ in range(test):
                self.centroids = np.zeros((len(X)), dtype=int)
                self.mu = X.iloc[[random.randint(0,len(X)-1) for _ in range(self.K)]].to_numpy()

                # Repeat until convergence
                for _ in range(N):

                    # setting on which cluster each points belong
                    for i in range(len(X)):
                        self.centroids[i] = np.argmin([(euclidean_distance(X.iloc[i], self.mu[k])) for k in range(self.K)])

                    # update the center of each centroids
                    for j in range(self.K):
                        if len(X[self.centroids == j]) != 0:
                            self.mu[j] = X[self.centroids == j].mean(axis=0)
                        else:
                            pass

                # collect information for this simulation
                self.scores.append([euclidean_distortion(X,self.centroids),
                                    euclidean_silhouette(X,self.centroids),
                                    self.centroids,
                                    self.mu, self.K])
                
            self.scores = np.array(self.scores, dtype=object)
            
            # retrieve the index of the best initialization for this K
            min_K = max((self.scores[self.scores[:,-1] == self.K])[:,1])
            i_loc = list((self.scores)[:,1]).index(min_K)
            self.scores = list(self.scores)
            self.K_scores.append(self.scores[i_loc])
        self.scores = np.array(self.scores, dtype=object)
        
        # True attribute
        if self.metric == 'silhouette':
            # for silhouette : higher is the best 
            best_ = list(self.scores[:,1]).index(max(self.scores[:,1]))
        else:
            # for distortion : lower is the best
            best_ = list(self.scores[:,0]).index(min(self.scores[:,0]))
            
        self.centroids = self.scores[best_, 2]
        self.mu = self.scores[best_, 3]
        self.centroids = self.scores[best_, 2]
        self.mu = self.scores[best_, 3]
        self.K = self.scores[best_, 4]
        
        
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # TODO: Implement 
        
        return self.centroids
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """        
        return self.mu
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum(axis=1).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  