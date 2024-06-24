#imports
import numpy as np


def find_closest_centroids(X, centroids):

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
          dist = [] 
          for j in range(centroids.shape[0]):
              ij = np.linalg.norm(X[i] - centroids[j])
              dist.append(ij)

          idx[i] = np.argmin(dist)       
                      
    
    return idx

# Load data stored in arrays X, y from data folder (ex7data2.mat)
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
data = sio.loadmat('data/ex7data2.mat')
X = data['X']

print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape)

# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])