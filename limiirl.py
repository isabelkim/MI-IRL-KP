import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

# Feature representation of demonstrations (X)


# Number of clusters or behavior modes (K)


# Warm-start: Cluster the demonstrations using K-Means to initialize EM.
kmeans = KMeans(n_clusters=K)
cluster_assignments = kmeans.fit_predict(X)

# EM (Expectation Maximization)
# Initialize reward function parameters.
theta = np.random.rand(K, num_features)

# Define the EM loop.
max_iterations = 100
for iteration in range(max_iterations):
    # E-step: Assign demonstrations to clusters.
    cluster_assignments, _ = pairwise_distances_argmin_min(X, theta)
    
    # M-step: Update the reward function parameters (theta).
    for k in range(K):
        mask = (cluster_assignments == k)
        if np.sum(mask) > 0:
            theta[k] = np.mean(X[mask], axis=0)

# After EM converges, theta should contain reward functions for each cluster.