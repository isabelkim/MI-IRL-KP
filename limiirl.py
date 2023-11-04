import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

def limiirl(taus, K=1000):
    # Feature representation of demonstrations (X)


    # Number of clusters or behavior modes (K)

    # Warm-start: Cluster the demonstrations using K-Means to initialize EM.
    kmeans = KMeans(n_clusters=K)
    cluster_model = kmeans.fit(taus)

    # EM (Expectation Maximization)
    # Initialize reward function parameters.
    n = len(taus) 
    u = np.zeros((n, K))

    for i in range(n):
        for j in range(K):
            c = cluster_model.predict(taus[i])
            u[i][j] = 1 if c == j else 0
                
    rho = np.zeros(K)
    for k in range(K):
        sum = 0
        for i in range(n):
            sum += u[i][k]
        rho[k] = sum / n


    theta = np.random.rand(K, num_features)

    # Define the EM loop.
    max_iterations = 100
    for iteration in range(max_iterations):
        # E-step: Assign demonstrations to clusters.
        cluster_assignments, _ = pairwise_distances_argmin_min(taus, theta)
        
        # M-step: Update the reward function parameters (theta).
        for k in range(K):
            mask = (cluster_assignments == k)
            if np.sum(mask) > 0:
                theta[k] = np.mean(taus[mask], axis=0)

    # After EM converges, theta should contain reward functions for each cluster.