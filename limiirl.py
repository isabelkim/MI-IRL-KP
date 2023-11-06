import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from maxent import feature_expectation_from_trajectories

"""
Monica Babe ̧s-Vroman, Vukosi Marivate, Kaushik Subramanian, and Michael L. Littman.
Apprenticeship Learning About Multiple Intentions. In Proceedings of the 28th International
Conference on Machine Learning, ICML ’11, pages 897–904, Bellevue, WA, USA, 2011. ACM,
New York, NY, USA. ISBN 978-1-4503-0619-5.
"""



def limiirl(taus, K=1000):
    # Feature representation of demonstrations (X) -> figure this out


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
        rho[k] = np.sum([u[i][k] for i in range(n)], axis=0) / n


    # theta = np.random.rand(K, num_features)

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