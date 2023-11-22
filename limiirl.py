import numpy as np
from sklearn.cluster import KMeans
import optimizer as O 
import solver as S   
from maxent import irl_causal

"""
Monica Babe ̧s-Vroman, Vukosi Marivate, Kaushik Subramanian, and Michael L. Littman.
Apprenticeship Learning About Multiple Intentions. In Proceedings of the 28th International
Conference on Machine Learning, ICML ’11, pages 897–904, Bellevue, WA, USA, 2011. ACM,
New York, NY, USA. ISBN 978-1-4503-0619-5.
"""

states = 100 # number of states 
smoothing_value = 1


init = O.Constant(1.0)
# choose our optimization strategy:
#  we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))


def likelihood(traj, features, p_transition, theta, gamma=0.9, num_clusters=100): 
    # calculate reward 

    reward = features.dot(theta)

    V, Q = S.value_iteration(p_transition, reward, gamma)
    # calculate policy using softmax selection model 

    Q = Q.reshape((5, num_clusters))
    soft_pi = (np.exp(Q)/ np.sum(np.exp(Q), axis = 0)).T
     
    prod = 1 
    n = len(traj)

    for i in range(0, n - 1, 2): 
        state = traj[i]
        action = traj[i + 1]
        prod *= soft_pi[state][action]

    return prod 

def format_traj(trajectories):
    lst = []
    for traj in trajectories:
        row = []
        n = len(traj)
        for i in range(0, n-2, 2):
            row.append((traj[i], traj[i+1], traj[i+2]))
        
        lst.append(row)
    
    return lst

def transition_model(T): 
    """ 
    T: trajectories { tau_1, ..., tau_{N - 1}}

    returns transition model P(s'|s, a) given trajectories T 
    """
    p_transition = np.zeros((states, states, 5)) + smoothing_value

    for traj in T:
        for tran in traj:
            p_transition[tran[0], tran[2], tran[1]] +=1

        p_transition = p_transition/ p_transition.sum(axis = 1)[:, np.newaxis, :]

    return p_transition 

def calc_terminal_states(taus): 
    terminal_states = set()

    for traj in taus:
        terminal_states.add(traj[-1])

    
    return list(terminal_states)

def init_parameters(X, taus, M: KMeans, K=100): 
    """
    X: feature representation of trajectories 
    taus: trajectories 
    M: fitted clustering model with `n_clusters` and `labels._`
    k: number of classes, i.e. experts 

    returns rho_init, u_init, and C: dict of form { cluster : taus_in_cluster }
    """

    n, _ = X.shape 
    C = {} 

    # initialize u 
    u = np.zeros((n, K))

    # initialize rho  
    rho = np.zeros(K)

    for i in range(n):
        for j in range(K):
            c = M.labels_[i]
            u[i][j] = 1 if c == j else 0

            if c in C: 
                C[c] = C[c] + [taus[i]]
            else: 
                C[c] = [taus[i]]
    
    for k in range(K): 
        rho[k] = np.sum([u[i][k] for i in range(n)], axis=0) / n
    
    return rho, u, { k: C[k] for k in sorted(C.keys())} 


def create_clusters(u, taus, K=100): 
    """
    creates new clusters C based on membership variable u
    """

    C = {} 
    for i, tau in enumerate(taus): 
        for k in range(K): 
            if u[i][k] == 1: 
                if k in C: 
                    C[k].append(tau)
                else: 
                    C[k] = [tau]

    return { k: C[k] for k in sorted(C.keys())} 


def limiirl(X, taus, features, M: KMeans, states, K=100, gamma=0.9, epsilon=0.01, max_iter=100):
    """
    X: feature representation of training trajectories 
    taus: training trajectories 
    M: fitted clustering model 
    S: number of states
    K: number of cluster classes 
    max_iter: max iterations of EM
    """

    n, _ = X.shape 

    # calculate initial rho, u, and inital clustering 
    rho, u, C = init_parameters(X, taus, M, K=K)


    # calculate initial theta for each cluster k 
    # for each cluster k, use the max-ent algorithm to obtain a theta estimate 
    theta = np.zeros((K, states))


    # calculate initial theta 
    for k in C:
        # format trajectories (s_1, a_1, s_2, ...) as (s_1, a_1, s_2), (s_2, a_2, ...)
        print(f"LiMIIRL: cluster {k}")
        T = format_traj(C[k])

        p_transition = transition_model(T)

        terminal_states = calc_terminal_states(C[k])

        _, theta_k = irl_causal(p_transition, features, terminal_states, T, optim, init, gamma,
                    eps=1e-3, eps_svf=1e-4, eps_lap=1e-4)
        
        for s in range(states): 
            theta[k][s] = theta_k[s] 

    print("---Finished Light-weight start")

    for it in range(max_iter): 
        prev_u = u 
        for i in range(n):
            for k in range(K): 
                # change call to likelihood
                trans = transition_model(format_traj(C[k]))
                u[i][k] = (rho[k] * likelihood(taus[i], features, trans, theta[k])) / np.sum([rho[k_prime] * likelihood(taus[i], features, transition_model(format_traj(C[k_prime])), theta[k_prime]) for k_prime in range(K)], axis=0)

        # M-step - update parameters 
        for k in range(K):
            rho[k] = np.sum([u[i][k] for i in range(n)]) / n

        # with u[i][k], create new clusters 
        C_prime = create_clusters(u, taus, K=K)
        for k in C_prime: 
            T = format_traj(C_prime[k])

            p_transition = transition_model(T)

            terminal_states = calc_terminal_states(C_prime[k])

            _, theta_prime = irl_causal(p_transition, features, terminal_states, T, optim, init, gamma,
                        eps=1e-3, eps_svf=1e-4, eps_lap=1e-4)
            
            for s in range(states): 
                theta[k][s] = theta_prime[s] 

        converge_cond = 0 
        for i in range(n): 
            for k in range(K): 
                converge_cond += np.abs(u[i][k] - prev_u[i][k])
        converge_cond /= n

        print(it)

        if converge_cond < epsilon: 
            break 
        
        # return model params 
    return rho, theta, u         