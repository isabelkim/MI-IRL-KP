import numpy as np
from sklearn.cluster import KMeans
import optimizer as O 
import solver as S   
from maxent import irl_causal
from maxent import irl_causal, feature_expectation_from_trajectories
import math 


"""
Monica Babe ̧s-Vroman, Vukosi Marivate, Kaushik Subramanian, and Michael L. Littman.
Apprenticeship Learning About Multiple Intentions. In Proceedings of the 28th International
Conference on Machine Learning, ICML ’11, pages 897–904, Bellevue, WA, USA, 2011. ACM,
New York, NY, USA. ISBN 978-1-4503-0619-5.
"""

smoothing_value = 1


init = O.Constant(1.0)
# choose our optimization strategy:
#  we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

actions = 5

def find_policy(p_transition, reward, states, discount=0.9):
    """ 
    p_transition: transition probabilities
    reward: reward function 
    discount: discount factor 

    calculates V and Q function using value iteration 
    
    returns softmax policy and policy 
    """ 
    V, Q = S.value_iteration(p_transition, reward, discount)
    Q = Q.reshape((actions, states))

    # get softmax policy 
    soft_pi = (np.exp(Q)/ np.sum(np.exp(Q), axis = 0)).T
    policy = np.argmax(Q, axis = 0).reshape(-1, )

    return policy, soft_pi 

def phi(tau, f, gamma): 
    """
    phi is a feature-based function that maps a trajectory `tau` -> R 
    
    tau: trajectory of the form (s_1, a_1, s_2, a_2, ..., s_{|tau|})
    f: feature function: S -> R
    gamma: discount factor 

    phi(tau) = \sum_{t = 1}^{|tau|}} gamma^{t - 1}phi(s_t) 

    define |tau| as number of (s_t, a_t, s_{t + 1}) tuples 
    """

    n = len(tau)
    return np.sum([(gamma ** (i // 2)) * f[tau[i]]  for i in range(0, n, 2)], axis = 0)

def q_tau(p_0, tau, p_transition):
    """
    tau: let tau be of the form [s_1, a_1, s_2, ...,]
    p_0: starting state distribution 
    p_transition: transition model 
    calculate q(tau)
    """

    n = len(tau)
    q = p_0[tau[0]]

    for i in range(0, n - 2, 2): 
        q *= p_transition[tau[i], tau[i + 2], tau[i + 1]] 

    return q 

def z_theta(theta, taus, p_0, p_transition, f, gamma): 
    z = 0 

    for tau in taus: 
        term = q_tau(p_0, tau, p_transition) * np.exp(theta * phi(tau, f, gamma))

        z += term 

    return z 



def likelihood(tau, states, features, p_transition, theta, gamma=0.9): 
    # calculate reward 

    n = len(tau) 

    reward = features.dot(theta)

    _, soft_pi = find_policy(p_transition, reward, states, gamma)

    l = 0

    for i in range(0, n - 2, 2):
        if soft_pi[tau[i], tau[i + 1]] == 0: 
            return 0
        l += np.log(soft_pi[tau[i], tau[i + 1]])

    if math.isnan(l): 
        return 0 
    
    return l 


def gradient_log_likelihood(tau, f, features, states, p_transition, theta, gamma=0.9): 
    reward = features.dot(theta)

    _, soft_pi = find_policy(p_transition, reward, states, gamma)

    term = 0 

    n = len(tau)

    for i in range(0, n - 2, 2): 
        fi = np.full(6, f[tau[i]])
        term += f[tau[i]] - np.dot(soft_pi[tau[i]], fi)

    return term 


def format_traj(trajectories):
    lst = []
    for traj in trajectories:
        row = []
        n = len(traj)
        for i in range(0, n-2, 2):
            row.append((traj[i], traj[i+1], traj[i+2]))
        
        lst.append(row)
    
    return lst

def transition_model(T, states): 
    """ 
    change this 

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


def limiirl(X, taus, features, M: KMeans, states, transition, f, K=100, gamma=0.9, epsilon=0.0001, max_iter=100, alpha=0.2, descent_iter=200, run_EM=True):
    """
    X: feature representation of training trajectories 
    taus: training trajectories 
    M: fitted clustering model 
    S: number of states
    K: number of cluster classes 
    max_iter: max iterations of EM
    gradient_descent: iter
    alpha: learning rate
    """

    n, _ = X.shape 

    # calculate initial rho, u, and inital clustering 
    rho, u, C = init_parameters(X, taus, M, K=K)


    # calculate initial theta for each cluster k 
    # for each cluster k, use the max-ent algorithm to obtain a theta estimate 
    theta = np.zeros((K, states))

    for k in C:
        # format trajectories (s_1, a_1, s_2, ...) as (s_1, a_1, s_2), (s_2, a_2, ...)
        T = format_traj(C[k])


        terminal_states = calc_terminal_states(C[k])

        _, theta_k = irl_causal(transition, features, terminal_states, T, optim, init, gamma,
                    eps=1e-3, eps_svf=1e-4, eps_lap=1e-4)
        
        print(f"LiMIIRL: cluster {k}")

        
        for s in range(states): 
            theta[k][s] = theta_k[s] 

    print("---Finished Light-weight start---")

    if run_EM: 

        for it in range(max_iter): 

            print(f"EM iteration: {it + 1}")

            # E-step
            prev_u = u 
            for i in range(n):
                for k in range(K): 
                    # change call to likelihood
                    l = likelihood(taus[i], states, features, transition, theta[k], gamma)
                    print(f"E-step: {i, k}")

                    denom = 0 
                    for k_prime in range(K): 
                        denom += rho[k_prime] * likelihood(taus[i], states, features, transition, theta[k_prime], gamma)

                    # if math.isnan(denom) or denom == 0: 
                    #     u[i][k] = 0 
                    #     continue 
                    u[i][k] = (rho[k] * l) / denom 

            print("---E-step---")
            # M-step - update parameters 
            for k in range(K):
                rho[k] = np.sum([u[i][k] for i in range(n)]) / n

            print("--M-step--")
            # update theta 
            for k in range(K): 
                # perform gradient descent 
                for descent_iter in range(descent_iter): 
                    # print(f"Descent iteration: {descent_iter}, expert: {k}")
                    s = 0 
                    for i_prime in range(n): 
                        s += u[i][k] * gradient_log_likelihood(taus[i_prime], f, features, states, transition, theta[k], gamma)
                    theta[k] = theta[k] + alpha * s 

            converge_cond = 0 
            for i in range(n): 
                for k in range(K): 
                    converge_cond += np.abs(u[i][k] - prev_u[i][k])
            converge_cond /= n

            if converge_cond < epsilon: 
                break 
            
            # return model params 
        return rho, theta, u         
    
    return rho, theta, u