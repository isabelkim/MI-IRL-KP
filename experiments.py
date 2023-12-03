
import pandas as pd 
import numpy as np 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from maxent import irl, irl_causal, feature_expectation_from_trajectories
import optimizer as O 
import solver as S                          # MDP solver (value-iteration)
import plot as P
from sklearn.preprocessing import OneHotEncoder
from limiirl import limiirl, phi, find_policy
from utils import *
from scipy.stats import wasserstein_distance
import random 
from datetime import datetime 


def feature_trajectories(taus, gamma=0.9): 
    """ 
    taus: dictionary of the form { patient: tau } where tau is the patient's trajectory 
    during stay in ICU (from MIMIC dataset)

    gamma: discount factor 

    returns X of shape N x d, where N is the number of trajectories and d represents number of features 
    """
    X = [] 

    for patient in taus: 
        traj = taus[patient] 
        phi_t = phi(traj, f, gamma)
        X.append([phi_t])

    return np.array(X)   


def convert_traj(trajectories):
    lst = []
    for patient in trajectories:
        traj = trajectories[patient]
        row = []
        n = len(traj)
        for i in range(0, n-2, 2):
            row.append((traj[i], traj[i+1], traj[i+2]))
        
        lst.append(row)
    
    return lst


def calc_tran_model(taus, states, smoothing_value=1): 
    p_transition = np.zeros((states, states, actions)) + smoothing_value

    for traj in taus:
        for tran in traj:

            p_transition[tran[0], tran[2], tran[1]] +=1

        p_transition = p_transition/ p_transition.sum(axis = 1)[:, np.newaxis, :]

    return p_transition

def calc_terminal_states(taus): 
    terminal_states = set() 

    for patient in taus: 
        terminal_states.add(taus[patient][-1])
    
    return list(terminal_states)


def train_single_intent(): 
    # set up features: we use one feature vector per state (1 hot encoding for each cluster/state)
    # choose our parameter initialization strategy:
    #   initialize parameters with constant
        init = O.Constant(1.0)

        # choose our optimization strategy:
        #   we select exponentiated stochastic gradient descent with linear learning-rate decay
        optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

        # actually do some inverse reinforcement learning
        # reward_maxent = maxent_irl(p_transition, features, terminal_states, trajectories, optim, init, eps= 1e-3)

        reward_maxent_causal, theta_causal = irl_causal(p_transition, features, terminal_states, T, optim, init, discount,
                    eps=1e-3, eps_svf=1e-4, eps_lap=1e-4)
        
        theta_list = list(theta_causal)
        reward_list = list(reward_maxent_causal)

        saved_data = { "theta": theta_list, "reward": reward_list }

        save_json(saved_data, "data/results/single_intent.json")
        
        return reward_maxent_causal, theta_causal


def cluster_trajectories(X, n_experts=100): 
    """
    X: feature representation of trajectories
    n_experts: assume trajectories come from `n_experts` experts 

    performs kmeans clustering on X and returns model 
    """
    kmeans = KMeans(n_clusters=n_experts, random_state=42).fit(X)

    return kmeans 

# calculate starting state probabilities 
def calc_start_dist(taus, S): 
    X = np.zeros(S)
    n = len(taus)

    for tau in taus: 
        X[tau[0]] += 1 

    return X / n 

if __name__ == "__main__": 

    states = 0 
    K = 0 
    n_patients = 0 

    try: 
        states = int(input("Number of states: ")) 
        n_patients = int(input("Number of patients: "))
        K = int(input("Number of experts: "))


    except: 
        print("Error: provide valid number for states")

    M = read_json(f"data/process/M_{states}.json")
    M = np.array(M) 

    print("--Read matrix M--")


    trajectories = read_json(f"data/process/trajs_{states}.json")
    trajectories = { int(k) : v for k, v in zip(trajectories.keys(), trajectories.values()) }

    print("--Read Trajectories--")

    actions = 5 
    discount = 0.9 
    T = convert_traj(trajectories)

    state_encoder = OneHotEncoder(sparse=False, categories= [np.arange(states)])

    p_0 = calc_start_dist(list(trajectories.values()), states)
    terminal_states = calc_terminal_states(trajectories)
    p_transition = calc_tran_model(T, states=states)
    features = state_encoder.fit_transform(np.arange(states).reshape(-1, 1))

    reward_single, theta_single = train_single_intent()

    print("---Trained Single Intention Model---")

    _, soft_pi = find_policy(p_transition, reward_single, states)

    gamma = 0.9

    random_patients = random.sample(list(trajectories.keys()), n_patients)

    trajectories_s = { patient: trajectories[patient] for patient in random_patients }

    ts = datetime.now().timestamp()

    save_json(trajectories_s, f"data/samples/trial_{ts}_{states}.json")

    f = feature_expectation_from_trajectories(features, T)
    X = feature_trajectories(trajectories_s)

    cluster_model = cluster_trajectories(X, n_experts=K)
    print("---Clustered Sampled Trajectories---")

    max_iterations = 30
    taus = list(trajectories_s.values())
    epsilon = 0.005

    print("---Start LIMIIRL---")
    rho, theta, u = limiirl(X, taus, features, cluster_model, states, f=f, transition=p_transition, epsilon=0.001, max_iter=max_iterations, K=K, descent_iter=50)

    print("---Finished LIMIIRL---")

    np.savez(f"data/experiments/trial_{ts}_{states}", u=u, rho=rho, theta=theta)



    