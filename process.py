from utils import *
from sklearn.cluster import KMeans
import threading



def construct_trajectories(p_events, p_vitals): 
    """
    p_events: events for each patient 
    p_vitals: vital readings for each patient 
    """
    trajs = {} 
    
    for patient in p_events: 
        tau = trajs_from_patient(p_events[patient], p_vitals[patient])
        # drop trajectories with length = 0
        if (len(tau) > 1):
            trajs[int(patient)] = trajs_from_patient(p_events[patient], p_vitals[patient])

    return trajs 
    

def discretize_S(M, n_clusters=100, random_state=42): 
    """
    M: matrix representation of data 

    returns a model that takes in a feature set D -> state s from 0 to K - 1
    where K is the number of clusters 
    """

    # instantiate model and fit data
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto').fit(M)

    # return model 
    return kmeans


if __name__ == "__main__": 

    try: 
        states = int(input("Number of states: ")) 
    except: 
        print("Error: provide valid number for states")

    # read from CSV files 
    patients_df = read_csv_to_dataframe("data/patients.csv")
    inputevents_df = read_csv_to_dataframe("data/inputevents.csv")
    vitalsign_df = read_csv_to_dataframe("data/vitalsign.csv")
    outputevents_df = read_csv_to_dataframe("data/outputevents.csv")
    d_items_df = read_csv_to_dataframe("data/d_items.csv")

    print("---Read from dataframe---")

    # merge patients with vitals dataframe 
    data_pv = pd.merge(patients_df, vitalsign_df, on='subject_id', how='inner')

    # fill in missing values 
    fill_NANS(data_pv)

    # map rhythms  
    unique_rhythms = data_pv['rhythm'].unique()
    n_rhythms = len(unique_rhythms)
    rhythms_mapping = {k:v for k,v in zip(unique_rhythms, range(n_rhythms))}

    features = ["gender", "anchor_age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "rhythm"]

    M = construct_M(data_pv, features, rhythms_mapping, states) 

    print("---Created Matrix M---")

    state_model = discretize_S(M, n_clusters=states)

    print("---discretized state space---")


    inputevents_sample = inputevents_df.sample(n=6000000, random_state = 42)

    n_actions = len(inputevents_df['ordercategorydescription'].unique())
    actions = list(inputevents_df['ordercategorydescription'].unique()) 
    action_mapping = {k:v for k, v in zip(actions, range(n_actions))}

    patient_data = {} 

    patient_events = find_patient_events(inputevents_sample, action_mapping)
    patient_vitals = find_patient_vitals(data_pv, state_model, features, rhythms_mapping)

    print("---constructing trajectories---")

    p_events, p_vitals = intersect_vitals_events(patient_events, patient_vitals)
    trajectories = construct_trajectories(p_events, p_vitals)
    save_json(trajectories, f"data/process/trajs_{states}.json")