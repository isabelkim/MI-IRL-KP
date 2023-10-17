import pandas as pd 
import numpy as np 

def read_csv_to_dataframe(file_path):
    try:
        # Read the CSV file into a Pandas DataFrame
        dataframe = pd.read_csv(file_path)
        return dataframe
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

patients_df = read_csv_to_dataframe("data/patients.csv")
prescriptions_df = read_csv_to_dataframe("data/prescriptions.csv")
inputevents_df = read_csv_to_dataframe("data/inputevents.csv")
procedureevents_df = read_csv_to_dataframe("data/procedureevents.csv")
d_icd_diagnoses_df = read_csv_to_dataframe("data/d_icd_diagnoses.csv")
triage_df = read_csv_to_dataframe("data/triage.csv")
vitalsign_df = read_csv_to_dataframe("data/vitalsign.csv")
prescriptions_df.columns
inputevents_df
d_icd_diagnoses_df

# Merge patients and triage dfs
data_pt = pd.merge(patients_df, triage_df, on='subject_id', how='inner')
# data_pt

# Merge patients and vitals dfs
data_pv = pd.merge(patients_df, vitalsign_df, on='subject_id', how='inner')
# data_pv


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

# Merge patients, vitals, and inputevents dfs
data_pi = pd.merge(data_pv, inputevents_df, on='subject_id', how='inner')

if __name__ == "main": 
    print(data_pi.shape) 