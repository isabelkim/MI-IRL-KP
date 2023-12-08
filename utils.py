import math 
import pandas as pd 
import json
import numpy as np
import datetime 

def save_json(data, filename):
    """
    Write a Python object to a JSON file.
    
    Args:
        data: The Python object to be serialized and written to the file.
        filename (str): The name of the JSON file where the data will be saved.
        
    Returns:
        None
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def read_json(filename):
    """
    Reads a JSON file and returns the data.
    
    :param filename: str, path to the JSON file.
    :return: data read from the JSON file.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

    

def read_csv_to_dataframe(file_path):
    try:
        # Read the CSV file into a Pandas DataFrame
        dataframe = pd.read_csv(file_path)
        dataframe.bfill(inplace=True)
        return dataframe
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def fill_NANS(df):
    columns = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

    medians = {}

    # Define cutoff ranges for each column
    cutoff_ranges = {
        # "temperature": (70, 120),
        "heartrate": (0, 600),
        "resprate": (0, 900),
        "sbp": (50, 370),
        "dbp": (20, 360),
        "value": (0, 200)
    }

    for col in columns:
        col_median = df[col].median()
        medians[col] = col_median

    for index, row in df.iterrows():
        for col in columns:
            if math.isnan(row[col]):
                # If the value is NaN, replace it with the median
                df.at[index, col] = medians[col]
            elif col in cutoff_ranges:
                # Check if the value is within the specified range, otherwise replace with the median
                min_val, max_val = cutoff_ranges[col]
                if not (min_val <= row[col] <= max_val):
                    df.at[index, col] = medians[col]

    return df

def map_rhythm(r, rhythms_mapping): 
    return rhythms_mapping[r]


def feature_map(row, features, rhythms_mapping): 
    """
    row: row from dataframe 'data_pv' 

    return a feature mapping of the row (i.e. row of matrix M)
    """
    r = [] 
    for feature in features: 
        if feature == "gender":
            r.append(0 if row[feature] == "M" else 1)
        elif feature == "rhythm": 
            r.append(map_rhythm(row[feature], rhythms_mapping))
        else:
            r.append(row[feature])

    return r 

def construct_M(df, features, rhythms_mapping, states):
    M = []
    for _, row in df.iterrows():
        r = feature_map(row, features, rhythms_mapping)
        M.append(r)
    
    save_json(M, f"data/process/M_{states}.json")
    return np.array(M)


def action_map(I, action_mapping):
    """
    I: event from inputevent dataframe 

    returns an action that represents event 
    """

    return action_mapping[I]


def find_patient_events(events_df, action_mapping): 
    subject_events = {} 
    
    for _, event in events_df.iterrows(): 
        subject = event['subject_id']

        patient_event = { 'caregiver': event['caregiver_id'], 'starttime': event['starttime'], 'endtime': event['endtime'], 'action': action_map(event['ordercategorydescription'], action_mapping), "type": 'action' }

        if subject not in subject_events: 
            value = [patient_event]
            subject_events[subject] = value 
        else: 
            subject_events[subject].append(patient_event)

    # sort these by time
    for s in subject_events: 
        subject_events[s] = sorted(subject_events[s], key=lambda x: x['starttime'])
    return subject_events

def group_timestamps_by_day(events, key):
    timestamp_dict = {}

    for event in events:
        # Parse the timestamp string into a datetime object
        dt = datetime.strptime(event[key], '%Y-%m-%d %H:%M:%S')
        
        # Extract the day part (date) from the datetime object
        day = dt.date()
        
        # Convert the date back to a string
        day_str = day.strftime('%Y-%m-%d')
        
        # Add the timestamp to the corresponding day's list in the dictionary
        if day_str not in timestamp_dict:
            timestamp_dict[day_str] = [event]
        else:
            timestamp_dict[day_str].append(event)

    return timestamp_dict

def find_patient_vitals(data_pv, state_model, features, rhythms_mapping): 
    subject_vitals = {} 
    
    for _, vitals in data_pv.iterrows(): 
        subject = vitals['subject_id']

        state = state_model.predict(np.array([feature_map(vitals, features, rhythms_mapping)]))

        patient_vital = { 'charttime': vitals['charttime'], 'state': state[0], "type": 'state' }
        
        if subject not in subject_vitals: 
            value = [patient_vital]
            subject_vitals[subject] = value 
        else: 
            subject_vitals[subject].append(patient_vital)

    # sort these by time
    for s in subject_vitals: 
        subject_vitals[s] = sorted(subject_vitals[s], key=lambda x: x['charttime'])
    return subject_vitals


def intersect_vitals_events(patient_events, patient_vitals): 
    new_patient_events = {}
    new_patient_vitals = {}

    for patient in patient_vitals: 
        if patient in patient_events: 
            new_patient_events[patient] = patient_events[patient]
            new_patient_vitals[patient] = patient_vitals[patient]

    return new_patient_events, new_patient_vitals 



def trajs_from_patient(event_series, vital_series): 
    """  
    event_series: inputevents applied on subject with 'subject_id'
    vital_series: vitals recorded for subject with 'subject_id' 

    iterates through combined event and vitals series S, and in order, for each state s, finds if action 
    occurs immediately after it?
    """
    # construct trajectory of state action pairs 
    T = [] 

    combined_series = sorted(event_series + vital_series, key=lambda x: x['starttime'] if 'starttime' in x else x['charttime'])

    n = len(combined_series)

    for i in range(n - 1): 
        event1 = combined_series[i]
        event2 = combined_series[i + 1]
        if event1['type'] == "state" and event2['type'] == 'action': 
            T.append(int(event1['state']))
            T.append(int(event2['action']))
        # elif event1['type'] == "state" and event2['type'] == "state": 
        #     T.append(int(event1['state']))
        #     T.append(0) 
        

    # taking the last vital reading that was recorded 
    # though we should check if this occurs before the last action in T?
    # also, issue arises if we have something like [event, vital] which maps to traj [event, vital, event]
    # for this we possibly drop trajectories where vital_series has length < 2?
    # perhaps, as we loop, we can check the lastest_action to be added 
    n_vitals = len(vital_series)
    T.append(int(vital_series[n_vitals - 1]['state']))
    
    return T