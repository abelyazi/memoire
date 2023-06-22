import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.impute import SimpleImputer
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
import os, numpy as np, scipy as sp, scipy.io, scipy.io.wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import pandas as pd

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False


# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False


# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold() == str(y).strip().casefold()
    except AttributeError:  # For Python 2.x compatibility
        return str(x).strip().lower() == str(y).strip().lower()


# Find patient data files.
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension == '.txt':
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames


# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data


# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = sp.io.wavfile.read(filename)
    return recording, frequency


# Load recordings.
def load_recordings_AV(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations + 1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        if 'AV.wav' in filename:
            recording, frequency = load_wav_file(filename)
            recordings.append(recording)
            frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings


# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations + 1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')  # AV 2530_AV.hea 2530_AV.wav 2530_AV.tsv
        recording_file = entries[2]  # 2530_AV.wav
        recording_type = entries[0]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)


    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings


# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split('\n')):
        if i == 0:
            try:
                patient_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return patient_id


# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split('\n')):
        if i == 0:
            try:
                num_locations = int(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_locations


# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, l in enumerate(data.split('\n')):
        if i == 0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency


# Get recording locations from patient data.
def get_locations(data):
    num_locations = get_num_locations(data)
    locations = list()
    for i, l in enumerate(data.split('\n')):
        entries = l.split(' ')
        if i == 0:
            pass
        elif 1 <= i <= num_locations:
            locations.append(entries[0])
        else:
            break
    return locations


# Get age from patient data.
def get_age(data):
    age = None
    for l in data.split('\n'):
        if l.startswith('#Age:'):
            try:
                age = l.split(': ')[1].strip()
            except:
                pass
    return age


# Get sex from patient data.
def get_sex(data):
    sex = None
    for l in data.split('\n'):
        if l.startswith('#Sex:'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex


# Get height from patient data.
def get_height(data):
    height = None
    for l in data.split('\n'):
        if l.startswith('#Height:'):
            try:
                height = float(l.split(': ')[1].strip())
            except:
                pass
    return height


# Get weight from patient data.
def get_weight(data):
    weight = None
    for l in data.split('\n'):
        if l.startswith('#Weight:'):
            try:
                weight = float(l.split(': ')[1].strip())
            except:
                pass
    return weight


# Get pregnancy status from patient data.
def get_pregnancy_status(data):
    is_pregnant = None
    for l in data.split('\n'):
        if l.startswith('#Pregnancy status:'):
            try:
                is_pregnant = bool(sanitize_binary_value(l.split(': ')[1].strip()))
            except:
                pass
    return is_pregnant


# Get murmur from patient data.
def get_murmur(data):
    murmur = None
    for l in data.split('\n'):
        if l.startswith('#Murmur:'):
            try:
                murmur = l.split(': ')[1]
            except:
                pass
    if murmur is None:
        raise ValueError('No murmur available. Is your code trying to load labels from the hidden data?')
    return murmur


# Get outcome from patient data.
def get_outcome(data):
    outcome = None
    for l in data.split('\n'):
        if l.startswith('#Outcome:'):
            try:
                outcome = l.split(': ')[1]
            except:
                pass
    if outcome is None:
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    return outcome


# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = str(x).replace('"', '').replace("'", "").strip()  # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x) == 1) or (x in ('True', 'true', 'T', 't')):
        return 1
    else:
        return 0


# Santize scalar values from Challenge outputs.
def sanitize_scalar_value(x):
    x = str(x).replace('"', '').replace("'", "").strip()  # Remove any quotes or invisible characters.
    if is_finite_number(x) or (is_number(x) and np.isinf(float(x))):
        return float(x)
    else:
        return 0.0


# Save Challenge outputs.
def save_challenge_outputs(filename, patient_id, classes, labels, probabilities):
    # Format Challenge outputs.
    patient_string = '#{}'.format(patient_id)
    class_string = ','.join(str(c) for c in classes)
    label_string = ','.join(str(l) for l in labels)
    probabilities_string = ','.join(str(p) for p in probabilities)
    output_string = patient_string + '\n' + class_string + '\n' + label_string + '\n' + probabilities_string + '\n'

    # Write the Challenge outputs.
    with open(filename, 'w') as f:
        f.write(output_string)


# Load Challenge outputs.
def load_challenge_outputs(filename):
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            if i == 0:
                patient_id = l.replace('#', '').strip()
            elif i == 1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i == 2:
                labels = tuple(sanitize_binary_value(entry) for entry in l.split(','))
            elif i == 3:
                probabilities = tuple(sanitize_scalar_value(entry) for entry in l.split(','))
            else:
                break
    return patient_id, classes, labels, probabilities


# Extract features from the data.
def get_features_wav(data, recordings):
    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)

    recording_features = np.zeros((num_recording_locations, 3), dtype=object)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations == num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                # recording_features[j, 1] = []
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i]) > 0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = recordings[i]
                    recording_features[j, 2] = j
                    # recording_features[j, 2] = np.var(recordings[i])
                    # recording_features[j, 3] = sp.stats.skew(recordings[i])
                    
    print("Locations :",locations)
    print("recordings_features :\n",recording_features)

    """
    Ici le résultat du double for : 
    [[1 array([-425, 1045,  518, ...,  509,  443,  122], dtype=int16)]
    [1 array([12672, 10041,  3215, ...,  -658,     8,   528], dtype=int16)]
    [1 array([2593, 1874, 1370, ..., -637, -255, -243], dtype=int16)]
    [1 array([ 2276,  2343,  2448, ..., -2732, -5634, -4290], dtype=int16)]
    [0 0]] 
    """
    recording_features = recording_features.flatten()
    # features = np.hstack(recording_features)
    features = recording_features
    # return np.asarray(features, dtype=np.float32)
    return features


# Group locations and corresponding numpy arrays .
def get_location_array(data, recordings):
    # Extract recording locations and numpy arrays when a location is present.
    # If there are multiple recordings for one location, then keep informations from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)

    recording_features = np.zeros((num_recording_locations, 2), dtype=object)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations == num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                # recording_features[j, 1] = []
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i]) > 0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = recordings[i]
                    # recording_features[j, 2] = np.var(recordings[i])
                    # recording_features[j, 3] = sp.stats.skew(recordings[i])
    """
    locations = ['AV', 'PV', 'TV', 'MV']
    
    Ici le résultat du double for : 
    recording_features = 
    [[1 array([-425, 1045,  518, ...,  509,  443,  122], dtype=int16)]
    [1 array([12672, 10041,  3215, ...,  -658,     8,   528], dtype=int16)]
    [1 array([2593, 1874, 1370, ..., -637, -255, -243], dtype=int16)]
    [1 array([ 2276,  2343,  2448, ..., -2732, -5634, -4290], dtype=int16)]
    [0 0]] 
    """
    recording_features = recording_features.flatten()
    # features = np.hstack(recording_features)
    features = recording_features
    # return np.asarray(features, dtype=np.float32)
    return features


def build_dataset_df(rec_list, patient_list, patient_murmur):
    new_dict = {}
    new_dict[0] = 1
    new_dict[2] = 2
    new_dict[4] = 3
    new_dict[6] = 4
    new_dict[8] = 5
    rec_type_arr = []
    patient_id_arr = []
    subframes_arr = []
    murmur_arr = []
    k = 0
    for h in range(len(patient_list)):
        for i in range(0, len(rec_list[h]), 2):
            if rec_list[h][i] == 1:
                # Building 1) subarrays of size 4096 for each recording type 2) array for recording type 3) array for patient id
                array = rec_list[h][i + 1]
                indices = [x for x in range(4096, len(array), 4096)]
                sub_arr = np.split(array, indices)
                length = len(sub_arr[-1])
                nb_additional_zeros = 4096 - length
                sub_arr[-1] = np.pad(sub_arr[-1], (0, nb_additional_zeros), 'constant')
                nb_frames = len(sub_arr)

                array_rec = np.zeros(nb_frames)
                array_rec = array_rec + new_dict[i]
                array_id = np.zeros(nb_frames)
                array_id = array_id + int(patient_list[h])
                array_mur = np.zeros(nb_frames)
                array_mur = array_mur + int(patient_murmur[h])

                # rec_type_arr = np.concatenate((rec_type_arr, array_rec), axis=None)
                # patient_id_arr = np.concatenate((patient_id_arr, array_id), axis=None)

                rec_type_arr = rec_type_arr + list(array_rec)
                patient_id_arr = patient_id_arr + list(array_id)
                subframes_arr = subframes_arr + list(sub_arr)
                murmur_arr = murmur_arr + list(array_mur)
                """
                if k==0:
                  subframes_arr = sub_arr
                  k = k + 1
                else:
                  subframes_arr = np.concatenate((subframes_arr,sub_arr), axis=0)
                """
    df = pd.DataFrame({'patient_id': patient_id_arr,
                       'recording_type': rec_type_arr,
                       'recording': subframes_arr,
                       'murmur': murmur_arr})
    return df


"""
def build_dataset_df(rec_list, patient_list, patient_murmur):
    new_dict = {}
    new_dict[0] = 1
    new_dict[2] = 2
    new_dict[4] = 3
    new_dict[6] = 4
    new_dict[8] = 5

    rec_type_arr = []
    patient_id_arr = []
    arrays = []
    murmur_arr = []
    k = 0
    for h in range(len(patient_list)):
        for i in range(0, len(rec_list[h]), 2):
            if rec_list[h][i] == 1:
                # Building
                # 1) subarrays of size 4096 for each recording type
                # 2) array for recording type
                # 3) array for patient id
                array = rec_list[h][i + 1]
                rec_type_arr = rec_type_arr + [new_dict[i]]
                patient_id_arr = patient_id_arr + [int(patient_list[h])]
                arrays = arrays + list(array)
                murmur_arr = murmur_arr + [int(patient_murmur[h])]

    df = pd.DataFrame({'patient_id': patient_id_arr,
                       'recording_type': rec_type_arr,
                       'recording': arrays,
                       'murmur': murmur_arr})
    return df
    """










