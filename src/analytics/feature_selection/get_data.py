import os
import re
import numpy as np
import json


def read_ark_file(ark_filepath):
    '''Creates array out of ARK file

    Parameters
    ----------
    eaf_filepath : str
        File path at which mlf file is located.

    Returns ndarray with row for each frame
    and col for each feature
    '''
    with open(ark_filepath,'r') as ark_filepath:
        fileAsString = ark_filepath.read()
    contentString = fileAsString.split("[ ")[1].split("\n]")[0].split("\n")
    vectors = [[float(i) for i in frame.split()] for frame in contentString]
    return np.array(vectors)


def read_htk_file(hlist_path, htk_filepath):
    '''Returns numpy array of vectors containing features read from htk file

    Parameters
    ----------
    hlist_path : str
        Path to hlist file.

    htk_path : str
        Path to htk file.
    '''  
    # Return array of vectors where each vector contains feature values for a frame
    process = os.popen("{} -i 100 {}".format(hlist_path, htk_filepath))
    vectors = []

    for line in  process.read().split("\n")[1:-2]: 
        vector = [float(x) for x in re.split("\.[0-9]{3} *", line.split(":")[-1]) if x.strip() != ""]
        vectors.append(vector)
    return np.array(vectors)


def mlf_to_dict(mlf_filepath: str) -> dict:
    '''Generates dictionary from mlf file

    Parameters
    ----------
    eaf_filepath : str
        File path at which mlf file is located.

    Returns dictionary with this format:
    {filename : 
        {word : 
            [
                [state, start, end]
                ...
            ]
        }
        ...
    ...
    }
    '''
    out_dict = {}

    # Iterate over lines of mlf file
    with open(mlf_filepath, "rb") as mlf:
        _ = mlf.readline()
        lines = mlf.readlines()
        for line in lines:
            line = line.decode('utf-8').strip()

            # If line is file name, add new entry in dictionary
            if len(line)>0 and line.endswith("\"") and line[0]=="\"":
                fname = '.'.join(line.split('/')[-1].split('.')[:-1])
                out_dict[fname] = {}

            # If line has state and boundary data
            elif line != '.':
                line_arr = line.split(" ")
                if len(line_arr) >= 5:
                    word = line_arr[4]
                    out_dict[fname][word] = []
                state = line_arr[2]
                start = int(line_arr[0])/1000
                end = int(line_arr[1])/1000
                out_dict[fname][word].append([state, start, end])
        return out_dict


def model(model_path: str) -> dict:
    '''Returns a dictionary of all means and variances for each feature, for each state, for each word

    Parameters
    ----------
    model_path : str
        Path of newMacros file containing all model information for data.
    '''

    # Creates models dict from newMacros file where each model contains data for a word
    models = {}
    current_label = None # current word
    current_state = None # current state
    current_vec   = None # either mean or variance vector
    current_cmp   = [] # current component, which contains a mean array and variance array
    current_model = {} # data corresponding to word
    for line in open(model_path):
        if line.startswith('~h'):
            # Create new entry in models dict corresponding to new word            
            if current_label is not None:
                models[current_label] = current_model
            current_model = {}
            current_label = line[4:-2]            
        elif line.startswith('<STATE>'):
            state_num = line.strip().split(' ')[-1]
            current_state = int(state_num)
            current_model[current_state] = []
        elif line.startswith('<MEAN>'):
            current_vec = 'mean'
        elif line.startswith('<VARIANCE>'):
            current_vec = 'variance'
        elif current_vec == 'mean':
            mean = np.array([float(x) for x in line.strip().split(' ')])
            current_cmp.append(mean)
            current_vec = None
        elif current_vec == 'variance':
            variances = np.array([float(x) for x in line.strip().split(' ')])
            current_cmp.append(variances)
            current_model[current_state].append(current_cmp)
            current_cmp = []
            current_vec = None
    if current_label is not None:
        models[current_label] = current_model
    return models


def get_feature_labels(feature_fp: str) -> dict:
    '''Generates dict containing all selected features for experiment and corresponding IDs

    Parameters
    ----------
    feature_fp : str
        File path to where selected features are located.
    '''
    with open(feature_fp, 'r') as fp:
        features = json.load(fp)['selected_features']
        feature_label_dict = { num: feature for num, feature in enumerate(features) }
    return feature_label_dict
