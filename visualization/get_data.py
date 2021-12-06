import os
import re
import numpy as np
import json


def read_ark_file(ark_filepath):
    '''Creates array out of ARK file

    Parameters
    ----------
    ark_filepath : str
        File path at which ark file is located.

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
    mlf_filepath : str
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


def make_model_dict(macros_filepath, feature_labels):
    """Processes raw macros data extracted from HMMs and coverts the data into a dictionary macros_data:
        [word][state_number][mixture_number][mean/variance/gconst/mixture_weight][if mean/var then feature label].

    Parameters
    ----------
    feature_labels : list of str
        List of features from the feature config file used when testing and training the HMM.

    macros_filepath : str
        File path to the corresponding newMacros result file that is generated from running HMM.

    Returns
    -------
    macros_data : dictionary
        The data extracted from the newMacros file in the following format:
        [word][state_number][mixture_number][mean/variance/gconst/mixture_weight][if mean/variance then feature_label].

    """
    macros_data = {}
    macro_lines = [ line.rstrip() for line in open(macros_filepath, "r") ]

    i = 0
    while i != len(macro_lines):
        while i != len(macro_lines) and "~h" not in macro_lines[i]:
            i += 1
        if i == len(macro_lines): break
        word = macro_lines[i].split("\"")[1]
        macros_data[word] = {}
        
        while "<NUMSTATES>" not in macro_lines[i]:
            i += 1
        num_states = int(macro_lines[i].split(" ")[1])
        for num_state in range(2, num_states):
            while "<STATE>" not in macro_lines[i]:
                i += 1
            macros_data[word][num_state] = {}

            while "<NUMMIXES>" not in macro_lines[i]:
                i += 1
            num_mixes = int(macro_lines[i].split(" ")[1])
            for num_mix in range(1, num_mixes + 1):
                macros_data[word][num_state][num_mix] = {}

                i += 1
                macros_data[word][num_state][num_mix]["mixture_weight"] = float(macro_lines[i].split(" ")[2])

                i += 2
                mean_list = macro_lines[i].split(" ")[1:]
                mean_list = [ float(item) for item in mean_list ]
                macros_data[word][num_state][num_mix]["mean"] = dict(zip(feature_labels, mean_list))

                i += 2
                variance_list = macro_lines[i].split(" ")[1:]
                variance_list = [ float(item) for item in variance_list ]
                macros_data[word][num_state][num_mix]["variance"] = dict(zip(feature_labels, variance_list))

                i += 1
                macros_data[word][num_state][num_mix]["gconst"] = float(macro_lines[i].split(" ")[1])

        while "<ENDHMM>" not in macro_lines[i]:
            i += 1
    return macros_data


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
