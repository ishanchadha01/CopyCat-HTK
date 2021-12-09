import os
import re
import json
from tqdm import tqdm

import numpy as np
from scipy.stats import norm 


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
        print('Finding boundaries')
        for line in tqdm(lines):
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


def scale_annotations(annotation_data: dict, video_len: int):
    # Get multiplier for each file
    multiplier_dict = {fname: video_len / list(fdata.values())[-1][-1][-1] for fname, fdata in annotation_data.items()}
    for filename, filedata in annotation_data.items():
        mult = multiplier_dict[filename]
        for word, state_list in filedata.items():
            annotation_data[filename][word] = [[state, start*mult, end*mult] for state, start, end in state_list]
    return annotation_data


def make_ll_dict(model_data, feature_data, phrase, annotations, feature_labels, video_len):
    # returns dict with {feature: [max_ll for each frame]}
    num_frames = feature_data.shape[0]
    num_features = feature_data.shape[1]
    frame_len = video_len / num_frames
    time_col = [frame_len * i for i in range(feature_data.shape[0])]
    ll_dict = {}
    word_state_times = [(word, state, start, end) for word, state_info in\
        annotations[phrase].items() for state, start, end in state_info]
    for feature_num in range(num_features):
        feat_over_time = list(feature_data[:, feature_num])
        data_col = []
        frame = 0
        time_slot = 0
        while frame < num_frames:
            word,state,start,end = word_state_times[time_slot]
            if start<=time_col[frame]<=end:
                max_ll = float('-inf')
                state = int(re.findall(r'\d+', state)[-1])
                for mixture in model_data[word][state].values():
                    mean = mixture['mean'][feature_labels[feature_num]]
                    var = mixture['variance'][feature_labels[feature_num]]
                    ll = norm.logpdf(feat_over_time[frame], mean, var)
                    max_ll = max(ll, max_ll)
                data_col.append(max_ll)
                frame += 1
            else:
                time_slot += 1
        ll_dict[feature_labels[feature_num]] = data_col
    return ll_dict


def get_ll_word(word_ll, phrase, annotations, model_data, feature_data, feature_labels):
    # gets max likelihood for word for each feature
    # for ark, if frame corresponds to correct word then add it to data col
    # take median of word for specific video
    num_frames = feature_data.shape[0]
    num_features = feature_data.shape[1]
    video_len = max([word[-1][-1] for word in annotations[phrase].values()])
    frame_len = video_len / num_frames
    time_col = [frame_len * i for i in range(feature_data.shape[0])]
    ll_list = []
    word_state_times = [(word, state, start, end) for word, state_info in\
        annotations[phrase].items() for state, start, end in state_info]
    for feature_num in range(num_features):
        feat_over_time = list(feature_data[:, feature_num])
        data_col = []
        frame = 0
        time_slot = 0
        while frame < num_frames:
            word,state,start,end = word_state_times[time_slot]
            if start<=time_col[frame]<=end:
                if word==word_ll:
                    max_ll = float('-inf')
                    state = int(re.findall(r'\d+', state)[-1])
                    for mixture in model_data[word][state].values():
                        mean = mixture['mean'][feature_labels[feature_num]]
                        var = mixture['variance'][feature_labels[feature_num]]
                        ll = norm.logpdf(feat_over_time[frame], mean, var)
                        max_ll = max(ll, max_ll)
                    data_col.append(max_ll)
                frame += 1
            else:
                time_slot += 1
        ll_list.append(max(data_col))
    return ll_list


def rank_ll(word1, word2, annotations, model_data, feature_data, feature_labels, data_dir):
    phrases = annotations.keys()
    print('Finding word1 medians')
    ll_medians_word1 = np. median(np.array([get_ll_word(word1, '.'.join(fname.split('/')[-1].split('.')[:-1]), annotations, model_data, feature_data_dict[fname], feature_labels) \
        for fname in tqdm(data_dir) if f'_{word1}_' in fname and '.'.join(fname.split('/')[-1].split('.')[:-1]) in phrases]), axis=0)
    print('Finding word2 medians')
    ll_medians_word2 = np.median(np.array([get_ll_word(word2, '.'.join(fname.split('/')[-1].split('.')[:-1]), annotations, model_data, feature_data_dict[fname], feature_labels) \
        for fname in tqdm(data_dir) if f'_{word2}_' in fname and '.'.join(fname.split('/')[-1].split('.')[:-1]) in phrases]), axis=0)

    print('Highest Importance Word1:')
    imp = sorted(enumerate(list(ll_medians_word1)), key=lambda x:x[1], reverse=True)
    for idx, val in imp:
        print(feature_labels[idx], val)

    print('Highest Importance Word2:')
    imp = sorted(enumerate(list(ll_medians_word2)), key=lambda x:x[1], reverse=True)
    for idx, val in imp:
        print(feature_labels[idx], val)

    print('Largest Differences:')
    imp = sorted(enumerate([abs(a-b) for a, b in zip(list(ll_medians_word1), list(ll_medians_word2))]), \
        key=lambda x:x[1], reverse=True)
    for idx, val in imp:
        print(feature_labels[idx], val)



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


if __name__=='__main__':
    annotations = mlf_to_dict('/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/SBHMM-HTK/SequentialClassification/main/projects/Kinect/results/11/res_hmm220.mlf')
    feature_labels = json.load(open('/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/SBHMM-HTK/SequentialClassification/main/projects/Kinect/configs/features.json'))['selected_features']
    model_data = make_model_dict('/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/SBHMM-HTK/SequentialClassification/main/projects/Kinect/models/11/hmm220/newMacros', feature_labels)
    data_dir = '/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/SBHMM-HTK/SequentialClassification/main/projects/Kinect/data/ark'
    data_fp_list = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) \
        if os.path.splitext(os.path.join(data_dir, fname))[-1] == '.ark' and (('_in_' in fname) or ('_and_' in fname))]
    feature_data_dict = {fname: read_ark_file(fname) \
        for fname in data_fp_list}
    rank_ll('in', 'above', annotations, model_data, feature_data_dict, feature_labels, data_fp_list)