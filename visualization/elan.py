#from pympi.Elan import Eaf, to_eaf
from pympi.Elan import Eaf, to_eaf
from get_data import read_ark_file, mlf_to_dict, make_model_dict
import os
import csv
import shutil
import json
import re
from collections import OrderedDict

from ffprobe import FFProbe
import numpy as np
from scipy.stats import norm 


def is_file_name(name: str) -> bool:
    return len(name)>0 and name.endswith("\"") and name[0]=="\""


def make_elan(data: OrderedDict, has_states: bool, video_dirs: list, eaf_savedir: str, elan_txt='./elan.txt') -> None:
    """Generates eaf files from data dict
    Returns eaf object list

    Parameters
    ----------
    data : dict
        Segmentation data extracted from mlf files.

    has_states : bool
        Whether or not to write individual states to eaf.

    video_dirs : list[str]
        List of videos to create eaf objects with.

    eaf_savedir : str
        Directory under which eaf files are saved.
    """
    video_names = [ vname.split('/')[-1] for vname in video_dirs ]
    eaf_files = []
    for fname_ext in video_names:
        ext = fname_ext.split('.')[-1]
        fname = fname_ext[:-1*len(ext)-1]
        if fname in data.keys():
            video_fp = video_dirs[video_names.index(fname_ext)]
            out_path = os.path.join(eaf_savedir, fname + '.eaf')

            # Create base eaf file
            shutil.copy(elan_txt, out_path)

            # Create eaf object and link video
            eaf_file = Eaf(out_path)
            eaf_file.add_linked_file(video_fp.replace(' ', '\ '), mimetype=f"video/{ext}")

            # Iterate over segmentation data
            if not has_states:
                eaf_file.add_tier(fname)
                for word in data[fname]:
                    start = int(data[fname][word][0][1])
                    end = int(data[fname][word][-1][2])
                    eaf_file.add_annotation(fname, start, end, word)
            else:
                for word in data[fname]:
                    eaf_file.add_tier(word)
                    for state in data[fname][word]:
                        state_num = state[0]
                        start = state[1]
                        end = state[2]
                        eaf_file.add_annotation(word, int(start), int(end), state_num)
            
            # Create eaf out of data
            to_eaf(out_path, eaf_file)
            eaf_files.append(eaf_file)
    return eaf_files

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


def scale_annotations(annotation_data: dict, video_len: int):
    # Get multiplier for each file
    multiplier_dict = {fname: video_len / list(fdata.values())[-1][-1][-1] for fname, fdata in annotation_data.items()}
    for filename, filedata in annotation_data.items():
        mult = multiplier_dict[filename]
        for word, state_list in filedata.items():
            annotation_data[filename][word] = [[state, start*mult, end*mult] for state, start, end in state_list]
    return annotation_data


def plot_features_ts(feature_data, text_filename, video_len, feature_nums=[]):
    # Take ark and convert to TS TXT and return track list
    frame_len = video_len / feature_data.shape[0]
    time_col = [frame_len * i for i in range(feature_data.shape[0])]
    data_cols = [time_col]
    for feature_num in feature_nums:
        data_col = list(feature_data[:, feature_num])
        data_cols.append(data_col)

    with open(text_filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(list(np.array(data_cols).astype(int).T))  


def plot_ll_ts(model_data, feature_data, annotations, text_filename, phrase, video_len, feature_nums=[]):
    # plot mean/variance of specific feature
    num_frames = feature_data.shape[0]
    frame_len = video_len / num_frames
    time_col = [frame_len * i for i in range(feature_data.shape[0])]
    data_cols = [time_col]
    word_state_times = [(word, state, start, end) for word, state_info in\
        annotations[phrase].items() for state, start, end in state_info]
    for feature_num in feature_nums:
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
        data_cols.append(data_col)

    with open(text_filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(list(np.array(data_cols).astype(int).T))


def plot_kl_divergence(model1, model2, compare_states=[], state_names=[]):
    # plot distance between gaussians
    pass


if __name__=='__main__':
    # Find a video, plot features/gaussians
    ark_fp = '/Users/ishan/Documents/Research/08-14-20_Thad_4K.alligator_in_wagon.0000000000.ark'
    vid_fp = '/Users/ishan/Documents/Research/video_dir/08-14-20_Thad_4K.alligator_in_wagon.0000000000.m4v'
    eaf_fp = '/Users/ishan/Documents/Research/08-14-20_Thad_4K.alligator_in_wagon.0000000000.eaf'
    text_fp = '/Users/ishan/Documents/Research/features_ts.txt'
    ll_fp = '/Users/ishan/Documents/Research/ll_ts.txt'
    phrase = '08-14-20_Thad_4K.alligator_in_wagon.0000000000'
    annotations = mlf_to_dict('/Users/ishan/Documents/Research/reduced.mlf')
    feature_data = read_ark_file(ark_fp)
    macros_fp = '/Users/ishan/Documents/Research/newMacros'
    feature_labels = json.load(open('/Users/ishan/Documents/Research/features.json'))['selected_features']
    model_data = make_model_dict(macros_fp, feature_labels)
    video_len = int(float(FFProbe(vid_fp).video[0].duration) * 1000)
    annotations = scale_annotations(annotations, video_len)
    eaf_obj = make_elan(annotations, has_states=True, video_dirs=[vid_fp], \
        eaf_savedir='/Users/ishan/Documents/Research')[0]
    plot_ll_ts(model_data, feature_data, annotations, ll_fp, phrase, video_len, feature_nums=range(40))
    