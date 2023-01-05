from pympi.Elan import Eaf, to_eaf
from elan_helpers import read_ark_file, mlf_to_dict, make_model_dict, compute_frame_dists
import os
import csv
import shutil
import json
import re
from collections import OrderedDict
from sys import platform
import pandas as pd

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
    video_names = [ vdir.split('/')[-1] for vdir in video_dirs ]
    print(video_names)
    eaf_files = []
    for fname_ext in video_names:
        ext = fname_ext.split('.')[-1]
        fname = fname_ext[:-1*len(ext)-1]
        if fname in data.keys():
            video_fp = video_dirs[video_names.index(fname_ext)]
            if platform == "linux" or platform == "linux2":
              video_fp = "file:///" + video_fp
            out_path = os.path.join(eaf_savedir, fname + '.eaf')

            # Create base eaf file
            shutil.copy(elan_txt, out_path)

            # Create eaf object and link video
            eaf_file = Eaf(out_path)
            print(video_fp)
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
            # print(word, state, start, end)
            # if word not in model_data:
            #     time_slot += 1
            #     continue # why arent any words in model data? word/state pairing not being made correctly
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


def label_worst_frames(dists, eaf_path, video_len, proportion=0.2):
    # get min prop of dist dict and change annotations of annotation file to be labeled something else?
    # or make new annotation for each word?

    # Get outliers from frame dists
    df = pd.DataFrame(dists)
    arr = df.to_numpy()
    medians = np.median(arr, axis=1)
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1

    outliers = arr
    for i in range(arr.shape[0]):
      q1 = np.quantile(arr[i], .25)
      q3 = np.quantile(arr[i], .75)
      iqr = q3 - q1
      outliers[i] = np.array([
        j if j<(q1-1.5*iqr)
        else 0
        for j in arr[i]
      ])

    bad_frames = []
    for feat_outliers in outliers:
      num_outliers = np.count_nonzero(feat_outliers)
      bad_frames.append(num_outliers > len(feat_outliers) * proportion)
    
    # Add bad frames as annotations in new tier
    eaf_obj = Eaf(eaf_path)
    eaf_obj.add_tier("bad_frames")
    for frame_num, is_frame_bad in enumerate(bad_frames):
      if is_frame_bad:
        start = (frame_num-1)/len(outliers) * video_len
        end = frame_num/len(outliers) * video_len
        eaf_obj.add_annotation("bad_frames", int(start), int(end), "")
    to_eaf(eaf_path, eaf_obj)
    return eaf_path



def label_worst_states():
    pass


if __name__=='__main__':
    # Find a video, plot features/gaussians
    ark_fp = '/home/ishan/Documents/research/ccg/copycat/CopyCat-HTK/projects/MediaPipe/data/ark/Jinghong.alligator_above_bed.1612482965.ark'
    # vid_fp = '/home/ishan/Documents/research/ccg/copycat/DATA/input/Jinghong/alligator_above_bed/1612482965/Jinghong.alligator_above_bed.1612482965.mp4'
    vid_fp = "/home/ishan/Documents/research/ccg/elan/landmarks.mp4"
    # eaf_fp = '/home/ishan/Documents/research/ccg/test.eaf' Why isn't this used?
    text_fp = '/home/ishan/Documents/research/ccg/features_ts.txt'
    ll_fp = '/home/ishan/Documents/research/ccg/ll_ts.txt'
    phrase = 'Jinghong.alligator_above_bed.1612482965'
    annotations = mlf_to_dict('/home/ishan/Documents/research/ccg/copycat/CopyCat-HTK/projects/MediaPipe/results/res_hmm5.mlf')
    feature_data = read_ark_file(ark_fp)
    macros_fp = '/home/ishan/Documents/research/ccg/copycat/CopyCat-HTK/projects/MediaPipe/models/hmm5/newMacros'
    feature_labels = json.load(open('/home/ishan/Documents/research/ccg/copycat/CopyCat-HTK/projects/MediaPipe/configs/features.json'))['selected_features']
    model_data = make_model_dict(macros_fp, feature_labels)
    video_len = int(float(FFProbe(vid_fp).video[0].duration) * 1000)
    annotations = scale_annotations(annotations, video_len)
    eaf_obj = make_elan(annotations, has_states=True, video_dirs=[vid_fp], \
        eaf_savedir='/home/ishan/Documents/research/ccg')[0]
    # plot_ll_ts(model_data, feature_data, annotations, ll_fp, phrase, video_len, feature_nums=range(40)) Adding a video works, trying it without ll_ts
    dists = compute_frame_dists(model_data, feature_data, annotations, feature_labels, phrase, video_len)
    label_worst_frames(dists, '/home/ishan/Documents/research/ccg/Jinghong.alligator_above_bed.1612482965.eaf', video_len)