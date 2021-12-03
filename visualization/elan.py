#from pympi.Elan import Eaf, to_eaf
from pympi import Eaf, to_eaf, TSConf, to_tsconf, TSTrack
from .get_data import read_ark_file
import os
import csv
import shutil
import glob
from collections import OrderedDict

from ffprobe import FFProbe
import numpy as np


def is_file_name(name: str) -> bool:
    return len(name)>0 and name.endswith("\"") and name[0]=="\""


def mlf_to_dict(mlf_filepath: str):
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
        out_path = None
        header = mlf.readline()
        lines = mlf.readlines()
        line_num = 0
        for line in lines:
            line = line.decode('utf-8').strip()

            # If line is file name, add new entry in dictionary
            if is_file_name(line):
                fname = '.'.join(line.split('/')[-1].split('.')[:-1])
                out_dict[fname] = OrderedDict()

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


def make_elan(data: OrderedDict, has_states: bool, video_dirs: list, eaf_savedir: str) -> None:
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
    print(video_dirs)
    video_names = [ vname.split('/')[-1] for vname in video_dirs ]
    eaf_files = []
    for fname_ext in video_names:
        ext = fname_ext.split('.')[-1]
        fname = fname_ext[:-1*len(ext)-1]
        if fname in data.keys():
            video_fp = video_dirs[video_names.index(fname_ext)]
            out_path = os.path.join(eaf_savedir, fname + '.eaf')

            # Create base eaf file
            shutil.copy("./Research/elan.txt", out_path)

            # Create eaf object and link video
            print(video_fp)
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


def scale_annotations(annotation_data: dict, video_len: int):
    # Get multiplier for each file
    multiplier_dict = {fname: video_len / list(fdata.values())[-1][-1][-1] for fname, fdata in annotation_data.items()}
    print(multiplier_dict)
    for filename, filedata in annotation_data.items():
        mult = multiplier_dict[filename]
        for word, state_list in filedata.items():
            annotation_data[filename][word] = [[state, start*mult, end*mult] for state, start, end in state_list]
    return annotation_data


def plot_features_ts(ark_filepath, text_filename, video_len, feature_nums=[], feature_names=[]):
    # Take ark and convert to TS TXT and return track list
    feats = read_ark_file(ark_filepath)
    frame_len = video_len / feats.shape[0]
    time_col = [frame_len * i for i in range(feats.shape[0])]
    tracks_list = []
    data_cols = [time_col]
    if len(feature_nums) > 0:
        for i, feature_num in enumerate(feature_nums):
            data_col = list(feats[:, feature_num])
            data_cols.append(data_col)
            tracks_list.append(
                TSTrack(feature_names[i], time_col=0, detect_range=True,\
                data_col=i+1, range_start=int(min(data_col)-1), range_end=int(max(data_col)+1))
            )
    with open(text_filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(list(np.array(data_cols).astype(int).T)) 
    return tracks_list    


if __name__=='__main__':
    # Find a video, plot features/gaussians
    ark_fp = '/Users/ishan/Documents/Research/08-14-20_Thad_4K.alligator_in_wagon.0000000000.ark'
    vid_fp = '/Users/ishan/Documents/Research/video_dir/08-14-20_Thad_4K.alligator_in_wagon.0000000000.m4v'
    eaf_fp = '/Users/ishan/Documents/Research/08-14-20_Thad_4K.alligator_in_wagon.0000000000.eaf'
    text_fp = '/Users/ishan/Documents/Research/features_ts.txt'
    annotations = mlf_to_dict('/Users/ishan/Documents/Research/reduced.mlf')
    video_len = int(float(FFProbe(vid_fp).video[0].duration) * 1000)
    annotations = scale_annotations(annotations, video_len)
    eaf_obj = make_elan(annotations, has_states=True, video_dirs=[vid_fp], \
        eaf_savedir='/Users/ishan/Documents/Research')[0]
    