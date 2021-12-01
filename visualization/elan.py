#from pympi.Elan import Eaf, to_eaf
from ...pympi.pympi import Eaf, to_eaf, TSConf, to_tsconf, TSTrack
from .get_data import read_ark_file
import os
import csv
import shutil
import glob

import FFProbe
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


def make_elan(data: dict, has_states: bool, video_dirs: list, eaf_savedir: str) -> None:
    """Generates eaf files from data dict

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
    video_names = [ '.'.join(vname.split('/')[-1].split('.')[:-1]) for vname in video_dirs ]
    for fname in data:
        if fname in video_names:
            video_fp = video_dirs[video_names.index(fname)]

            out_path = os.path.join(eaf_savedir, fname + '.eaf')

            # Create base eaf file
            shutil.copy("elan.txt", out_path)

            # Create eaf object and link video
            eaf_file = Eaf(out_path)
            eaf_file.add_linked_file(video_fp.replace(' ', '\ '), mimetype="video/mp4")

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


def mlf_to_elan(mlf_filepath: str, video_dirs: list, eaf_savedir: str) -> None:
    """Generates eaf files from mlf file

    Parameters
    ----------
    eaf_filepath : str
        File path at which mlf file is located.

    video_dirs : list[str]
        List of videos to create eaf objects with.

    eaf_savedir : str
        Directory under which eaf files are saved.
    """

    # Iterate over lines of mlf file
    with open(mlf_filepath, "rb") as mlf:
        eaf_file = None
        out_path = None
        video_fp = None

        lines = mlf.readlines()
        line_num = 1
        while line_num < len(lines):
            line = str(lines[line_num])
            updated = False

            if len(line) < 10:
                line_num += 1
                continue

            # Move on to next eaf file if new file name is presented
            elif not line[2].isdigit():

                # Save existing data to current eaf object
                if eaf_file:
                    to_eaf(out_path, eaf_file)

                # create filename out of header info
                fname = line.split('/')[-1][:-8]

                # take eaf_savedir and append filename to create out_path
                out_path = os.path.join(eaf_savedir, fname + '.eaf')

                # check if mlf has corresponding video
                for name in video_dirs:
                    if fname == name.split('/')[-1][:-4]:
                        video_fp = name
                        updated = True
                        break
                if not updated:
                    line_num += 1
                    while line_num < len(lines) and len(str(lines[line_num])) > 10:
                        line_num += 1
                    updated = True
                    continue

                # Create base eaf file
                shutil.copy("elan.txt", out_path)

                # Create eaf object and link video
                eaf_file = Eaf(out_path)
                eaf_file.add_linked_file(video_fp.replace(' ', '\ '), mimetype="video/mp4")

            # Gather data from mlf and add tiers, annotations, and start/end times
            else:
                line_arr = line[2:-3].split(" ")
                if len(line_arr) >= 5:
                    word = line_arr[4]
                    eaf_file.add_tier(word)
                state = line_arr[2]
                start = line_arr[0]
                end = line_arr[1]
                eaf_file.add_annotation(word, int(int(start)/1000), int(int(end)/1000), state)
            
            line_num+=1
        
        # Save existing data to current eaf object
        if eaf_file:
            to_eaf(out_path, eaf_file)


def make_tsconf(eaf_obj, eaf_filepath, tsconf_filepath, tracks_list):
    # create tsconf from tracks list and link to eaf
    eaf_obj.add_secondary_linked_file(tsconf_filepath)
    to_eaf(eaf_filepath, eaf_obj)
    new_tsconf = TSConf(tsconf_filepath)
    for new_track in tracks_list:
        new_tsconf.add(new_track.name, new_track)
    to_tsconf(tsconf_filepath, new_tsconf)


def plot_features_ts(ark_filepath, text_filename, video_filepath, feature_nums=[], feature_names=[]):
    # Take ark and convert to TS TXT and return track list
    feats = read_ark_file(ark_filepath)
    video_len = int(float(FFProbe(video_filepath).video[0].duration) * 1000)
    frame_len = video_len / feats.size[0]
    time_col = [frame_len * i for i in range(feats.size[0])]
    tracks_list = []
    data_cols = [time_col]
    if len(feature_nums) > 0:
        for i, feature_num in enumerate(feature_nums):
            data_col = list(feats[:, feature_num])
            data_cols.append(data_col)
            tracks_list.append(TSTrack(feature_names, time_col=0, detect_range=True,\
                data_col=i+1, range_start=min(data_col), range_end=max(data_col)))
    with open(text_filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(list(np.array(data_cols).T)) 
    return tracks_list    


def plot_models_ts():
    # Take newMacros and get means/vars for each word/state/feature and convert to TS TXT and call make_tsconf
    # conversion to txt will need gaussian approximation
    pass


if __name__=='__main__':
    # Find a video, plot features/gaussians
    pass