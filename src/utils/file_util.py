import os
import shutil
import random

import numpy as np

def returnUserDependentSplits(unique_users, htk_filepaths, test_size):
    splits = [[[],[]] for i in range(len(unique_users))]
    for htk_idx, curr_file in enumerate(htk_filepaths):
        curr_user = curr_file.split("/")[-1].split(".")[0].split('_')[-2]
        for usr_idx, usr in enumerate(unique_users):
            if usr == curr_user:
                if random.random() > test_size:
                    splits[usr_idx][0].append(htk_idx)
                else:
                    splits[usr_idx][1].append(htk_idx)
    splits = np.array(splits)
    return splits

def copyFiles(fileNames: list, newFolder: str, originalFolder: str, ext: str):
    if os.path.exists(newFolder):
        shutil.rmtree(newFolder)
    os.makedirs(newFolder)

    for currFile in fileNames:
        shutil.copyfile(os.path.join(originalFolder, currFile+ext), os.path.join(newFolder, currFile+ext))

def get_user(filepath):
    return filepath.split('/')[-1].split('.')[0].split('_')[-2]

def get_phrase_len(filepath):
    return len(os.path.basename(filepath).split('.')[1].split('_'))

def get_video(filepath):
    extension = '.' + os.path.basename(filepath).split('.')[-1]
    return os.path.basename(filepath).replace(extension, '')