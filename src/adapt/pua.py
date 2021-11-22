from src.prepare_data import prepare_data
from src.utils.file_util import *
from src.utils.cross_val_util import *
from src.utils import get_hresults_data

import os
import glob
from joblib import Parallel, delayed
from statistics import mean

from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneGroupOut, train_test_split)

def pua(args, features_config, all_results):
    stats_list = []
    all_results_list = []

    #### 3-SIGN
    prepare_data(features_config, args.users, phrase_len=[3, 4, 5], prediction_len=[3])
    if not args.users:
        htk_filepaths = glob.glob('data/htk/*htk')
    else:
        htk_filepaths = []
        for user in args.users:
            htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))
    phrases = [' '.join(filepath.split('.')[1].split("_"))
        for filepath
        in htk_filepaths]         
    cross_val_method, _ = (LeaveOneGroupOut(), True)
    users = [get_user(filepath) for filepath in htk_filepaths]
    unique_users = list(set(users))
    unique_users.sort()
    group_map = {user: i for i, user in enumerate(unique_users)}
    groups = [group_map[user] for user in users]   
    cross_val = cross_val_method
    splits = list(cross_val.split(htk_filepaths, phrases, groups))
    stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                    (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                    for currFold in range(len(splits)))
    all_results['average']['error'] = mean([i[0] for i in stats])
    all_results['average']['sentence_error'] = mean([i[1] for i in stats])
    all_results['average']['insertions'] = mean([i[2] for i in stats])
    all_results['average']['deletions'] = mean([i[3] for i in stats])
    print(stats)
    print(all_results)
    input()

    ### 4-SIGN
    prepare_data(features_config, args.users, phrase_len=[3, 4, 5], prediction_len=[4])
    if not args.users:
        htk_filepaths = glob.glob('data/htk/*htk')
    else:
        htk_filepaths = []
        for user in args.users:
            htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))
    phrases = [' '.join(filepath.split('.')[1].split("_"))
        for filepath
        in htk_filepaths]    
    cross_val_method, use_groups = (LeaveOneGroupOut(), True)
    users = [get_user(filepath) for filepath in htk_filepaths]
    unique_users = list(set(users))
    unique_users.sort()
    group_map = {user: i for i, user in enumerate(unique_users)}
    groups = [group_map[user] for user in users]   
    cross_val = cross_val_method
    splits = list(cross_val.split(htk_filepaths, phrases, groups))
    splits = [list(item) for item in splits]
    accepted_3_signs = set()
    for i, htk_filepath in enumerate(htk_filepaths):
        phrase_len = get_phrase_len(htk_filepath)
        phrase_fold = groups[i]
        if phrase_len == 3:
            splits[phrase_fold][1] = np.delete(splits[phrase_fold][1], np.where(splits[phrase_fold][1] == i))
            hresults_filepath = sorted(glob.glob(f'hresults/{phrase_fold}/*.txt'))[1]
            hresults_data = get_hresults_data.get_hresults_data(hresults_filepath)
            if get_video(htk_filepath) not in hresults_data:
                accepted_3_signs.add(htk_filepath)
                splits[phrase_fold][0] = np.append(splits[phrase_fold][0], i)
    stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                    (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                    for currFold in range(len(splits)))
    all_results['average']['error'] = mean([i[0] for i in stats])
    all_results['average']['sentence_error'] = mean([i[1] for i in stats])
    all_results['average']['insertions'] = mean([i[2] for i in stats])
    all_results['average']['deletions'] = mean([i[3] for i in stats])
    print(stats)
    print(all_results)
    input()

    #### 5-SIGN
    prepare_data(features_config, args.users, phrase_len=[3, 4, 5], prediction_len=[5])
    if not args.users:
        htk_filepaths = glob.glob('data/htk/*htk')
    else:
        htk_filepaths = []
        for user in args.users:
            htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))
    phrases = [' '.join(filepath.split('.')[1].split("_"))
        for filepath
        in htk_filepaths]    
    cross_val_method, use_groups = (LeaveOneGroupOut(), True)
    users = [get_user(filepath) for filepath in htk_filepaths]
    unique_users = list(set(users))
    unique_users.sort()
    group_map = {user: i for i, user in enumerate(unique_users)}
    groups = [group_map[user] for user in users]   
    cross_val = cross_val_method
    splits = list(cross_val.split(htk_filepaths, phrases, groups))
    splits = [list(item) for item in splits]
    for i, htk_filepath in enumerate(htk_filepaths):
        phrase_len = get_phrase_len(htk_filepath)
        phrase_fold = groups[i]
        if phrase_len == 3 or phrase_len == 4:
            splits[phrase_fold][1] = np.delete(splits[phrase_fold][1], np.where(splits[phrase_fold][1] == i))
            hresults_filepath = sorted(glob.glob(f'hresults/{phrase_fold}/*.txt'))[1]
            hresults_data = get_hresults_data.get_hresults_data(hresults_filepath)
            if htk_filepath in accepted_3_signs or (phrase_len == 4 and get_video(htk_filepath) not in hresults_data):
                splits[phrase_fold][0] = np.append(splits[phrase_fold][0], i)
    for i, split in enumerate(splits):
        splits[i][0] = np.array(splits[i][0])
        splits[i][1] = np.array(splits[i][1])
    print(type(splits), type(splits[0]), type(splits[0][0]), type(splits[0][1]))
    stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                    (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                    for currFold in range(len(splits)))
    all_results['average']['error'] = mean([i[0] for i in stats])
    all_results['average']['sentence_error'] = mean([i[1] for i in stats])
    all_results['average']['insertions'] = mean([i[2] for i in stats])
    all_results['average']['deletions'] = mean([i[3] for i in stats])
    print(stats)
    print(all_results)

    return all_results, stats