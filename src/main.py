"""Main file used to prepare training data, train, and test HMMs.
    HMM EX = python3 driver.py --test_type standard --train_iters 25 50 --users Naoki --hmm_insertion_penalty -70
    SBHMM EX = python3 driver.py --test_type standard --users Naoki --train_iters 25 50 --sbhmm_iters 25 50 --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --parallel_jobs 4 --hmm_insertion_penalty -70 --sbhmm_insertion_penalty -115
"""
"""Main file used to prepare training data, train, and test HMMs.
    HMM EX = python3 driver.py --test_type standard --train_iters 25 50 75 100 --users Prerna Linda | 
    HMM CV = python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 --users 02-22-20_Prerna_Android 04-29-20_Linda_Android 07-24-20_Matthew_4K --cross_val_method stratified --n_splits 10 --cv_parallel --parallel_jobs 10  --hmm_insertion_penalty -80
    SBHMM EX = python3 driver.py --test_type standard --train_iters 25 50 75 --sbhmm_iters 25 50 75 --users Prerna Linda --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --parallel_jobs 4 --hmm_insertion_penalty -70 --sbhmm_insertion_penalty -115 --neighbors 70
    SBHMM CV = python3 driver.py --test_type cross_val --train_iters 25 50 75 --sbhmm_iters 25 50 75 --users Linda Prerna --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --parallel_jobs 4 --hmm_insertion_penalty -85 --sbhmm_insertion_penalty -85 --neighbors 70 --cross_val_method kfold --n_splits 10 --beam_threshold 2000.0
    SBHMM CV Parallel = python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 --sbhmm_iters 25 50 75 100 --users Ishan Matthew David --train_sbhmm --sbhmm_cycles 1  --include_word_level_states --include_word_level_states --parallel_classifier_training --hmm_insertion_penalty -80 --sbhmm_insertion_penalty -80 --neighbors 73 --cross_val_method stratified --n_splits 5 --beam_threshold 3000.0 --cv_parallel --parallel_jobs 10
    Prepare Data = python3 driver.py --test_type none --prepare_data --users Matthew_4 Ishan_4 David_4
    Old SBHMM CV = python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 --sbhmm_iters 25 50 75 100 --users Ishan Matthew David --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --parallel_classifier_training --hmm_insertion_penalty -80 --sbhmm_insertion_penalty -80 --cross_val_method stratified --n_splits 5 --beam_threshold 3000.0 --cv_parallel --parallel_jobs 5 --multiple_classifiers
    SBHMM CV Parallel User Independent =  python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 240 --sbhmm_iters 25 50 75 100 --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --hmm_insertion_penalty 150 --sbhmm_insertion_penalty 150 --neighbors 200 --cross_val_method leave_one_user_out --n_splits 5 --beam_threshold 50000.0 --cv_parallel --parallel_jobs 7 --users Linda_4 Kanksha_4 Thad_4 Matthew_4 Prerna_4 David_4 Ishan_4
    HMM CV Parallel User Independent =  python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 240 --hmm_insertion_penalty 10 --cross_val_method leave_one_user_out --n_splits 5 --beam_threshold 50000.0 --cv_parallel --parallel_jobs 4
"""

"""Verification commands
    HMM Standard (Dry run) = python3 driver.py --test_type standard --train_iters 10 20 --users Matthew --method verification
    HMM CV = python3 driver.py --test_type cross_val --train_iters 10 --users 07-24-20_Matthew_4KDepth 11-08-20_Colby_4KDepth 11-08-20_Ishan_4KDepth --cross_val_method leave_one_user_out --n_splits 10 --cv_parallel --parallel_jobs 3  --hmm_insertion_penalty -80 --method verification --verification_method logistic_regression
"""
import sys
import glob
import argparse
import os
import shutil
import sys
import random
import numpy as np
import tqdm
import pickle

from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneGroupOut, train_test_split)

sys.path.insert(0, '../../')
from src.prepare_data import prepare_data
from src.train import create_data_lists, train, trainSBHMM, get_logistic_regressor, get_neural_net_classifier
from src.adapt import pua, adapt
from src.utils import get_results, save_results, load_json, get_arg_groups, get_hresults_data
from src.test import test, testSBHMM, verify_simple, return_average_ll_per_sign, return_ll_per_correct_and_one_off_sign, verify_zahoor, verify_classifier
from src.utils.file_util import *
from src.utils.cross_val_util import *
from joblib import Parallel, delayed
from statistics import mean
from src.data_augmentation import DataAugmentation

def main():
    
    parser = argparse.ArgumentParser()
    ############################## ARGUMENTS #####################################
    # Important
    parser.add_argument('--prepare_data', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_results_file', type=str,
                        default='all_results.json')
    # ASK FOR DATASET PATHS @TODO

    # Arguments for create_data_lists()
    parser.add_argument('--test_type', type=str, default='test_on_train',
                        choices=['none', 'test_on_train', 'cross_val', 'standard', 'progressive_user_adaptive', 'adaptive_htk'])
    parser.add_argument('--users', nargs='*', default=None)
    parser.add_argument('--cross_val_method', default='kfold', choices=['kfold',
                                                  'leave_one_phrase_out',
                                                  'stratified',
                                                  'leave_one_user_out',
                                                  'user_dependent',
                                                  ])
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--cv_parallel', action='store_true')
    parser.add_argument('--parallel_jobs', default=4, type=int)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--phrase_len', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=42) #The answer to life, the universe and everything

    # Arguments for training
    parser.add_argument('--train_iters', nargs='*', type=int, default=[2,3,4])
    parser.add_argument('--hmm_insertion_penalty', default=-10)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--variance', type=float, default=0.00001)
    parser.add_argument('--transition_prob', type=float, default=0.6)

    # Arguments for SBHMM
    parser.add_argument('--train_sbhmm', action='store_true')    
    parser.add_argument('--sbhmm_iters', nargs='*', type=int, default=[2,3,4])
    parser.add_argument('--include_word_position', action='store_true')
    parser.add_argument('--include_word_level_states', action='store_true')
    parser.add_argument('--sbhmm_insertion_penalty', default=-10)
    parser.add_argument('--classifier', type=str, default='knn',
                        choices=['knn', 'adaboost'])
    parser.add_argument('--neighbors', default=50)
    parser.add_argument('--sbhmm_cycles', type=int, default=1)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--pca_components', type=int, default=32)
    parser.add_argument('--multiple_classifiers', action='store_true')
    parser.add_argument('--parallel_classifier_training', action='store_true')
    parser.add_argument('--beam_threshold', default=100000000.0)

    # Arguments for testing
    parser.add_argument('--start', type=int, default=-2)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--method', default='recognition', 
                        choices=['recognition', 'verification'])
    parser.add_argument('--acceptance_threshold', default=-150)
    parser.add_argument('--verification_method', default='zahoor', 
                        choices=['zahoor', 'logistic_regression', 'neural_net'])

    # Arguments for adaptation
    parser.add_argument('--adapt_iters', nargs='*', type=int, default=[2,3,4])
    parser.add_argument('--hmm_num', nargs='*', type=int, default=4)
    parser.add_argument('--new_users', nargs='*', default=None)
    
    # Arguments for data augmentation
    '''
    1) Tell if data aug should be done
    2) Ask for rotationsX, Y
    3) Ask for dataset folder
    4) Ask for bodypix model
    6) autoTranslate=True
    7) pointForAutoTranslate=(3840 // 2, 2160 //2)
    '''
    parser.add_argument('--data_augmentation', type=bool, default=False)
    parser.add_argument('--rotationsX', type=str, default="-10_-5_0_5_10")
    parser.add_argument('--rotationsY', type=str, default="-10_-5_0_5_10")
    parser.add_argument('--bodypix_model', type=int, default=1)
    parser.add_argument('--autoTranslate', type=bool, default=True)
    parser.add_argument('--pointForAutoTranslateX', type=int, default=3840 // 2)
    parser.add_argument('--pointForAutoTranslateY', type=int, default=2160 // 2)

    args = parser.parse_args()
    ########################################################################################
    
    #if args.users: args.users = [user.capitalize() for user in args.users]
    
    # SECTION TO ADD DATA AUG TO PIPELINE
    if args.data_augmentation:
        try:
            rotationsX = [int(x) for x in args.rotationsX.split('_')]
            rotationsY = [int(y) for y in args.rotationsY.split('_')]
        except:
            raise ValueError('rotationsX and rotationsY must be integers separated by "_"')
        # @Ishan specify the dataset folder and output path of where all the augmented videos should be saved
        # The Data augmentation object does all the bounds checking, so you dont have to worry about that
        da = DataAugmentation(datasetFolder='INSERT', outputPath='INSERT', rotationsX=rotationsX, rotationsY=rotationsY, useBodyPixModel=args.bodypix_model, pointForAutoTranslate=(args.pointForAutoTranslateX, args.pointForAutoTranslateY), autoTranslate=args.autoTranslate)
        # @Ishan listOfVideos is a list of strings of the locations of all the augmented videos
        listOfVideos = da.createDataAugmentedVideos()
        pass
    
    cross_val_methods = {'kfold': (KFold, True),
                         'leave_one_phrase_out': (LeaveOneGroupOut(), True),
                         'stratified': (StratifiedKFold, True),
                         'leave_one_user_out': (LeaveOneGroupOut(), True),
                         'user_dependent': (None, False),
                         }
    cvm = args.cross_val_method
    cross_val_method, use_groups = cross_val_methods[args.cross_val_method]

    features_config = load_json('configs/features.json')
    all_results = {'features': features_config['selected_features'],
                   'average': {}}
                   
    if args.train_sbhmm:
        hresults_file = f'hresults/res_hmm{args.sbhmm_iters[-1]-1}.txt'
    else:
        hresults_file = f'hresults/res_hmm{args.train_iters[-1]-1}.txt'

    if args.prepare_data and not args.test_type == 'progressive_user_adaptive' and not args.test_type == 'adaptive_htk':
        # this will include users in verification
        print(args.users)
        prepare_data(features_config, args.users)

    if args.test_type == 'none':
        sys.exit()

    elif args.test_type == 'test_on_train':
        
        if not args.users:
            htk_filepaths = glob.glob('data/htk/*.htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        create_data_lists(htk_filepaths, htk_filepaths, args.phrase_len)
        
        if args.train_sbhmm:
            classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                        args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                        args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                        args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold)
            testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty,
                    args.parallel_jobs, args.parallel_classifier_training)
        else:
            train(args.train_iters, args.mean, args.variance, args.transition_prob)
            if args.method == "recognition":
                test(args.start, args.end, args.method, args.hmm_insertion_penalty)
            elif args.method == "verification":
                positive, negative, false_positive, false_negative = verify_simple(args.end, args.insertion_penalty, 
                                                                    args.acceptance_threshold, args.beam_threshold)
        
        if args.method == "recognition":
            all_results['fold_0'] = get_results(hresults_file)
            all_results['average']['error'] = all_results['fold_0']['error']
            all_results['average']['sentence_error'] = all_results['fold_0']['sentence_error']

            print('Test on Train Results')
        
        if args.method == "verification":
            all_results['average']['positive'] = positive
            all_results['average']['negative'] = negative
            all_results['average']['false_positive'] = false_positive
            all_results['average']['false_negative'] = false_negative

            print('Test on Train Results')
    
    elif args.test_type == 'cross_val' and args.cv_parallel:
        print("You have invoked parallel cross validation. Be prepared for dancing progress bars!")

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*.htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [filepath.split('/')[-1].split(".")[0] + " " + ' '.join(filepath.split('/')[-1].split(".")[1].split("_"))
            for filepath
            in htk_filepaths]
        

        if cvm == 'kfold' or cvm == 'stratified':
            unique_phrases = set(phrases)
            print(len(unique_phrases), len(phrases))
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]      
            cross_val = cross_val_method(n_splits=args.n_splits)
        elif cvm == 'leave_one_phrase_out':
            unique_phrases = set(phrases)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]
            cross_val = cross_val_method
        elif cvm == 'leave_one_user_out':
            users = [get_user(filepath) for filepath in htk_filepaths]
            unique_users = list(set(users))
            unique_users.sort()
            print(unique_users)
            group_map = {user: i for i, user in enumerate(unique_users)}
            groups = [group_map[user] for user in users]            
            cross_val = cross_val_method
        elif cvm == 'user_dependent':
            users = [get_user(filepath) for filepath in htk_filepaths]
            unique_users = list(set(users))
            unique_users.sort()
            print(unique_users)
        
        if cvm == 'user_dependent':
            splits = returnUserDependentSplits(unique_users, htk_filepaths, args.test_size)
        elif use_groups:
            splits = list(cross_val.split(htk_filepaths, phrases, groups))
        else:
            splits = list(cross_val.split(htk_filepaths, phrases))
        if args.method == "recognition":         
            stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                            (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                            for currFold in range(len(splits)))
            all_results['average']['error'] = mean([i[0] for i in stats])
            all_results['average']['sentence_error'] = mean([i[1] for i in stats])
            all_results['average']['insertions'] = mean([i[2] for i in stats])
            all_results['average']['deletions'] = mean([i[3] for i in stats])

        elif args.method == "verification":
            stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValVerificationFold)
                            (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                            for currFold in range(len(splits)))
            all_results['average']['positive'] = mean(i[0] for i in stats)
            all_results['average']['negative'] = mean(i[1] for i in stats)
            all_results['average']['false_positive'] = mean(i[2] for i in stats)
            all_results['average']['false_negative'] = mean(i[3] for i in stats)
        
        print(stats)

    elif args.test_type == 'cross_val':
        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [' '.join(filepath.split('.')[1].split("_"))
            for filepath
            in htk_filepaths]
        
        users = [get_user(filepath) for filepath in htk_filepaths]     

        if cvm == 'kfold' or cvm == 'stratified':
            unique_phrases = set(phrases)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]      
            cross_val = cross_val_method(n_splits=args.n_splits)
        elif cvm == 'leave_one_phrase_out':
            unique_phrases = set(phrases)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]
            cross_val = cross_val_method
        elif cvm == 'leave_one_user_out':
            unique_users = set(users)
            group_map = {user: i for i, user in enumerate(unique_users)}
            groups = [group_map[user] for user in users]            
            cross_val = cross_val_method
        elif cvm == 'user_dependent':
            users = [get_user(filepath) for filepath in htk_filepaths]
            unique_users = list(set(users))
            unique_users.sort()
            print(unique_users)
        
        if cvm == 'user_dependent':
            splits = returnUserDependentSplits(unique_users, htk_filepaths, args.test_size)
        elif use_groups:
            splits = list(cross_val.split(htk_filepaths, phrases, groups))
        else:
            splits = list(cross_val.split(htk_filepaths, phrases))

        for i, (train_index, test_index) in enumerate(splits):

            print(f'Current split = {i}')
            
            train_data = np.array(htk_filepaths)[train_index]
            test_data = np.array(htk_filepaths)[test_index]

            phrase = np.array(phrases)[test_index][0]
            phrase_len = len(phrase.split(' '))
            phrase_count = len(test_data)
            word_count = phrase_len * phrase_count
            word_counts.append(word_count)
            phrase_counts.append(phrase_count)
            create_data_lists(train_data, test_data, args.phrase_len)

            if args.train_sbhmm:
                classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                        args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                        args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                        args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold)
                testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty, 
                        args.parallel_jobs, args.parallel_classifier_training)
            else:
                train(args.train_iters, args.mean, args.variance, args.transition_prob)
                test(args.start, args.end, args.method, args.hmm_insertion_penalty)
            
            results = get_results(hresults_file)
            all_results[f'fold_{i}'] = results
            all_results[f'fold_{i}']['phrase'] = phrase
            all_results[f'fold_{i}']['phrase_count'] = phrase_count

            print(f'Current Word Error: {results["error"]}')
            print(f'Current Sentence Error: {results["sentence_error"]}')

            substitutions += (word_count * results['substitutions'] / 100)
            deletions += (word_count * results['deletions'] / 100)
            insertions += (word_count * results['insertions'] / 100)
            sentence_errors += (phrase_count * results['sentence_error'] / 100)

        total_words = sum(word_counts)
        total_phrases = sum(phrase_counts)
        total_errors = substitutions + deletions + insertions
        mean_error = (total_errors / total_words) * 100
        mean_error = np.round(mean_error, 4)
        mean_sentence_error = (sentence_errors / total_phrases) * 100
        mean_sentence_error = np.round(mean_sentence_error, 2)

        all_results['average']['error'] = mean_error
        all_results['average']['sentence_error'] = mean_sentence_error

        print('Cross-Validation Results')

    elif args.test_type == 'standard':

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))
        
        phrases = [' '.join(filepath.split('.')[1].split('_'))
            for filepath
            in htk_filepaths]
        train_data, test_data, _, _ = train_test_split(
            htk_filepaths, phrases, test_size=args.test_size,
            random_state=args.random_state)

        create_data_lists(train_data, test_data, args.phrase_len)
        if args.train_sbhmm:
            classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                        args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                        args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                        args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold)
            testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty, 
                    args.parallel_jobs, args.parallel_classifier_training)
        else:
            train(args.train_iters, args.mean, args.variance, args.transition_prob)
            if args.method == "recognition":
                test(args.start, args.end, args.method, args.hmm_insertion_penalty)
            elif args.method == "verification":
                positive, negative, false_positive, false_negative = verify_simple(args.end, args.hmm_insertion_penalty, args.acceptance_threshold, args.beam_threshold)
        
        if args.method == "recognition":
            all_results['fold_0'] = get_results(hresults_file)
            all_results['average']['error'] = all_results['fold_0']['error']
            all_results['average']['sentence_error'] = all_results['fold_0']['sentence_error']

            print('Test on Train Results')
        
        if args.method == "verification":
            all_results['average']['positive'] = positive
            all_results['average']['negative'] = negative
            all_results['average']['false_positive'] = false_positive
            all_results['average']['false_negative'] = false_negative

            print('Standard Train/Test Split Results')

    elif args.test_type == 'progressive_user_adaptive':
        all_results, stats = pua(args, features_config, all_results)

    elif args.test_type == 'adaptive_htk':

        #prepare_data(features_config, args.users)
        # if not args.users:
        #     htk_filepaths = glob.glob('data/htk/*htk')
        # else:
        #     htk_filepaths = []
        #     for user in args.users:
        #         htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))
        
        # phrases = [' '.join(filepath.split('.')[1].split('_'))
        #     for filepath
        #     in htk_filepaths]
        # train_data, test_data, _, _ = train_test_split(
        #     htk_filepaths, phrases, test_size=args.test_size,
        #     random_state=args.random_state)

        # create_data_lists(train_data, test_data, args.phrase_len)
        # train(args.train_iters, args.mean, args.variance, args.transition_prob)

        # prepare_data(features_config, args.new_users)
        # htk_filepaths = []
        # for user in args.new_users:
        #     htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))
        
        # phrases = [' '.join(filepath.split('.')[1].split('_'))
        #     for filepath
        #     in htk_filepaths]
        # train_data, test_data, _, _ = train_test_split(
        #     htk_filepaths, phrases, test_size=args.test_size,
        #     random_state=args.random_state)

        # create_data_lists(train_data, test_data, args.phrase_len)
        adapt(args.adapt_iters, args.train_iters[-1])
        test(args.start, args.end, args.method, args.hmm_insertion_penalty)
        hresults_file = f'hresults/res_hmm5.txt'
        all_results['fold_0'] = get_results(hresults_file)
        all_results['average']['error'] = all_results['fold_0']['error']
        all_results['average']['sentence_error'] = all_results['fold_0']['sentence_error']

    if args.method == "recognition":
        
        print(f'Average Error: {all_results["average"]["error"]}')
        print(f'Average Sentence Error: {all_results["average"]["sentence_error"]}')
        
        if args.test_type == 'cross_val' and args.cv_parallel:
            print(f'Average Insertions: {all_results["average"]["insertions"]}')
            print(f'Average Deletions: {all_results["average"]["deletions"]}')
    
    if args.method == "verification":

        print(f'Positive Pairs: {all_results["average"]["positive"]}')
        print(f'Negative Pairs: {all_results["average"]["negative"]}')
        print(f'False Positive Pairs: {all_results["average"]["false_positive"]}')
        print(f'False Negative Pairs: {all_results["average"]["false_negative"]}')
        percent_correct = (all_results["average"]["positive"] + all_results["average"]["negative"]) \
                            /(all_results["average"]["positive"] + all_results["average"]["negative"] + all_results["average"]["false_positive"] + all_results["average"]["false_negative"])
        print(f'Correct %: {percent_correct*100}')
        print(f'Precision %: {100*all_results["average"]["positive"]/(all_results["average"]["positive"] + all_results["average"]["false_positive"])}')
        print(f'Recall %: {100*all_results["average"]["positive"]/(all_results["average"]["positive"] + all_results["average"]["false_negative"])}')

    # Loads data as new run into pickle
    if args.save_results:
        save_results(all_results, args.save_results_file, 'a')
