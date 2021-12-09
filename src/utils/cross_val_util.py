from .file_util import *
from .get_results import get_results
from src.train import *
from src.test import *

import os
import pickle


def crossValVerificationFold(train_data: list, test_data: list, args: object, fold: int):
    print(f"Current split = {str(fold)}. Current Test data Size = {len(test_data)}")
    ogDataFolder = "data"
    currDataFolder = os.path.join("data", str(fold))
    trainFiles = [i.split("/")[-1].replace(".htk", "") for i in train_data]
    testFiles = [i.split("/")[-1].replace(".htk", "") for i in test_data]
    allFiles = trainFiles + testFiles

    copyFiles(allFiles, os.path.join(currDataFolder, "ark"), os.path.join(ogDataFolder, "ark"), ".ark")
    copyFiles(allFiles, os.path.join(currDataFolder, "htk"), os.path.join(ogDataFolder, "htk"), ".htk")
    test_user = get_user(testFiles[0])
    users_in_train = set([get_user(filepath) for filepath in trainFiles])
    average_ll_per_sign = {}
    for user in users_in_train:
        curr_train_files = []
        curr_test_files = []
        for filepath in trainFiles:
            if get_user(filepath) != user:
                curr_train_files.append(filepath)
            else:
                curr_test_files.append(filepath)
        
        create_data_lists([os.path.join(currDataFolder, "htk", i+".htk") for i in curr_train_files], [
                    os.path.join(currDataFolder, "htk", i+".htk") for i in curr_test_files], args.phrase_len, fold)

        train(args.train_iters, args.mean, args.variance, args.transition_prob, fold=os.path.join(str(fold), ""))
        if args.verification_method == "zahoor":
            curr_average_ll_sign = return_average_ll_per_sign(args.end, args.hmm_insertion_penalty, 
                                                            args.beam_threshold, fold=os.path.join(str(fold), ""))
            if len(average_ll_per_sign) == 0:
                average_ll_per_sign = curr_average_ll_sign
            else:
                for sign in average_ll_per_sign:
                    average_ll_per_sign[sign] = np.concatenate((average_ll_per_sign[sign], curr_average_ll_sign[sign]), axis=None)

        elif args.verification_method == "logistic_regression" or args.verification_method == "neural_net":
            curr_average_ll_sign = return_ll_per_correct_and_one_off_sign(args.end, args.hmm_insertion_penalty, args.verification_method == "logistic_regression",
                                                            args.beam_threshold, fold=os.path.join(str(fold), ""))
            for data_set in curr_average_ll_sign:
                if data_set not in average_ll_per_sign:
                    average_ll_per_sign[data_set] = {}
                for sign in curr_average_ll_sign[data_set]:
                    if sign in average_ll_per_sign:
                        average_ll_per_sign[data_set][sign] = np.concatenate((average_ll_per_sign[data_set][sign], curr_average_ll_sign[data_set][sign]), axis=0)
                    else:
                        average_ll_per_sign[data_set][sign] = np.array(curr_average_ll_sign[data_set][sign])
            print(f"Signs in correct set = {str(len(average_ll_per_sign['correct']))} and signs in incorrect set = {str(len(average_ll_per_sign['incorrect']))}")
        else:
            raise Exception("Please select correct verification method")
    
    # Save user independent log likelihoods
    pickle.dump(average_ll_per_sign, open(os.path.join(currDataFolder, f"{test_user}_UI_loglikelihoods.pkl"), "wb"))
    classifier = {}
    if args.verification_method == "zahoor":
        for sign in average_ll_per_sign:
            classifier[sign] = [np.mean(average_ll_per_sign[sign]), np.std(average_ll_per_sign[sign])]
    elif args.verification_method == "logistic_regression":
        print("Training logistic regression classifier for each sign")
        for sign in tqdm.tqdm(average_ll_per_sign["correct"]):
            classifier[sign] = get_logisitc_regressor(average_ll_per_sign["correct"][sign], average_ll_per_sign["incorrect"][sign], args.random_state)
    elif args.verification_method == "neural_net":
        print("Training neural net classifier for each sign")
        for sign in tqdm.tqdm(average_ll_per_sign["correct"]):
            classifier[sign] = get_neural_net_classifier(average_ll_per_sign["correct"][sign], average_ll_per_sign["incorrect"][sign], args.random_state)
    
    else:
        raise Exception("Please select correct verification method")

    
    create_data_lists([os.path.join(currDataFolder, "htk", i+".htk") for i in trainFiles], [
                    os.path.join(currDataFolder, "htk", i+".htk") for i in testFiles], args.phrase_len, fold)
    train(args.train_iters, args.mean, args.variance, args.transition_prob, fold=os.path.join(str(fold), ""))
    if args.verification_method == "zahoor":
        positive, negative, false_positive, false_negative, test_log_likelihoods = verify_zahoor(args.end, args.hmm_insertion_penalty, classifier, 
                                                                        args.beam_threshold, fold=os.path.join(str(fold), ""))
    elif args.verification_method == "logistic_regression":
        positive, negative, false_positive, false_negative, test_log_likelihoods = verify_classifier(args.end, args.hmm_insertion_penalty, classifier, True, 
                                                                        args.beam_threshold, fold=os.path.join(str(fold), ""))
    elif args.verification_method == "neural_net":
        positive, negative, false_positive, false_negative, test_log_likelihoods = verify_classifier(args.end, args.hmm_insertion_penalty, classifier, False,
                                                                        args.beam_threshold, fold=os.path.join(str(fold), ""))
    else:
        raise Exception("Please select correct verification method")

    pickle.dump(test_log_likelihoods, open(os.path.join(currDataFolder, f"{test_user}_test_split_loglikelihoods.pkl"), "wb"))

    print(f'Current Positive Rate: {positive/(positive + false_negative)}')
    print(f'Current Negative Rate: {negative/(negative + false_positive)}')
    print(f'Current False Positive Rate: {false_positive/(negative + false_positive)}')
    print(f'Current False Negative Rate: {false_negative/(positive + false_negative)}')

    return positive, negative, false_positive, false_negative


def crossValFold(train_data: list, test_data: list, args: object, fold: int):
    train_data = np.array(train_data)
    np.random.seed(args.random_state)
    np.random.shuffle(train_data)
    print(f"Current split = {str(fold)}. Current Test data Size = {len(test_data)}")
    ogDataFolder = "data"
    currDataFolder = os.path.join("data", str(fold))
    trainFiles = [i.split("/")[-1].replace(".htk", "") for i in train_data]
    testFiles = [i.split("/")[-1].replace(".htk", "") for i in test_data]
    allFiles = trainFiles + testFiles

    copyFiles(allFiles, os.path.join(currDataFolder, "ark"), os.path.join(ogDataFolder, "ark"), ".ark")
    copyFiles(allFiles, os.path.join(currDataFolder, "htk"), os.path.join(ogDataFolder, "htk"), ".htk")
    create_data_lists([os.path.join(currDataFolder, "htk", i+".htk") for i in trainFiles], [
                    os.path.join(currDataFolder, "htk", i+".htk") for i in testFiles], args.phrase_len, fold)
    
    if args.train_sbhmm:
        classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold, os.path.join(str(fold), ""))
        testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty, 
                args.parallel_jobs, args.parallel_classifier_training, os.path.join(str(fold), ""))
    else:
        train(args.train_iters, args.mean, args.variance, args.transition_prob, fold=os.path.join(str(fold), ""))
        test(args.start, args.end, args.method, args.hmm_insertion_penalty, fold=os.path.join(str(fold), ""))

    if args.train_sbhmm:
        hresults_file = f'hresults/{os.path.join(str(fold), "")}res_hmm{args.sbhmm_iters[-1]-1}.txt'
    else:
        hresults_file = f'hresults/{os.path.join(str(fold), "")}res_hmm{args.train_iters[-1]-1}.txt'    

    results = get_results(hresults_file)

    print(f'Current Word Error: {results["error"]}')
    print(f'Current Sentence Error: {results["sentence_error"]}')
    print(f'Current Insertion Error: {results["insertions"]}')
    print(f'Current Deletions Error: {results["deletions"]}')

    test(-1, -1, "alignment", args.hmm_insertion_penalty, beam_threshold=args.beam_threshold, fold=os.path.join(str(fold), ""))

    return [results['error'], results['sentence_error'], results['insertions'], results['deletions']]