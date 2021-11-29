from get_confusion_matrix import get_confusion_matrix
import os

def sequence_results(hresults_dir, phrases, multi_user=True):
    data_dict = {}

    # Iterate over users
    hresults_dir_original = hresults_dir
    for user in os.listdir(hresults_dir):

        # If multi user, iterate one directory further for users
        hresults_dir = hresults_dir_original
        hresults_dir_original = hresults_dir
        if multi_user:
            hresults_dir = os.path.join(hresults_dir, user)
            print(hresults_dir)
            fp_list = os.listdir(hresults_dir)
        else:
            fp_list = [user]

        for fp in fp_list:

            # Try to extract confusion matrix data
            try:
                user_results = get_confusion_matrix(os.path.join(hresults_dir, fp))
            except:
                print("No data found for user {}".format(user))
                continue

            for phrase in phrases:
                words = phrase.split('_')
                phrase_len = len(words)
                accuracy = 0

                # Get average of word level accuracy
                # Can look at insertions/deletions specifically
                for word in words:
                    word_abrev1 = [k for k in user_results['matrix'].keys() if word.startswith(k)][0]
                    word_abrev2 = [k for k in user_results['matrix'][word_abrev1].keys() if word.startswith(k)][0]
                    accuracy += user_results['matrix'][word_abrev1][word_abrev2] / sum(user_results['matrix'][word_abrev1].values())
                if user_results['user'] in data_dict:
                    data_dict[user_results['user']][phrase] = accuracy / phrase_len
                else:
                    data_dict[user_results['user']] = {phrase: accuracy / phrase_len}

    print(data_dict)

    # Can implement some sorting function over here
    return data_dict


hd = '/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/SBHMM-HTK/SequentialClassification/main/projects/Mediapipe/hresults/'
p = ['in', 'above']
sequence_results(hd, p)