import numpy as np
import pandas as pd
from get_data import read_ark_file, read_htk_file, mlf_to_dict
import os
import sys
from matplotlib import pyplot as plt
sys.path.append('../dimensionality_reduction')
from lda import lda, lda2

def lda_confused_words(confused_words_list, sample_size, ftype, data_dir, mlf_file, hlist_path=None):
    label_dict = {}
    df = pd.DataFrame(columns=['label', 'data'])
    boundaries = mlf_to_dict(mlf_file)

    # Iterate over confused words
    for label, word in enumerate(confused_words_list):
        label_dict[label + 1] = word

        # Sample from all phrases containing confused words
        filenames = os.listdir(data_dir)
        if label == 0:
            word_filenames = np.array([name for name in filenames if 'Colby' in name and 'bed' in name])
        else:
            word_filenames = np.array([name for name in filenames if not 'Colby' in name and 'bed' in name])
        word_filenames_sample = np.random.choice(word_filenames, size=sample_size)

        # Iterate over sampled files for specific word
        count = 0
        for fname in word_filenames_sample:
            try:
                phrase = fname.replace('.{}'.format(ftype), '')
                phrase_data = read_ark_file(os.path.join(data_dir, fname)) if ftype == 'ark' else read_htk_file(hlist_path, os.path.join(data_dir, fname))
                mult = float(list(boundaries[phrase].items())[-1][1][0][2]) / len(phrase_data)
                word_data = phrase_data[int(boundaries[phrase][word][0][1]/mult) : int(boundaries[phrase][word][0][2]/mult)]
                for frame in word_data:
                    df = df.append({'label': label + 1, 'data': np.array(frame)}, ignore_index=True)
                    if label == 0:
                        plt.plot(frame[0], "r+")
                    else:
                        plt.plot(frame[1], "bo")
                count += 1
            except:
                print('skip')
            if count > 100:
                break
    #print(df)
    plt.show()
    #lda2(df)
    lda(df)

    return df

if __name__=='__main__':
    results = './data/results'
    for mlf in os.listdir(results):
        if mlf.endswith('mlf'):
            lda_confused_words(['bed', 'bed'], 100, 'htk', './data/htk', os.path.join(results, mlf),'./data/HList')
