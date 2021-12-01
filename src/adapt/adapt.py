import os
from tqdm import tqdm

from adapt.gen_baseclass_regtree import gen_baseclass_regtree

def adapt(adapt_iters, nstates, fold, hmm_num):
    # Create base class and regression tree
    print("Making base class and regression tree")
    gen_baseclass_regtree(fold, hmm_num)

    # Create additional hmm dirs
    for i in range(hmm_num, hmm_num + adapt_iters[-1] + 1):
        hmm_dir = os.path.join('models', f'{fold}hmm{i}')
        if not os.path.exists(hmm_dir):
            os.makedirs(hmm_dir)

    # Transform models
    for i, n_iters in enumerate(adapt_iters):

        for iter_ in tqdm(range(start, n_iters)):
            iter_num = hmm_num + iter_
            n_iters_num = hmm_num + n_iters

            # transformation command
            # can use -k to set input transform
            # -j performs incremental MLLR unsupervised adaptation
            HVite_command = (f'HVite -A -I all_labels.mlf -K transforms/ -H'
                             f'models/{fold}hmm{iter_num}/newMacros' 
                             f'-j 1 -m -S lists/{fold}test.data -i '
                             f'$results -w wordNet.txt -s 25 dict wordList')
            # copy files over to next folder
            # see what running this does on random hmm folder
            
            # alternative: perform supervised adaptation and call herest with regtree
            # HERest_command = (f'HERest -A -c 500.0 -v 0.0005 -A -H '
            #                 f'models/{fold}hmm{iter_num}/newMacros -I all_labels.mlf -M '
            #                 f'models/{fold}hmm{iter_num+1} -S lists/{fold}train.data -T 1 wordList '
            #                 f'>> logs/{fold}train.log')
            os.system(HVite_command)
        print('Adaptation Step Complete')

        # Copy model setup over to next hmm iteration folder
        if n_iters != adapt_iters[-1]:
            print(f'Running HHed Iteration: {hmm_num + n_iters_num}...')
            HHed_command = (f'HHEd -A -H models/{fold}hmm{n_iters_num-1}/newMacros -M '
                            f'models/{fold}hmm{n_iters_num} configs/hhed{i}.conf '
                            f'wordList')
            os.system(HHed_command)
            print('HHed Complete')
            start = n_iters

    cmd = 'HParse -A -T 1 grammar.txt wordNet.txt'
    os.system(cmd)
    

    # Give metrics on models during transformation

    # Output new model
    

