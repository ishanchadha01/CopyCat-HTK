import os
import shutil

def gen_baseclass_regtree(hmm_num):
    """Adapts the HMM using HTK. Calls HCompV, HRest, HERest, HHEd, and
    HParse. Configuration files for prototypes and increasing mixtures
    are found in configs/. 

    Parameters
    ----------
    train_args : Namespace
        Argument group defined in train_cli() and split from main
        parser.
    """

    if os.path.exists(f'tree/'):
        shutil.rmtree(f'tree/')

    # Make tree
    # Always pass -A -D -V -T 1 for debugging purposes
    # -B output HMM definitions in binary format
    # -H newmacros_file the model file to adapt
    # -w all_defs master macros file with all macros and model definitions
    os.system(f'HHEd -A -D -V -T 1 -B -H models/hmm{hmm_num} -w delete.me tree/regtree.hed \
        tree/mlists/treeg.list > logs/regtree.log')