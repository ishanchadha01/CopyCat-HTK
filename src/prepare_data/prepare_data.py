"""Prepares training data. Creates .ark files, .htk files, wordList,
dict, grammar, and all_labels.mlf.

Methods
-------
prepare_data
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

from . import create_ark_files, create_htk_files
from .generate_text_files import generate_text_files

def prepare_data(features_config: dict, users: list, num_jobs: int, phrase_len:list=[3,4,5], prediction_len:list=[3,4,5]) -> None:

    """Prepares training data. Creates .ark files, .htk files, wordList,
    dict, grammar, and all_labels.mlf.

    Parameters
    ----------
    features_config : dict
        A dictionary defining which features to use when creating the 
        data files.
    """
    create_ark_files(features_config, users, num_jobs, phrase_len, verbose=False, is_select_features=True, use_optical_flow=False)
    print('.ark files created')

    print('Creating .htk files')
    create_htk_files()
    print('.htk files created')

    print('Creating .txt files')
    generate_text_files(prediction_len)
    print('.txt files created')
