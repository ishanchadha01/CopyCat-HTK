"""Creates .ark files needed as intermediate step to creating .htk files

Methods
-------
_create_ark_file
create_ark_files
"""
import os
import glob
from pyexpat import features
import shutil
from tqdm import tqdm
from p_tqdm import p_map
import pandas as pd
from functools import partial

from .feature_selection import select_features
from .interpolate_feature_data import interpolate_feature_data
from .feature_extraction_kinect import feature_extraction_kinect
from .feature_extraction_alphapose import feature_extraction_alphapose
from scipy import stats
import numpy as np

def _select_features_func(features_filepath, verbose, features_config, is_select_features, use_optical_flow, ark_dir):
    if verbose:
        print(features_filepath)

    features_filename = features_filepath.split('/')[-1]
    features_extension = features_filename.split('.')[-1]
    features_df = None

    ark_filename = features_filename.replace(features_extension, 'ark')
    ark_filepath = os.path.join(ark_dir, ark_filename)
    title = ark_filename.replace('.ark', "")

    features_df = select_features(features_filepath, features_config['selected_features'], center_on_nose=True, scale=100, square=True,
                                    drop_na=True, do_interpolate=True, use_optical_flow=use_optical_flow)

    return features_df, ark_filepath, title


def _create_ark_file(df: pd.DataFrame, ark_filepath: str, title: str) -> None:
    """Creates a single .ark file

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing selected feature.

    ark_filepath : str
        File path at which to save .ark file.

    title : str
        Title containing label needed as header of .ark file.
    """

    with open(ark_filepath, 'w') as out:
        out.write('{} [ '.format(title))

    df.to_csv(ark_filepath, mode='a', header=False, index=False, sep=' ')
    
    with open(ark_filepath, 'a') as out:
        out.write(']')


def create_ark_files(features_config: dict, users: list, num_jobs: int, phrase_len: list, verbose: bool,
                     is_select_features: bool, use_optical_flow: bool) -> None:
    """Creates .ark files needed as intermediate step to creating .htk
    files

    Parameters
    ----------
    features_config : dict
        Contains features_dir and features_to_extract

    verbose : bool, optional, by default False
        Whether to print output during process.
    """

    ark_dir = os.path.join('data', 'ark')

    if os.path.exists(ark_dir):
        shutil.rmtree(ark_dir)

    os.makedirs(ark_dir)
    if not users:
        features_filepaths = glob.glob(os.path.join(
            features_config['features_dir'], '**', '*.data'), recursive=True)
        features_filepaths.extend(glob.glob(os.path.join(
            features_config['features_dir'], '**', '*.json'), recursive=True))
    else:
        features_filepaths = []
        for user in users:
            features_filepaths.extend(glob.glob(os.path.join(
                features_config['features_dir'], '*{}_*'.format(user), '**', '*.data'), recursive=True))
            features_filepaths.extend(glob.glob(os.path.join(
                features_config['features_dir'], '*{}_*'.format(user), '**', '*.json'), recursive=True))

    features_filepaths = list(filter(lambda x: len(os.path.basename(x).split('.')[1].split('_')) in phrase_len, features_filepaths))

    if is_select_features:
        features_df_list = p_map(
            partial(
                _select_features_func,
                verbose=verbose,
                features_config=features_config,
                is_select_features=is_select_features,
                use_optical_flow=use_optical_flow,
                ark_dir=ark_dir
            ),
            features_filepaths,
            num_cpus=num_jobs,
            desc="Generating ark using select_features"
        )
        pbar = tqdm(total=len(list(features_df_list)), desc="Writing ark files to disk")
        for features_df, ark_filepath, title in features_df_list:
            if features_df is not None:
                _create_ark_file(features_df, ark_filepath, title)
            pbar.update(1)     
            
    else:
        for features_filepath in tqdm(features_filepaths, desc="Generating ark/htk using interpolate_features data model"):

            if verbose:
                print(features_filepath)

            features_filename = features_filepath.split('/')[-1]
            features_extension = features_filename.split('.')[-1]
            features_df = None

            ark_filename = features_filename.replace(features_extension, 'ark')
            ark_filepath = os.path.join(ark_dir, ark_filename)
            title = ark_filename.replace('.ark', "")

            if 'alphapose' in features_filename:
                features_df = feature_extraction_alphapose(features_filepath, features_config['selected_features'], scale = 10, drop_na = True)

            elif features_extension == 'json':
                features_df = feature_extraction_kinect(features_filepath, features_config['selected_features'], scale = 10, drop_na = True)

            else:
                features_df = interpolate_feature_data(features_filepath, features_config['selected_features'], center_on_face = False, is_2d = True, scale = 10, drop_na = True)

            if features_df is not None:
                _create_ark_file(features_df, ark_filepath, title)
