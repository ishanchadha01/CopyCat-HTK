"""Converts .ark files to .htk files for use by HTK.

Methods
-------
create_htk_files
"""
import os
import glob
import shutil
import tqdm
from p_tqdm import p_map
from functools import partial

def convert_ark_htk(ark_file, htk_dir, sample_period=40000):
    """Converts .ark files to .htk files for use by HTK.

    Parameters
    ----------
    ark_file : str
        Path to .ark file.
    htk_file : str
        Path to .htk file.
    sample_period : int
        Sample period.
    """
    # CHANGE KALDI PATH TO BE CORRECT!!!
    kaldi_command = (f'/espnet/tools/kaldi/src/featbin/copy-feats-to-htk '
                     f'--output-dir={htk_dir} '
                     f'--output-ext=htk '
                     f'--sample-period={sample_period} '
                     f'ark:{ark_file}'
                     f'>/dev/null 2>&1')

    ##last line silences stdout and stderr

    os.system(kaldi_command)

def create_htk_files(htk_dir: str = os.path.join('data', 'htk'), ark_dir: str = os.path.join('data', 'ark', '*.ark'), num_jobs: int = os.cpu_count()) -> None:
    """Converts .ark files to .htk files for use by HTK.
    """
    if os.path.exists(htk_dir):
        shutil.rmtree(htk_dir)

    os.makedirs(htk_dir)
    
    ark_files = glob.glob(ark_dir)
    
    p_map(partial(convert_ark_htk, htk_dir=htk_dir, sample_period=40000), ark_files, num_cpus=num_jobs)