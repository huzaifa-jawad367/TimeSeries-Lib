# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4 Dataset
"""
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm
import logging
import os
import pathlib
import sys
from urllib import request


def url_file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split('/')[-1] if len(url) > 0 else ''


def download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """

    def progress(count, block_size, total_size):
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\rDownloading {} to {} {:.1f}%'.format(url, file_path, progress_pct))
        sys.stdout.flush()

    if not os.path.isfile(file_path):
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')
    else:
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    # Original NPZ-based loader (commented out)
    # @staticmethod
    # def load(training: bool = True, dataset_file: str = '../dataset/m4') -> 'M4Dataset':
    #     """
    #     Load cached dataset.
    #
    #     :param training: Load training part if training is True, test part otherwise.
    #     """
    #     info_file = os.path.join(dataset_file, 'M4-info.csv')
    #     train_cache_file = os.path.join(dataset_file, 'training.npz')
    #     test_cache_file = os.path.join(dataset_file, 'test.npz')
    #     m4_info = pd.read_csv(info_file)
    #     return M4Dataset(ids=m4_info.M4id.values,
    #                      groups=m4_info.SP.values,
    #                      frequencies=m4_info.Frequency.values,
    #                      horizons=m4_info.Horizon.values,
    #                      values=np.load(
    #                          train_cache_file if training else test_cache_file,
    #                          allow_pickle=True))
    
    @staticmethod
    def load(training: bool = True, dataset_file: str = '../dataset/m4') -> 'M4Dataset':
        """
        Load M4 dataset from CSV files.
        
        :param training: Load training part if training is True, test part otherwise.
        :param dataset_file: Path to directory containing M4 CSV files
        """
        # Load M4 info file
        info_file = os.path.join(dataset_file, 'M4-info.csv')
        m4_info = pd.read_csv(info_file)
        
        # Load all seasonal pattern CSV files
        seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
        all_ids = []
        all_values = []
        
        for pattern in seasonal_patterns:
            # Construct CSV file name
            if training:
                csv_file = os.path.join(dataset_file, f'{pattern}-train.csv')
            else:
                csv_file = os.path.join(dataset_file, f'{pattern}-test.csv')
            
            # Check if file exists
            if not os.path.exists(csv_file):
                print(f'Warning: {csv_file} not found, skipping {pattern}')
                continue
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # First column contains IDs, remaining columns contain values
            ids = df.iloc[:, 0].values  # First column: series IDs
            values = df.iloc[:, 1:].values  # Remaining columns: time series values
            
            # Convert to list of arrays (handle variable length by removing trailing NaNs)
            for i, row in enumerate(values):
                # Remove trailing NaNs
                valid_values = row[~pd.isna(row)]
                all_values.append(valid_values)
                all_ids.append(ids[i])
        
        # Convert to numpy arrays
        all_ids = np.array(all_ids)
        all_values = np.array(all_values, dtype=object)
        
        # Get corresponding metadata from M4-info
        # Create a mapping from ID to info
        info_dict = {row['M4id']: row for _, row in m4_info.iterrows()}
        
        groups = []
        frequencies = []
        horizons = []
        
        for id_val in all_ids:
            if id_val in info_dict:
                groups.append(info_dict[id_val]['SP'])
                frequencies.append(info_dict[id_val]['Frequency'])
                horizons.append(info_dict[id_val]['Horizon'])
            else:
                # Default values if not found in info
                groups.append('Unknown')
                frequencies.append(1)
                horizons.append(6)
        
        return M4Dataset(
            ids=all_ids,
            groups=np.array(groups),
            frequencies=np.array(frequencies),
            horizons=np.array(horizons),
            values=all_values
        )


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }  # different predict length
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }  # from interpretable.gin


def load_m4_info() -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(INFO_FILE_PATH)
