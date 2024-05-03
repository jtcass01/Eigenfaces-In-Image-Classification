
__author__ = "Jacob Taylor Cassady"
__email__ = "jcassady@jh.edu"

from os.path import join
from typing import List

from pandas import DataFrame, Series, read_pickle
from numpy import array

from process_raw_adfes import ADFES_DIRECTORY

def load_adfes_dataframe() -> DataFrame:
    """Load the ADFES DataFrame from the pickle file.

    Returns:
        DataFrame: _description_"""
    return read_pickle(join(ADFES_DIRECTORY, 'adfes_dataframe.pkl'))


def min_max_norm(values: array, desired_min: float = 0., desired_max: float = 1.) -> array:
    min_value: float = min(values)
    max_value: float = max(values)
    return ((values-min_value)/(max_value-min_value)) *  (desired_max-desired_min) + desired_min
