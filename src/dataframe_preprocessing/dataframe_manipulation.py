"""
dataframe manipulation file
"""

import pandas as pd

def load_data()->pd.DataFrame:
    """
    This function loads the initial dataframe

    Arguments:
    -----------------------------------------
    -None       :    Nothing

    Returns:
    -----------------------------------------
    -dataframe  :    The credit cards dataframe
    """
    dataframe = pd.read_csv("data/cc_approvals.data", header=None)
    return dataframe
