"""
dataframe manipulation file
"""
import numpy as np
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

def handle_nan_values(dataframe: pd.DataFrame)->pd.DataFrame:
    """
    This function handles the special characters and NaN values

    Arguments:
    -----------------------------------------
    -dataframe  :   The initial dataframe
    
    Returns:
    -----------------------------------------
    -dataframe  :   dataframe
    """
    dataframe = dataframe.replace('?', np.nan)
    dataframe.fillna(dataframe.mean(), inplace=True)
    for col in dataframe.columns:
        if dataframe[col].dtypes == 'object':
            # Impute with the most frequent value
            dataframe = dataframe.fillna(dataframe[col].value_counts().index[0])
    return dataframe