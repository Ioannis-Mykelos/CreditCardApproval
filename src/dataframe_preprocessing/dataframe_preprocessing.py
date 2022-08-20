"""
dataframe preprocessing file
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encoding_the_columns(dataframe: pd.DataFrame)->pd.DataFrame:
    """
    This function encodes the columns

    Arguments:
    -----------------------------------------
    -dataframe  :   The initial dataframe
    
    Returns:
    -----------------------------------------
    -dataframe  :   dataframe
    """

    # Instantiate LabelEncoder
    le=LabelEncoder()
    for column in dataframe.columns.to_numpy():
        # Compare if the dtype is object
        if dataframe[column].dtypes=='object':
            # Use LabelEncoder to do the numeric transformation
            dataframe[column]=le.fit_transform(dataframe[column])
    return dataframe
    
