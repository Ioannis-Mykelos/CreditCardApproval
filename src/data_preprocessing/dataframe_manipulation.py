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
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "cc_approvals.data"
    dataframe = pd.read_csv(data_path, header=None)
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

def rename_columns(dataframe: pd.DataFrame)->pd.DataFrame:
    """
    This function handles the special characters and NaN values

    Arguments:
    -----------------------------------------
    -dataframe  :   The initial dataframe
    
    Returns:
    -----------------------------------------
    -dataframe  :   dataframe
    """
    dictionary_columns = {0:'col1', 1:'col2', 3:'col3', 4:'col4', 5:'col5', 6:'col6', 
                      7:'col7', 8:'col8', 9:'col9', 10:'col10', 11:'col11', 12: 'col12',
                      13: 'col13', 14: 'col14', 15: 'col15'}
    dataframe.rename(columns = dictionary_columns, inplace = True)
    return dataframe

    
