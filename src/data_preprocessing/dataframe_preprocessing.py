"""
dataframe preprocessing file
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,  MinMaxScaler
from sklearn.model_selection import train_test_split

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

def split_data(dataframe: pd.DataFrame)->pd.DataFrame:
    """
    This function splits our dataframe

    Arguments:
    -----------------------------------------
    -dataframe  :   The initial dataframe
    
    Returns:
    -----------------------------------------
    -dataframe  :   dataframe
    """
    dataframe = dataframe.drop(labels = ['col11','col13'],axis = 1)
    dataframe = dataframe.to_numpy()
    X, y = dataframe[:,0:13] , dataframe[:,13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)

def scale_data(dataframe_train: pd.DataFrame, dataframe_test: pd.DataFrame)->pd.DataFrame:
    """
    This function rescaling the X_train and X_test dataframes

    Arguments:
    -----------------------------------------
    -dataframe_train  :   The X_train dataframe
    -dataframe_train  :   The X_test dataframe

    Returns:
    -----------------------------------------
    -scaledX_train    :   The X_train dataframe scaled
    -scaledX_test     :   The X_test dataframe scaled
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX_train = scaler.fit_transform(X_train)
    rescaledX_test = scaler.transform(X_test)
    return (rescaledX_train, rescaledX_test)
    