"""
preprocessing file.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

def logistic_regression(scaledX_train, y_train_df):
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
    # Instantiate a LogisticRegression classifier with default parameter values
    logreg = LogisticRegression()

    # Fit logreg to the train set
    logreg.fit(scaledX_train, y_train_df)
    # Import confusion_matrix

    # Use logreg to predict instances from the test set and store it
    y_pred = logreg.predict(scaledX_train)

    # Get the accuracy score of logreg model and print it
    print("Accuracy of logistic regression classifier: ", logreg.score(scaledX_train, y_train_df))
    print( confusion_matrix(y_train_df,y_pred))

    return (logreg.score(scaledX_train, y_train_df),  confusion_matrix(y_train_df,y_pred))

def best_logistic_regression(X_dataframe: pd.DataFrame, y_dataframe:pd.DataFrame):
    """
    This function finds the best parameters and score

    Arguments:
    -----------------------------------------
    -X_dataframe  :   The X dataframe
    -y_dataframe  :   The y dataframe

    Returns:
    -----------------------------------------
    -best_score    
    -best_params
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    logreg = LogisticRegression()
    tol = [1, 0.1, 0.01, 0.001 ,0.0001, 0.00001]
    max_iter = [50, 100, 150, 200, 250, 300]
    param_grid = dict(tol=tol, max_iter=max_iter)
    grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
    rescaledX = scaler.fit_transform(X_dataframe)
    grid_model_result = grid_model.fit(rescaledX, y_dataframe)
    best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
    print("Best: %f using %s" % (best_score, best_params))

    return (best_score, best_params)