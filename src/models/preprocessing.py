"""
preprocessing file.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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
   
    return (logreg.score(scaledX_train, y_train_df),  confusion_matrix(y_train_df,y_pred))
