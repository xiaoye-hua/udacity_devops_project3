import logging

import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier
logging.getLogger(__name__)


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = XGBClassifier()
    model.fit(X=X_train, y=y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_sliced_model_metrics(col: str, df: pd.DataFrame, y_df):
    unique_lst = df[col].unique()
    logging.info(f"Sliced precision, recall & fbeta; Col: {col}")
    for value in unique_lst:
        preds = df[df[col]==value]['predict']
        y = y_df[df[col]==value]
        precision, recall, fbeta = compute_model_metrics(y=y, preds=preds)
        logging.info(f"    value: {value}")
        logging.info(f"    {precision}; {recall}; {fbeta}")


def inference(model: XGBClassifier, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pred = model.predict(X=X)
    return pred
