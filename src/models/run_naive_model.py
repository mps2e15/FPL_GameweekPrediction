# %%
import pandas as pd
import numpy as np
import os
from src.models.utils import export_predictions, load_ml_data

# %%
def build_output_dirs(paths):
    "Function to build output directories for saved data"
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def gen_naive_yhat(X,y):
    """
    Function to return naive y_hat
    This assumes that the last total_points is repeated over the prediction window
    """
    X = X['total_points_lag1']
    X = X.fillna(X.mean())
    X = np.expand_dims(X.values, axis=1)
    y_hat = np.repeat(X,y.shape[1], axis=1)
    return  y_hat

def gen_naive_avg_yhat(X,y):
    """
    Function to return naive y_hat
    This assumes average of the past performance is repeated over the prediction window
    """
    lagged_target_cols = X.columns.str.contains('total_points')
    X = X.loc[:,lagged_target_cols]
    X = X.fillna(X.mean())
    X=X.mean(axis=1)
    X = np.expand_dims(X.values, axis=1)
    y_hat = np.repeat(X,y.shape[1], axis=1)
    return  y_hat


if __name__ == '__main__':

    # Build the output dirs for storing the preds
    paths = ['./references/test_preds/']
    build_output_dirs(paths)

    # Load the ML train/test/validatio ndata
    X_train,y_train, X_val,y_val, X_test,y_test = load_ml_data()
    
    #Produce the naive model (using t-1)
    # y_hat_naive = gen_naive_yhat(X_test,y_test)
    # export_predictions(y_hat_naive,'Naive Model')

    #Produce the naive averaging model (using avg(t-n))
    y_hat_naive_avg = gen_naive_avg_yhat(X_test,y_test)
    export_predictions(y_hat_naive_avg ,'Naive Averaging Model')
# %%
