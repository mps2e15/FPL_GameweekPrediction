# %%
import pandas as pd
import numpy as np
import os
import yaml
import datetime

# %%
def build_output_dirs(paths):
    "Function to build output directories for saved data"
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def load_data():
    """
    Function to load X,y data
    """
    X_train,y_train = pd.read_csv('data/processed/ml_data/X_train.csv',index_col=0,low_memory=False),pd.read_csv('data/processed/ml_data/y_train.csv',index_col=0,low_memory=False).astype('float')
    X_val,y_val = pd.read_csv('data/processed/ml_data/X_val.csv',index_col=0,low_memory=False),pd.read_csv('data/processed/ml_data/y_val.csv',index_col=0,low_memory=False).astype('float')
    X_test,y_test = pd.read_csv('data/processed/ml_data/X_test.csv',index_col=0,low_memory=False),pd.read_csv('data/processed/ml_data/y_test.csv',index_col=0,low_memory=False).astype('float')
    
    return X_train,y_train, X_val,y_val, X_test,y_test

def gen_naive_yhat(X,y):
    """
    Function to return naive y_hat
    This assumes that the last total_points is repeated over the prediction window
    """
    X = X['total_points_lag1']
    X = X.fillna(X.median())
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
    X = X.fillna(X.median())
    X=X.mean(axis=1)
    X = np.expand_dims(X.values, axis=1)
    y_hat = np.repeat(X,y.shape[1], axis=1)
    return  y_hat

# %%
def export_predictions(preds,model_name):
    """
    Function to export model predictions and log the result in a YAML file.

    Inputs:
    preds (np.darray): Numpy array of future predictions
    model_name (str): Name of model (without formatting)

    Outputs:
    Exports numpy array to output folder
    Exports .yml file containing model name, file reference and timestamp

    """
    assert isinstance(preds,np.ndarray), "Predictons should be numpy array"
    name_fmt = model_name.lower().replace(' ','_')
    np.save(f'./references/test_preds/{name_fmt}.npy', preds)

    log_file = './references/test_preds/model_log.yml'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as file:
            yaml.dump({}, file, default_flow_style=False)
    else:
        with open(log_file, "r") as file:
            log_dict = yaml.safe_load(file)

        log_dict[name_fmt] = {'name':model_name,\
                            'file':f'./references/test_preds/{name_fmt}.npy',\
                            'timestamp':datetime.datetime.now() }
        
        with open(log_file, 'w') as file:
            yaml.dump(log_dict, file, default_flow_style=False)


if __name__ == '__main__':

    # Build the output dirs for storing the preds
    paths = ['./references/test_preds/']
    build_output_dirs(paths)

    # Load the ML train/test/validatio ndata
    X_train,y_train, X_val,y_val, X_test,y_test = load_data()
    
    #Produce the naive model (using t-1)
    y_hat_naive = gen_naive_yhat(X_test,y_test)
    export_predictions(y_hat_naive,'Naive Model')

    #Produce the naive averaging model (using avg(t-n))
    y_hat_naive_avg = gen_naive_avg_yhat(X_test,y_test)
    export_predictions(y_hat_naive_avg ,'Naive Averaging Model')
# %%
