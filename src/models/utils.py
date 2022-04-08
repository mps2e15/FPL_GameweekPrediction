
import yaml
import numpy as np
import os
import datetime
import pandas as pd
import joblib

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

def export_model(model,model_name,tf_model=False):
    """
    Function to export model and log the model in a YAML file.

    Inputs:
    model (sklearn.pipeline.Pipeline): Sci-kit learn model pipline
    model_name (str): Name of model (without formatting)

    Outputs:
    Exports model to output folder
    Exports .yml file containing model name, file reference and timestamp

    """
    name_fmt = model_name.lower().replace(' ','_')
    dir = './models/'
    if tf_model:
        model.save(f'{dir}{name_fmt}')
    else:
        joblib.dump(model, f"{dir}{name_fmt}.joblib")

    log_file = f"{dir}model_log.yml"
    if not os.path.exists(log_file):
        with open(log_file, 'w') as file:
            yaml.dump({}, file, default_flow_style=False)
    else:
        with open(log_file, "r") as file:
            log_dict = yaml.safe_load(file)

        log_dict[name_fmt] = {'name':model_name,\
                            'file':f'{dir}{name_fmt}.joblib',\
                            'timestamp':datetime.datetime.now() }
        
        with open(log_file, 'w') as file:
            yaml.dump(log_dict, file, default_flow_style=False)

def load_ml_data():
    """
    Function to load X,y data
    """
    X_train,y_train = pd.read_csv('data/processed/ml_data/X_train.csv',index_col=0,low_memory=False),pd.read_csv('data/processed/ml_data/y_train.csv',index_col=0,low_memory=False).astype('float')
    X_val,y_val = pd.read_csv('data/processed/ml_data/X_val.csv',index_col=0,low_memory=False),pd.read_csv('data/processed/ml_data/y_val.csv',index_col=0,low_memory=False).astype('float')
    X_test,y_test = pd.read_csv('data/processed/ml_data/X_test.csv',index_col=0,low_memory=False),pd.read_csv('data/processed/ml_data/y_test.csv',index_col=0,low_memory=False).astype('float')
    
    return X_train,y_train, X_val,y_val, X_test,y_test

def infer_dtypes(df):
    reals = df.select_dtypes(include=np.number).columns.values.tolist()
    cats = df.select_dtypes(exclude=np.number).columns.values.tolist()
    return reals, cats