# %% 
import yaml
from src.models.utils import load_ml_data
import numpy as np
from sklearn.metrics import mean_squared_error
import os 

def build_output_dirs(paths):
    "Function to build output directories for saved data"

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def load_test_logs():
    log_file = 'references/test_preds/model_log.yml'
    with open(log_file, "r") as file:
            model_dict = yaml.safe_load(file)
    return model_dict

def get_test_weights(y):
    return (~y.isna()).astype(int)

def run_assessment_loop(log,y_true,y_mask):
    rmse_dict = {}
    r2_dict = {}

    for model in log.keys():
        model_name = log[model]['name']
        y_hat = np.load(log[model]['file'])
        
        rmse = mean_squared_error(y_true,y_hat ,sample_weight=y_mask,squared=True,multioutput='raw_values')
        rmse_dict[model_name] = rmse.tolist()
   
    return rmse_dict

def save_results(result_dict):
    res_file = './references/results/rmse_result.yml'
    with open(res_file, 'w') as file:
            yaml.dump(result_dict, file)




if __name__ == '__main__':

    #Generate output paths
    build_output_dirs(['./references/results/'])

    # Load data
    _,_, _,_, _,y_test = load_ml_data()

    #Get the weights for na data
    y_test_mask = get_test_weights(y_test).values

    #Fill missing data (req for sklearn)
    y_test = y_test.fillna(-100).values

    #Get the model logs
    model_dict = load_test_logs()

    rmse_dict = run_assessment_loop(model_dict,y_test,y_test_mask)

    #Export the result
    save_results(rmse_dict)

# %%