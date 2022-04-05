# %%
import pandas as pd
import yaml
import os
from src.configs.data_config import TIME_VARYING_UNKNOWN_REALS,TIME_VARYING_UNKNOWN_CATEGORICALS, TIME_VARYING_KNOWN_REALS,TIME_VARYING_KNOWN_CATEGORICALS,STATIC_REALS,STATIC_CATEGORICALS,TARGET
from src.data.feature_processing import gen_grouped_lagged_features
from src.configs.model_config import MAX_ENCODER_LENGTH,MAX_PREDICTION_LENGTH

# %%
def build_output_dirs():
    "Function to build output directories for saved data"
    paths = ['./data/processed/ml_data/']

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def load_data():
    """Load raw data including list of uids for training validation and test"""

    data  = pd.read_csv('data/interim/raw.csv',index_col=0)

    with open('data/interim/train_val_test_uids.yml', 'r') as file:
        uids = yaml.safe_load(file)

    return data, uids

def get_train_indices(df,uids):
    """
    Maps the train/val/test uids to the indices of the dataframe
    """
    train_idx = df[lambda x:x.uid.isin(uids['train'])].index
    val_idx = df[lambda x:x.uid.isin(uids['val'])].index
    test_idx = df[lambda x:x.uid.isin(uids['test'])].index
    return train_idx,val_idx,test_idx


def generate_X(df,feats2lag,feats2lead,static_feats):
    """
    Generates X data according to the information availability i.e. lagged, lead and static
    
    """
    #Lagged data
    lagged_data = gen_grouped_lagged_features(df.loc[:,feats2lag+['uid']],group_col='uid',n_lags=MAX_ENCODER_LENGTH)

    # Lead data
    lead_data = gen_grouped_lagged_features(df.loc[:,feats2lead+['uid']],group_col='uid',n_lags=-MAX_PREDICTION_LENGTH)

    # Static data
    static_data = fpl_data.loc[:,static_feats]

    #Combine data sources
    X_data = pd.concat([static_data,lagged_data,lead_data],axis=1)

    return X_data

# %%
if __name__ == '__main__':

    #Build output directory
    build_output_dirs()

    #Load the raw data + train/val/test uids
    fpl_data,uids = load_data()

    # Create lags/leads and statics of the data
    feats2lag = TIME_VARYING_UNKNOWN_REALS+TIME_VARYING_UNKNOWN_CATEGORICALS+TIME_VARYING_KNOWN_REALS+TIME_VARYING_KNOWN_CATEGORICALS
    feats2lead = TIME_VARYING_KNOWN_REALS+TIME_VARYING_KNOWN_CATEGORICALS
    static_feats = STATIC_REALS+STATIC_CATEGORICALS
    X_data = generate_X(fpl_data,feats2lag,feats2lead,static_feats)

    # Create series of future y values
    y_data = gen_grouped_lagged_features(fpl_data.loc[:,TARGET+['uid']],group_col='uid',n_lags=-MAX_PREDICTION_LENGTH)
    assert y_data.isna().sum(axis=1).max()<MAX_PREDICTION_LENGTH, "Instances of all NaN in y_data"

    # Get the indicies of the df according to train/val/test membership
    train_idx,val_idx,test_idx = get_train_indices(fpl_data,uids)

    #Split the data
    X_train,X_val,X_test = X_data.loc[train_idx],X_data.loc[val_idx],X_data.loc[test_idx]
    y_train,y_val,y_test = y_data.loc[train_idx],y_data.loc[val_idx],y_data.loc[test_idx]

    # Save the data to file
    output_dir="./data/processed/ml_data/"
    X_train.to_csv(output_dir+'X_train.csv'), y_train.to_csv(output_dir+'y_train.csv')
    X_val.to_csv(output_dir+'X_val.csv'), y_val.to_csv(output_dir+'y_val.csv')
    X_test.to_csv(output_dir+'X_test.csv'), y_test.to_csv(output_dir+'y_test.csv')



# %%
