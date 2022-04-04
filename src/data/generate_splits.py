# %%
import pandas as pd
import yaml
from src.configs.data_config import MINUMUM_SEASON_POINTS
from src.configs.model_config import RANDOM_SEED
from sklearn.model_selection import train_test_split

# %%
def load_data():
    """Load raw data"""
    return pd.read_csv('data/interim/raw.csv',index_col=0)

def get_player_filter(df,minimum_season_points):
    "Filtering criteria for players returning series same size as df input"
    points_filter = df.groupby('uid')['total_points'].transform(lambda x:x.sum()>=25)
    return points_filter

def get_valid_unique_players(df,filter):
    "Get unique player ids for valid players"
    return df.loc[filter,'uid'].drop_duplicates().values

def get_splits(players): 
    "Generate train/val/test splits"
    train_idx, test_idx = train_test_split(players,test_size=0.4, shuffle=True,random_state=RANDOM_SEED)
    test_idx, val_idx = train_test_split(test_idx,test_size=0.5, shuffle=True,random_state=RANDOM_SEED)
    return train_idx, val_idx, test_idx

def save_index(train_idx, val_idx, test_idx):
    "Save outputs to yml file"
    output_dict = {'train':train_idx.tolist(),'val':val_idx.tolist(),'test':test_idx.tolist()}
    with open('data/interim/train_val_test_uids.yml', 'w') as yaml_file:
        yaml.dump(output_dict, yaml_file, )


# %%
if __name__ == '__main__':
    fpl_data = load_data()
    player_filter = get_player_filter(fpl_data  ,MINUMUM_SEASON_POINTS)
    valid_unique_players  = get_valid_unique_players(fpl_data,player_filter)
    train_idx, val_idx, test_idx = get_splits(valid_unique_players)
    save_index(train_idx, val_idx, test_idx)
