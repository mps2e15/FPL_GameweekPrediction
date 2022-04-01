"""Script used to scrape historic player and fixture data"""

# %%
import pandas as pd
import numpy as np
import os
from src.configs.data_config import MIN_YEAR,MAX_YEAR,RAW_COLUMNS,POS_IDS

# %%

MIN_YEAR = int(str(MIN_YEAR)[-2:])
MAX_YEAR = int(str(MAX_YEAR)[-2:])

def build_output_dirs():
    "Function to build output directories for saved data"
    paths = ['./data/raw/','./data/interim/']

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def collect_data(min_year,max_year,raw_cols):
    """
    Function to scrape data courtest of the vaastav
    """

    season_list = []
    for year in range (min_year,max_year+1):
    
        #Load player GW histories
        df= pd.read_csv(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{year}-{year+1}/gws/merged_gw.csv",
                        usecols=raw_cols,
                    encoding="ISO-8859-1")
        
        df['season'] = f"20{year}-{year+1}"
        
        
        #Load player position data and merge
        pos_ids = pd.read_csv(f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{year}-{year+1}/players_raw.csv',usecols=['id','element_type'],encoding="ISO-8859-1").rename(columns={'element_type':'pos_id'})
        df = df.merge(pos_ids,how='left',left_on='element',right_on='id')
        
        #Load fixture difficulty data
        fixture_df = pd.read_csv(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{year}-{year+1}/fixtures.csv",
                    usecols=['id','team_h_difficulty','team_a_difficulty'],
                encoding="ISO-8859-1").rename(columns={'id':'fixture'})
        df = df.merge(fixture_df,how='left',on='fixture')

        #append to season list
        season_list.append(df)

    # Merge the data
    historic_data  = pd.concat(season_list,axis=0).reset_index(drop=True)

    #Export data
    historic_data.to_csv('./data/raw/raw.csv')


def clean_data(position_dict):
    """Minor cleaning procedure which saves intermediate data outputs"""

    df = pd.read_csv('./data/raw/raw.csv',index_col=0)
    df['position'] = df.pos_id.map(position_dict)

    #Calculate fixture difficulty rating
    df['fdr_player'] = np.where(df.was_home==True,df.team_h_difficulty,df.team_a_difficulty)
    df['fdr_opposition'] = np.where(df.was_home==True,df.team_a_difficulty,df.team_h_difficulty)
    df['fdr_diff'] = df['fdr_player']-df['fdr_opposition']

    #Export data
    df.to_csv('./data/interim/raw.csv')


if __name__ == '__main__':
    build_output_dirs()
    collect_data(MIN_YEAR,MAX_YEAR,RAW_COLUMNS)
    clean_data(POS_IDS)




