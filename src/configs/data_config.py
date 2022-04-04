


MIN_YEAR = 2018
MAX_YEAR = 2021

RAW_COLUMNS = ['name','element','influence','creativity','threat','ict_index','total_points','value','minutes','bps','GW','fixture','was_home']

POS_IDS = {1:'GKP', 2:'DEF',3:'MID', 4:'FWD'}

IDENTIFIERS =  ['uid','element','season','name','fixture','GW']
REALS = ['value','bps', 'influence','creativity','threat','ict_index','minutes','fdr_player','fdr_diff']
CATEGORIES = ['position','was_home']
TARGET = ['total_points']

STATIC_FEATS = ['value','position','was_home']
STATIC_FEATS = ['value','position','was_home']


#Definitions inspired bt pytorch-forecasting
# https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/data/timeseries.py

STATIC_CATEGORICALS = ['position']
STATIC_REALS = []
TIME_VARYING_KNOWN_CATEGORICALS = ['was_home']
TIME_VARYING_KNOWN_REALS = ['fdr_player','fdr_diff']
TIME_VARYING_UNKNOWN_CATEGORICALS = []
TIME_VARYING_UNKNOWN_REALS = ['bps', 'influence','creativity','threat','ict_index','minutes','total_points','value']

#Min season minutes forseason points inclusion in analysis
MINUMUM_SEASON_POINTS = 25