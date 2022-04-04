import pandas as pd

def crosscorr(data,x_col,y_col, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return data[x_col].corr(data.groupby(['name','season'])[y_col].shift(lag),method='spearman')

def gen_grouped_lagged_features(df,group_col,n_lags):
    """
    Function to return a new DF containing lagged features accorging to group memebership

    Inputs:
    df (pd.DataFrame): Pandas df containing columns to lag
    group col (str): Column define any discrete points where the lags should occur i.e. player/season ref

    Returns:
    laggged_X (pd.DataFrame): New df containing lagged feautures sharing same index as original df
    """
    assert group_col in df.columns, f"No feature named {group_col} in df"

    #Remove grouper from columns
    cols2lag = [col for col in df.columns if col!=group_col]

    #Define if to lag or lead
    if n_lags>0:
        lag_range = range(1,n_lags+1)
        name='lag'
    else:
        lag_range = range(n_lags+1,0+1)
        name='lead'

    #The lag loop
    lagged_cols = []
    for col in cols2lag:
        for lag in lag_range:
            #lagged_X[f'{col}_l{lag}'] = df.groupby(group_col)[col].shift(lag).name()
            lagged_col = df.groupby(group_col)[col].shift(lag).rename(f'{col}_{name}{abs(lag)}')
            lagged_cols.append(lagged_col)
    
    #Concat the vars to a single df
    lagged_df = pd.concat(lagged_cols,axis=1)

    #add index using same as input data
    lagged_df.index = df.index 

    #reorder cols if lead
    if n_lags<0:
        lagged_df = lagged_df.iloc[:,::-1]
    return lagged_df
