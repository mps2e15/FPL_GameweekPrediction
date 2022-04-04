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


    #The lag loop
    lagged_cols = []
    for col in cols2lag:
        for lag in range(1,n_lags+1):
            #lagged_X[f'{col}_l{lag}'] = df.groupby(group_col)[col].shift(lag).name()
            lagged_col = df.groupby(group_col)[col].shift(lag).rename(f'{col}_l{lag}')
            lagged_cols.append(lagged_col)
    
    #Concat the vars to a single df
    lagged_df = pd.concat(lagged_cols,axis=1)

    #add index using same as input data
    lagged_df.index = df.index 

    return lagged_df
