
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_example(fpl_data,target,reals,uid=None):
    """
    Function to sample random player/season and plot contuninuous vals

    inputs:
    fpl_data (pd.DataFrame): pandas df containing player features
    target (list): List containing target value i.e. total_points
    reals (list): List of continuous value to be used in predictive setting
    uid (int): Unique player id to plot, if None then a random player is selected

    returns:
    plot of input and taget features
    
    """

    #Sample if no uid provided
    if uid==None:
        print('Random choice selected')
        uid = fpl_data.uid.drop_duplicates().sample(1).values[0]

    player_details = fpl_data[lambda x: x.uid==uid].head(1).loc[:,['name','season']]
    fpl_data.loc[fpl_data.uid==uid,target+reals].reset_index(drop=True).\
                        plot(subplots=True,xlabel='Game',title=f"Player: {player_details.name.values[0]}",figsize=(12,6))
    plt.show()

def plot_static_and_future_importances(importances,static_categoricals,time_varying_known_categoricals,time_varying_known_reals):

    """
    Function to plot static importances and known time varying plots

    Inputs:
    importances (pd.Series): pandas series with feature index and importance val
    static_categoricals (list): cols
    time_varying_known_categoricals (list): cols
    time_varying_known_reals (list): cols

    Returns:
    matplotlib.plot    
    
    """
    fig,axs = plt.subplots(1, 3,figsize = (12,3),sharey=True)

    #static categores
    is_static_categorical = np.sum([(importances.index.str.contains(col) & ~importances.index.str.contains('_l'))  for col in static_categoricals],axis=0)>0
    importances.loc[is_static_categorical].plot.bar(ylim=(0,importances.max()),ax=axs[0],title='Static Categoricals')

    #time varying categories
    is_time_varying_categorical = np.sum([(importances.index.str.contains(col) & ~importances.index.str.contains('_l'))  for col in time_varying_known_categoricals],axis=0)>0
    importances.loc[is_time_varying_categorical].plot.bar(ylim=(0,importances.max()),ax=axs[1],title='Time Varying Categoricals')

    #Tiem varying reals
    importances.loc[time_varying_known_reals].plot.bar(ylim=(0,importances.max()),ax=axs[2],title='Time Varying Reals')
    plt.show()

def plot_time_varying_importances(importances,time_varying_unknown_reals_transformed,time_varying_known_reals_transformed,time_varying_known_categoricals_transformed):

    """
    Function to plot static time varying plots using historical lagged data

    Inputs:
    importances (pd.Series): pandas series with feature index and importance val
    time_varying_unknown_reals_transformed (list): cols
    time_varying_known_reals_transformed (list): cols
    time_varying_known_categoricals_transformed (list): cols

    Returns:
    matplotlib.plot    
    
    """

    fig,axs = plt.subplots(1, 3,figsize = (12,4),sharey=True)


    #Time varying unknown
    subset_importances = importances.loc[time_varying_unknown_reals_transformed].to_frame()
    subset_importances['lag'] = subset_importances.index.str.split('_').str[-1]
    subset_importances['lag']  = subset_importances['lag'].str.extract('(\d+)').astype(int)
    subset_importances['feature'] = subset_importances.index.str.split('_').str[:-1].str.join('_')
    subset_importances = subset_importances.sort_values(by=['lag']).reset_index(drop=True)
    subset_importances = subset_importances.set_index(['lag','feature']).unstack().rolling(5,min_periods=1).mean()
    subset_importances.columns = subset_importances.columns.droplevel()
    importance_order = subset_importances.max().sort_values(ascending=False).index
    subset_importances.loc[:,importance_order].plot(ax=axs[0],title='Time Varying Unknown Importance')
    axs[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Time varying known (reals)
    subset_importances = importances.loc[time_varying_known_reals_transformed].to_frame()
    subset_importances['lag'] = subset_importances.index.str.split('_').str[-1]
    subset_importances['lag']  = subset_importances['lag'].str.extract('(\d+)').astype(int)
    subset_importances['feature'] = subset_importances.index.str.split('_').str[:-1].str.join('_')
    subset_importances = subset_importances.sort_values(by=['lag']).reset_index(drop=True)
    subset_importances = subset_importances.set_index(['lag','feature']).unstack().rolling(5,min_periods=1).mean()
    subset_importances.columns = subset_importances.columns.droplevel()
    importance_order = subset_importances.max().sort_values(ascending=False).index
    subset_importances.loc[:,importance_order].plot(ax=axs[1],title='Time Varying Known Importance (REALS)')

    # Time varying known (categories) - not uses different indexing with required grouping
    is_time_varying_categorical = np.sum([importances.index.str.contains(col) for col in time_varying_known_categoricals_transformed],axis=0)>0
    subset_importances = importances.loc[is_time_varying_categorical].to_frame()
    subset_importances['lag'] = subset_importances.index.str.split('_').str[-2]
    subset_importances['lag']  = subset_importances['lag'].str.extract('(\d+)').astype(int)
    subset_importances['feature'] = subset_importances.index.str.split('_').str[:-2].str.join('_')
    subset_importances = subset_importances.sort_values(by=['lag']).reset_index(drop=True)
    subset_importances = subset_importances.groupby(['lag','feature']).sum()
    subset_importances = subset_importances.unstack().rolling(5,min_periods=1).mean()
    subset_importances.columns = subset_importances.columns.droplevel()
    importance_order = subset_importances.max().sort_values(ascending=False).index
    subset_importances.loc[:,importance_order].plot(ax=axs[2],title='Time Varying Known Importance (CATS)')

    fig.suptitle('Lagged Importance by Feature Type')

    for ax in axs.flatten():
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 2))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

    plt.tight_layout()
    plt.show()