# Fantasy Premier League: Points Prediction

Over 8 million players participate in the English [Fantasy Premier League](https://draft.premierleague.com/)  and [Draft Fantasy Premier League](https://draft.premierleague.com/), whereby managers must select an initial team of 15 players at the start of the season. During the course of the season, over 38 gameweeks, managers must also select teams and can also transfer players in and out of their teams given a set of constraints i.e. budget (fantasy), or, player availability (draft). 

A key part of of this challenge is anticipating how many **points** players will score in future gameweeks!

This repository uses **Machine Learning** (inc **Deep Learning**) in order to forecast future individual player performance given performance in the preceeding gameweeks. It does so by making use of openly available data which can be sourced weekly from the [Official Premier League API](https://fantasy.premierleague.com/api/).


## Table of Contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [ File Descriptions ](#File_Description)
2. [ Technologies Used ](#Technologies_Used)    
3. [ Executive Summary ](#Executive_Summary)   
3. [ Future Development ](#Future_Development)  

</details>

## File Descriptions
<details>
<a name="File_Description"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>[ Data ](./data/)</strong>: folder containing all data files
    * <strong> /raw/raw.csv</strong>: scraped data courtesy of [vastaav](https://github.com/vaastav)
    * <strong> /interim/raw.csv</strong>: Cleaned data without pre-processing
    * <strong> /interim/train_val_test_uids.yml</strong>: Split of player IDs according to train/val/test memebership
    * <strong> /interim/dl2ml_test_idx.npy</strong>: Mapping of TF predictions to ML index (used for MSE assessment)
    * <strong> /processed/ml_data/X_{type}.csv</strong>: Data used for training of ML models
    * <strong> /processed/TF_records/{subset}/shard_{.no}.tfrecord</strong>: TF records for training deep models
* <strong>[ Models ](./models/)</strong>: folder containing sci-kit learn and TF models
    * <strong> /dnn</strong>: A simple Deep Neural Network
    * <strong> /LSTM</strong>: Long short-term memory sequence model
    * <strong> /TCN</strong>: Temporal Convolutional Network sequence model
    * <strong> /elasticnet_lm.joblib</strong>: Linear ML Model
    * <strong> /lightgbm.joblib</strong>: Light Gradient Boosting ML model
* <strong>[ Notebooks ](./notebooks/)</strong>: folder containing Jupyter notebooks
    * <strong> /1. EDA.ipynb</strong>: Data exploration of raw data
* <strong>[ References ](./references/)</strong>: folder containing misc outputs of analysis
    * <strong> /test_preds/model_log.yml</strong>: Log recording model type and date last updated
    * <strong> /test_preds/{model_name}.npy</strong>: Numpy array of test predictions
    * <strong> /results/mse_result.yml</strong>: YAML file containing MSE result for each model, for each timestep
* <strong>[src ](./src/)</strong>: folder containing scripts and tooling
    * <strong> /configs/{type}.py</strong>: Files containing data and modelling for setting global parameters
    * <strong> /data/make_raw_dataset.py</strong>: Script for scraping raw data
    * <strong> /data/generate_splits.py</strong>: Script for splitting player seasons into test/val/train groups
    * <strong> /data/generate_ml_data.py</strong>: Script for processing ML data, and amongst other functions generates the relevant lags and leads for the the data
    * <strong> /data/generate_tf_data.py</strong>: Script for pre-processing the data and writting to TF-records
    * <strong> /data/tf_data_utils.py</strong>: Supporting functions for processing tf records
    * <strong> /data/feature_processing.py</strong>: Supporting functions for processing tf records and sci-kit learn models
    * <strong> /models/run_{model_name}.py</strong>: Scripts to optimize, train models then export the model and test predictions
    * <strong> /models/tf_model_utils.py</strong>: Supporting functions and custom layers for deep models
    * <strong> /models/utils.py</strong>: Utility functions for reporting results and saving models
    * <strong> /plotting/eda.py</strong>: Utility functions plottinh eda data
* <strong>[ environment-py39.yml ](./environment-py39.yml)</strong>: Conda environment
* <strong>[ requirments.txt ](./requirments.txt)</strong>: Environment dependencies

</details>


## Technologies Used:
<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>Matplotlib</strong>
* <strong>Scikit-Learn</strong>
* <strong>Keras</strong>
* <strong>Tensorflow</strong>

</details>

## Executive Summary:
<a name="Executive_Summary"></a>

In order to predict total weekly points (top plot below) we have a number of input featured available from the Premier League API.

<h5 align="center">Example player history with continuous features</h5>
<p align="center">
  <img src="https://github.com/mps2e15/FPL_GameweekPrediction/blob/main/references/plots/player_example.png" width=500>
</p>

While some future information is known (i.e. position, Fixture Difficulty Ratings), most data is only available retrospectively. Our EDA analysis shows that for this past information, most information tends to be relevant going back 6 games with some residual importance lasting ~14 games.

<h5 align="center">Feature Importances for Lagged Features</h5>
<p align="center">
  <img src="https://github.com/mps2e15/FPL_GameweekPrediction/blob/main/references/plots/lagged_importance.png" width=500>
</p>


To leverage this information to predict future gameweek points for the next 6 games, a range of models are trialled:

* <strong>Naive Model</strong>: Assumes average total points from last 16 games projected forward
* <strong>ElasticNet LM</strong>:   Linear ML model
* <strong>LightGBM</strong>: Non-linear Gradient Boosting Machine
* <strong>LSTM</strong>: Deep Long Short Term Memory model
* <strong>TCN</strong>: Deep Temporal Convolutional Neural Network

The test MSE scores are presented below:

<h5 align="center">Means Squared Error Performance</h5>
<p align="center">
  <img src="https://github.com/mps2e15/FPL_GameweekPrediction/blob/main/references/plots/performance.png" width=400>
</p>

## Future Development:
<a name="Future_Development"></a>
* <strong>REST API</strong>: Deployment of model using api that pulls the data and returns future gameweek predictions
* <strong>TFT Network</strong>: Includion of an additional model, the [Temporal Fusion Transformer (TFT)](https://ai.googleblog.com/2021/12/interpretable-deep-learning-for-time.html) which has been shown to perform well for timeseries data
    
