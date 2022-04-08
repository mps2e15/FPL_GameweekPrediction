# Fantasy Premier League: Points Prediction

## Overview
For predicting weekly Premier League fantasy points at a player level.

## Table of Contents

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


## Tecnologies Used:
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

