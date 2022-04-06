# %%

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from src.configs.model_config import RANDOM_SEED
from src.models.utils import export_predictions, export_ml_model, load_ml_data, infer_dtypes
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def prepare_X_data(X_train,X_val,X_test):
    reals, cats = infer_dtypes(X_train)
    X_train.loc[:,cats] = X_train.loc[:,cats].astype(str)
    X_val.loc[:,cats] = X_val.loc[:,cats].astype(str)
    X_test.loc[:,cats]= X_test.loc[:,cats].astype(str)

    return X_train,X_val,X_test

def get_weights(y_train,y_val,y_test):
    y_train_weight,y_val_weight,y_test_weight = (~y_train.isna()).astype(int),\
                                                    (~y_val.isna()).astype(int),\
                                                        (~y_test.isna()).astype(int)
    return y_train_weight,y_val_weight,y_test_weight

def prepare_y_data(y_train,y_val,y_test):
    y_train,y_val,y_test =y_train.fillna(0),y_val.fillna(0),y_test.fillna(0)
    return y_train,y_val,y_test 

def build_preprocessor_pipe(reals,cats):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant',fill_value=-100)),
        ('scaler',  StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant',fill_value='missing')),
        ('one_hot', OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, reals),
            ('cats', categorical_transformer, cats)])
    
    return preprocessor


def build_and_train_lm(X_train,y_train, X_val,y_val, X_test,y_test,\
                        y_train_weight,y_val_weight,y_test_weight,\
                        preprocessor):

    #define model
    regr = MultiOutputRegressor(RidgeCV(cv=5))

    #Define and fit full model pipe
    model_pipe = Pipeline(steps=[('preprocessor',preprocessor),
                            ('regr',regr)])\
                                .fit(X_train,y_train.values)

    test_preds = model_pipe.predict(X_test)

    return model_pipe,test_preds

if __name__ == '__main__':
    # Load the ML train/test/validation data
    X_train,y_train, X_val,y_val, X_test,y_test = load_ml_data()

    #Prepare data for model
    X_train,X_val,X_test = prepare_X_data(X_train,X_val,X_test)
    y_train_weight,y_val_weight,y_test_weight = get_weights(y_train,y_val,y_test)
    y_train,y_val,y_test  =prepare_y_data(y_train,y_val,y_test)

    #Infer data types from X data
    reals, cats = infer_dtypes(X_train)

    #Define preprocessing pipe
    preprocessor = build_preprocessor_pipe(reals,cats)

    ##LINEAR MODEL
    #Build model
    regr,test_preds =build_and_train_lm(X_train,y_train, X_val,y_val, X_test,y_test,\
                                        y_train_weight,y_val_weight,y_test_weight,\
                                        preprocessor)

    #Export result
    model_name = 'ElasticNet LM'
    export_predictions(test_preds,model_name )
    export_ml_model(regr,model_name )



# %%
