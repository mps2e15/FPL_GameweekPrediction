# %%
import pandas as pd
import numpy as np
import yaml
from src.configs.data_config import TIME_VARYING_UNKNOWN_REALS,TIME_VARYING_UNKNOWN_CATEGORICALS, TIME_VARYING_KNOWN_REALS,TIME_VARYING_KNOWN_CATEGORICALS,STATIC_REALS,STATIC_CATEGORICALS,TARGET
from src.configs.data_config import TF_RECORDS_PER_SHARD
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from dataclasses import dataclass
import joblib
import os

def load_data():
    """Load raw data including list of uids for training validation and test"""

    data  = pd.read_csv('data/interim/raw.csv',index_col=0)

    with open('data/interim/train_val_test_uids.yml', 'r') as file:
        uids = yaml.safe_load(file)

    return data, uids

def build_output_dirs(paths):
    "Function to build output directories for saved data"

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def to_object(x):
  return pd.DataFrame(x).astype(object)

fun_tr = FunctionTransformer(to_object)

@dataclass
class TimeseriesDataTransformer:
    """Class for preprocessing data timeseries data"""
    time_varying_unknown_reals: list[str]
    time_varying_unknown_categoricals: list[str]
    time_varying_known_reals: list[str]
    time_varying_known_categoricals: list[str]
    static_reals: list[str]
    static_categoricals: list[str]

    def __post_init__(self):
        self.static_transformer = self.build_column_transformer(self.static_reals,\
                                                                self.static_categoricals)
        self.time_varying_known_transformer = self.build_column_transformer(self.time_varying_known_reals,\
                                                        self.time_varying_known_categoricals)
        self.time_varying_unknown_transformer = self.build_column_transformer(self.time_varying_unknown_reals,\
                                                self.time_varying_unknown_categoricals)                                                 
    
    def fit(self,X_train):
        self.static_transformer.fit(X_train)
        self.time_varying_known_transformer.fit(X_train)
        self.time_varying_unknown_transformer.fit(X_train)

    def transform(self,X):
        static_transformed = self.static_transformer.transform(X)
        time_varying_known_transformed = self.time_varying_known_transformer.transform(X)
        time_varying_unknown_transformed = self.time_varying_unknown_transformer.transform(X)
        return static_transformed,time_varying_known_transformed,time_varying_unknown_transformed

    def build_column_transformer(self,reals,cats):

        transfomers = []

        if len(reals)>0:
            numeric_transformer = Pipeline(steps=[
            ('scaler',  MinMaxScaler()),
            ('imputer', SimpleImputer(strategy='constant',fill_value=-1))])

            transfomers.append(('num', numeric_transformer, reals))
        
        if len(cats)>0:
            categorical_transformer = Pipeline(steps=[
                ('object_transfotmer',fun_tr),
                ('imputer', SimpleImputer(strategy='constant',fill_value='missing')),
                ('one_hot', OneHotEncoder(handle_unknown="ignore"))])
            transfomers.append(('cats', categorical_transformer, cats))

        preprocessor = ColumnTransformer(
            transformers=transfomers,sparse_threshold=0)
        
        return preprocessor

# %%
@dataclass
class TFdata_serializer:
    """Class for preprocessing data timeseries data"""

    def _int64_feature(self,value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64s_feature(self,value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(self,value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _floats_feature(self,value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bools_feature(self,value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def serialize_element(self,index,static_features,time_varying_known_features,time_varying_unknown_features,labels):
        """Returns serialized data given numpy array"""
        feature = {
            'index': self._int64s_feature(index.flatten()),
            'seq_len': self._int64_feature(index.shape[0]),

            'static_features': self._floats_feature(static_features.flatten()),
            'time_varying_known': self._floats_feature(time_varying_known_features.flatten()),
            'time_varying_unknown': self._floats_feature(time_varying_unknown_features.flatten()),
            'labels': self._floats_feature(labels.flatten()),
            }

        return tf.train.Example(features=tf.train.Features(feature=feature))




# %%
if __name__ == '__main__':

    #Create new output directory for TF records
    new_dirs = [f'./data/processed/TF_records/{subset}/' for subset in ['train','val','test']]
    build_output_dirs(new_dirs)

    #Load the data and uids
    data, uids = load_data()

    #Defien data transformer for pre-processing data
    transformer = TimeseriesDataTransformer(time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
                                    time_varying_unknown_categoricals=TIME_VARYING_UNKNOWN_CATEGORICALS,
                                    time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
                                    time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
                                    static_reals=STATIC_REALS,
                                    static_categoricals=STATIC_CATEGORICALS)

    #Fit and save the transformer
    transformer.fit(data[lambda x:x.uid.isin(uids['train'])])
    joblib.dump(transformer,'./models/ts_data_transformer.joblib')

    #Loop to output TF records
    for subset in ['train','val','test']:

        subset_uids = uids[subset] 

        #Calculate number of required shards
        n_shards = (len(subset_uids)//TF_RECORDS_PER_SHARD)+(1 if len(subset_uids) % TF_RECORDS_PER_SHARD != 0 else 0)
        
        index=0 #stard index

        for shard in range(n_shards):
            
            filename=f"./data/processed/TF_records/{subset}/shard_{shard}.tfrecord"
            end = index + TF_RECORDS_PER_SHARD if len(subset_uids) > (index + TF_RECORDS_PER_SHARD) else len(subset_uids)
            
            with tf.io.TFRecordWriter(filename) as writer:

                for uid in subset_uids[index:end]:

                    player_data = data[lambda x:x.uid==uid] #subset player

                    #Transform the data
                    static,time_varying_know,time_varying_unknow = transformer.transform(player_data)
                    
                    #Serialize the data and add meta info
                    se = TFdata_serializer().serialize_element(index=player_data.index.values,
                                                                static_features=static,
                                                                time_varying_known_features=time_varying_know,
                                                                time_varying_unknown_features=time_varying_unknow,
                                                                labels=player_data[TARGET].values)

                    #Write to record
                    writer.write(se.SerializeToString())

                #Update index
                index=end


# %%