from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from dataclasses import dataclass,field
import pandas as pd

def to_object(x):
  return pd.DataFrame(x).astype(object)

fun_tr = FunctionTransformer(to_object)


@dataclass
class TimeseriesDataTransformer:
    """Class for preprocessing data timeseries data"""
    time_varying_unknown_reals: list[str] = field(default_factory=list)
    time_varying_unknown_categoricals: list[str] = field(default_factory=list)
    time_varying_known_reals: list[str] = field(default_factory=list)
    time_varying_known_categoricals: list[str] = field(default_factory=list)
    static_reals: list[str] = field(default_factory=list)
    static_categoricals: list[str] = field(default_factory=list)

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

def make_window_dataset(ds, window_size=38, shift=2, stride=1):
  windows = ds.window(window_size, shift=shift, stride=stride)

  def sub_to_batch(sub):
    return sub.batch(window_size, drop_remainder=True)
    #return sub.padded_batch(window_size, padded_shapes=(([None,None])),padding_values=(-1.0),drop_remainder=True)

  windows = windows.flat_map(sub_to_batch)
  return windows


def _parse_data(element,lag=16,lead=38):
    """Func to parse and truncate input sequences"""

    keys_to_features = {'index': tf.io.FixedLenSequenceFeature((),tf.int64,allow_missing=True),
                        'seq_len':tf.io.FixedLenFeature((), tf.int64),
                        'static_features': tf.io.FixedLenSequenceFeature((),tf.float32,allow_missing=True),
                        'time_varying_known': tf.io.FixedLenSequenceFeature((),tf.float32,allow_missing=True),
                        'time_varying_unknown': tf.io.FixedLenSequenceFeature((),tf.float32,allow_missing=True),
                        'labels': tf.io.FixedLenSequenceFeature((),tf.float32,allow_missing=True)
                        }

    parsed_features = tf.io.parse_single_example(element, keys_to_features)
    index =  tf.reshape(parsed_features['index'],[parsed_features['seq_len'],-1])
    static_features =  tf.reshape(parsed_features['static_features'],[parsed_features['seq_len'],-1])
    time_varying_known = tf.reshape(parsed_features['time_varying_known'],[parsed_features['seq_len'],-1])
    time_varying_unknown= tf.reshape(parsed_features['time_varying_unknown'],[parsed_features['seq_len'],-1])
    labels = tf.reshape(parsed_features['labels'],[parsed_features['seq_len'],-1])

    paddings = tf.constant([[lag,lead-1], [0, 0]])
    time_varying_known = tf.pad(time_varying_known, paddings, "CONSTANT",constant_values=-1.0)
    time_varying_unknown= tf.pad(time_varying_unknown, paddings, "CONSTANT",constant_values=-1.0)
    labels = tf.pad(labels, paddings, "CONSTANT",constant_values=-1)

    static_features = tf.repeat([static_features[0]],parsed_features['seq_len']+lag+(lead-1),axis=0)

    return (static_features,time_varying_known,time_varying_unknown),labels


# %%
def prepare_data(file_list,lag=16,lead=38,batch_size=32,train=False):
    input_ds = tf.data.TFRecordDataset(file_list)
    input_ds = input_ds.map(lambda x: _parse_data(x,lag=lag,lead=lead),num_parallel_calls=tf.data.AUTOTUNE)
    labels = input_ds.map(lambda x,y: tf.data.Dataset.from_tensor_slices(y),num_parallel_calls=tf.data.AUTOTUNE)\
                    .flat_map(lambda z: make_window_dataset(z, window_size=lead+lag, shift=1, stride=1)\
                        .map(lambda j:j[lag:,:]))
    mask = labels.map(lambda y:tf.cast(tf.math.logical_not(tf.math.equal(y, -1.)),dtype=tf.float32))

    statics = input_ds.map(lambda x,y: tf.data.Dataset.from_tensor_slices(x[0]),num_parallel_calls=tf.data.AUTOTUNE)\
                    .flat_map(lambda z: make_window_dataset(z, window_size=lead+lag, shift=1, stride=1)\
                        .map(lambda j:j[lag,:]))

    time_varying_known = input_ds.map(lambda x,y: tf.data.Dataset.from_tensor_slices(x[1]),num_parallel_calls=tf.data.AUTOTUNE)
    time_varying_known_past = time_varying_known.flat_map(lambda z: make_window_dataset(z, window_size=lead+lag, shift=1, stride=1)\
                         .map(lambda j:j[:lag,:]))
    time_varying_known_future = time_varying_known.flat_map(lambda z: make_window_dataset(z, window_size=lead+lag, shift=1, stride=1)\
                        .map(lambda j:j[lag:,:]))

    time_varying_unknown =  input_ds.map(lambda x,y: tf.data.Dataset.from_tensor_slices(x[2]),num_parallel_calls=tf.data.AUTOTUNE)\
                     .flat_map(lambda z: make_window_dataset(z, window_size=lead+lag, shift=1, stride=1)\
                         .map(lambda j:j[:lag,:]))


    X = tf.data.Dataset.zip((statics,time_varying_known_past,time_varying_unknown,time_varying_known_future))
    
    ds = tf.data.Dataset.zip((X,labels,mask))
    
    if train:
        ds = ds.batch(batch_size).shuffle(100, reshuffle_each_iteration=False).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
