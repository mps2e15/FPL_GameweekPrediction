# %%
from cgi import test
import tensorflow as tf

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

file_list = tf.io.matching_files('./data/processed/TF_records/train/*.tfrecord')
train_ds = prepare_data(file_list,lag=16,lead=6,batch_size=32,train=True).cache()

file_list = tf.io.matching_files('./data/processed/TF_records/val/*.tfrecord')
val_ds = prepare_data(file_list,lag=16,lead=6,batch_size=128,train=False).cache()

file_list = tf.io.matching_files('./data/processed/TF_records/test/*.tfrecord')
test_ds = prepare_data(file_list,lag=16,lead=6,batch_size=128,train=False).cache()


# %%
seq_past_len=16
seq_fut_len=6
static_inpt = tf.keras.layers.Input(shape=(4),name='static_input')
time_varying_known_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,4),name='tv_known_past')
time_varying_unknown_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,8),name='tv_unknown_past')
time_varying_known_future_inpt = tf.keras.layers.Input(shape=(seq_fut_len,4),name='tv_known_future')

time_varying_known_past = tf.keras.layers.Flatten()(time_varying_known_past_inpt)
time_varying_unknown_past = tf.keras.layers.Flatten()(time_varying_unknown_past_inpt)
past_concat = tf.keras.layers.Concatenate(name='concat_past')([static_inpt,time_varying_known_past,time_varying_unknown_past])
repeat_past = tf.keras.layers.RepeatVector(seq_fut_len)(past_concat)
future_concat = tf.keras.layers.Concatenate(name='concat_future')([repeat_past,time_varying_known_future_inpt])
drop = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3))(future_concat)
dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(16))(drop)
out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(dense)


model = tf.keras.Model([static_inpt,time_varying_known_past_inpt,time_varying_unknown_past_inpt,time_varying_known_future_inpt],out,name='CombinedModel')
opt = tf.keras.optimizers.Adam(1e-3)
model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                optimizer=opt)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,min_delta=0.0025,restore_best_weights=True)
model.fit(train_ds,validation_data=val_ds ,callbacks=[es],epochs=100,verbose=True)

# %%
model.evaluate(test_ds)

# %%
import numpy as np
preds = np.squeeze(model.predict(test_ds))
model_name= 'DNN'
from src.models.utils import export_predictions
export_predictions(preds,model_name )


# %%
seq_past_len=16
seq_fut_len=6
static_inpt = tf.keras.layers.Input(shape=(4),name='static_input')
static_rep = tf.keras.layers.RepeatVector(seq_past_len)(static_inpt)
time_varying_known_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,4),name='tv_known_past')
time_varying_unknown_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,8),name='tv_unknown_past')
past_concat = tf.keras.layers.Concatenate(name='concat_past')([static_rep,time_varying_known_past_inpt,time_varying_unknown_past_inpt])
encoder_l1 = tf.keras.layers.LSTM(16, return_state=True,dropout=0.2,name='encoder')
encoder_outputs1 = encoder_l1(past_concat)

decoder_inputs = tf.keras.layers.RepeatVector(seq_fut_len,name='encoder_repeat')(encoder_outputs1[0])
time_varying_known_future_inpt = tf.keras.layers.Input(shape=(seq_fut_len,4),name='tv_known_future')
fut_concat = tf.keras.layers.Concatenate(name='concat_fut')([decoder_inputs,time_varying_known_future_inpt])

decoder_l1 = tf.keras.layers.LSTM(16, return_sequences=True,name='decoder',dropout=0.1)(fut_concat,initial_state = encoder_outputs1[1:])
out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(decoder_l1)

model = tf.keras.Model([static_inpt,time_varying_known_past_inpt,time_varying_unknown_past_inpt,time_varying_known_future_inpt],out,name='CombinedModel')
opt = tf.keras.optimizers.Adam(1e-3)
model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                optimizer=opt)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,min_delta=0.0025,restore_best_weights=True)
model.fit(train_ds,validation_data=val_ds ,callbacks=[es],epochs=100,verbose=True)

# %%
import numpy as np
preds = np.squeeze(model.predict(test_ds))
model_name= 'LSTM'
from src.models.utils import export_predictions
export_predictions(preds,model_name )

# %%
import pandas as pd
import yaml
def load_data():
    """Load raw data including list of uids for training validation and test"""

    data  = pd.read_csv('data/interim/raw.csv',index_col=0)

    with open('data/interim/train_val_test_uids.yml', 'r') as file:
        uids = yaml.safe_load(file)

    return data, uids

data, uids = load_data()
# %%
