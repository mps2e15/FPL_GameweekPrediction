# %%
import tensorflow as tf
from src.data.tf_data_utils import prepare_data
#from src.models.tf_model_utils import TFT_Decoder,TFT_Encoder
from src.models.utils import export_predictions,export_model
from src.configs.model_config import MAX_ENCODER_LENGTH,MAX_PREDICTION_LENGTH,RANDOM_SEED
import numpy as np
import optuna


# %%

def get_TF_dataset(lag,lead):

    """Function to load TF dataset"""

    file_list = tf.io.matching_files('./data/processed/TF_records/train/*.tfrecord')
    train_ds = prepare_data(file_list,lag=lag,lead=lead,batch_size=32,train=True).cache()

    file_list = tf.io.matching_files('./data/processed/TF_records/val/*.tfrecord')
    val_ds = prepare_data(file_list,lag=lag,lead=lead,batch_size=128,train=False).cache()

    file_list = tf.io.matching_files('./data/processed/TF_records/test/*.tfrecord')
    test_ds = prepare_data(file_list,lag=lag,lead=lead,batch_size=128,train=False).cache()

    return train_ds,val_ds,test_ds



class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = tf.keras.layers.Dense(units)
        self.sigmoid = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense =tf.keras.layers.Dense(units, activation="elu")
        self.linear_dense =tf.keras.layers.Dense(units)
        self.dropout =tf.keras.layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm =tf.keras.layers.LayerNormalization()
        self.project =tf.keras.layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class VariableSelection(tf.keras.layers.Layer):
    def __init__(self, num_features, seq_len=16,units=16, dropout_rate=0.2,time_varying=True,split_input=True):
        super(VariableSelection, self).__init__()
        self.units = units
        self.time_varying=time_varying 
        self.num_features=num_features
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            if time_varying:
                grn = tf.keras.layers.TimeDistributed(GatedResidualNetwork(units, dropout_rate))
                self.grns.append(grn)
            else:
                grn = GatedResidualNetwork(units, dropout_rate)
                self.grns.append(grn)
               

        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        
        self.softmax = tf.keras.layers.Dense(units=num_features, activation="softmax")

        self.feature_concat = tf.keras.layers.Concatenate()
        self.flat_features = tf.keras.layers.Flatten()
        self.mult = tf.keras.layers.Multiply()
        self.feat_repeat = tf.keras.layers.RepeatVector(seq_len)


    def call(self, inputs,static_input):

        v = self.flat_features(inputs)
        v = self.feature_concat([v,static_input])
        v = self.grn_concat(v)
        v = self.softmax(v)

        if self.time_varying:
            v  = self.feat_repeat(v)

        v  = tf.expand_dims(v,axis=-1)

        x = []
        if self.time_varying:
            inputs = tf.split(inputs,self.num_features,axis=2)
        else:
            inputs = tf.split(inputs,self.num_features,axis=1)

        for idx, input in enumerate(inputs):
             x.append(self.grns[idx](input))

        x = tf.stack(x, axis=2)

        out = tf.matmul(v, x, transpose_a=True)

        if self.time_varying:
            out =  tf.squeeze(out,axis=2)
        else:
            out =  tf.squeeze(out,axis=1)

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)




class CombineAndSplit(tf.keras.layers.Layer):
    """Combines the data then splits the features into an indiviudal list"""

    def __init__(self,name="combine_and_split", **kwargs):
        super(CombineAndSplit, self).__init__(name=name, **kwargs)
        self.concatenate = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, static_inputs,time_varying_known_past,time_varying_unknown_past):
        x = self.concatenate([static_inputs,time_varying_known_past,time_varying_unknown_past])
        return x

class StaticEmbeddings(tf.keras.layers.Layer):

    
    def __init__(self,proj_size,dropout_rate=0.0,name="CatEmbedding", **kwargs):
        super(StaticEmbeddings, self).__init__(name=name, **kwargs)
        self.static_context_variable_selection_emb = GatedResidualNetwork(proj_size, dropout_rate)
        self.static_context_enrichment_emb  = GatedResidualNetwork(proj_size, dropout_rate)
        self.static_context_state_h_emb  = GatedResidualNetwork(proj_size, dropout_rate)
        self.static_context_state_c_emb  = GatedResidualNetwork(proj_size, dropout_rate)


    def call(self, input):
        static_context_variable_selection = self.static_context_variable_selection_emb(input)
        static_context_enrichment = self.static_context_enrichment_emb(input)
        static_context_state_h = self.static_context_state_h_emb(input)
        static_context_state_c= self.static_context_state_c_emb(input)
        return static_context_variable_selection, static_context_enrichment,static_context_state_h,static_context_state_c

class AddAndNorm(tf.keras.layers.Layer):  
    def __init__(self,name="add_norm", **kwargs):
        super(AddAndNorm, self).__init__(name=name, **kwargs)
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, input):
        x = self.add(input)
        x = self.norm(x)
        return x


# %%
# static_inpt = tf.keras.layers.Input(shape=(4),name='static_input')
# out = StaticEmbeddings(proj_size=4)(static_inpt)
# model = tf.keras.Model([static_inpt],out,name='CombinedModel')
# model.summary()

# %%
seq_past_len=16
seq_fut_len=6
units = 8
dropout=0.1
static_inpt = tf.keras.layers.Input(shape=(4),name='static_input')
static_context_variable_selection, static_context_enrichment,\
    static_context_state_h,static_context_state_c = StaticEmbeddings(proj_size=8)(static_inpt)

time_varying_known_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,4),name='tv_known_past')
time_varying_unknown_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,8),name='tv_unknown_past')
time_varying_past  = tf.keras.layers.Concatenate()([time_varying_known_past_inpt,time_varying_unknown_past_inpt])
time_varying_known_future_inpt = tf.keras.layers.Input(shape=(seq_fut_len,4),name='tv_known_future')


out = VariableSelection(num_features=12,units=units,dropout_rate=0.2,time_varying=True)(time_varying_past,static_context_variable_selection)
history_lstm, state_h, state_c = tf.keras.layers.LSTM(units, dropout=dropout,return_state=True,\
                                    return_sequences=True,name='lstm_history')(out,initial_state=[static_context_state_h,static_context_state_c])
out = AddAndNorm()([out,history_lstm])
out = GatedResidualNetwork(units, dropout_rate=dropout)(out)
# out = tf.keras.layers.LSTM(units, dropout=dropout,return_state=False,\
#                                     return_sequences=True,name='lstm_future')(history_lstm,initial_state=[state_h, state_c])
#out = tf.keras.layers.TimeDistributed(GatedResidualNetwork(16,dropout_rate=0.2))(out)
#out = tf.keras.layers.Concatenate()(out)
model = tf.keras.Model([static_inpt,time_varying_known_past_inpt,time_varying_unknown_past_inpt,time_varying_known_future_inpt],out ,name='CombinedModel')
model.summary()




# %%

def objective(trial):

    """Optuna objective"""

    # Params
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    tft_dim = trial.suggest_categorical('tft_dim', [8, 16,32])
    dropout = trial.suggest_float('dropout', 0.05, 0.6, log=True)

    # Build model
    model = build_TFT(tft_dim=tft_dim, dropout=dropout,learning_rate=learning_rate,seq_past_len=16,seq_fut_len=6)

    #Define fit params
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,min_delta=0.0025,restore_best_weights=True)
    callbacks = [es,optuna.integration.TFKerasPruningCallback(trial, 'val_loss')]
    model.fit(train_ds,validation_data=val_ds ,callbacks= callbacks,epochs=20,verbose=False)

    #Return loss
    loss = min(model.history.history['val_loss'])

    return loss


# %%

if __name__ == '__main__':

    # Get the TF dataset
    train_ds,val_ds,test_ds = get_TF_dataset(lag=MAX_ENCODER_LENGTH,lead=MAX_PREDICTION_LENGTH)

    #Create the optuna study using TPE sampler with Hyperband pruning
    study = optuna.create_study(
        sampler=optuna.sampl        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(ers.TPESampler(seed=RANDOM_SEED, n_startup_trials=5),

            min_resource=1, max_resource=15, reduction_factor=4
        ),
    )
    study.optimize(objective, n_trials=20)

    #Fut the best model
    model = build_TFT(**study.best_trial.params)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,min_delta=0.0025,restore_best_weights=True)
    model.fit(train_ds,validation_data=val_ds ,callbacks=[es],epochs=100,verbose=True)

    #Export the model and the results
    preds = np.squeeze(model.predict(test_ds))

    dl2ml_test_idx = np.load('./data/interim/dl2ml_test_idx.npy') # reorganised predictions to match ML model index
    preds = preds[dl2ml_test_idx,:]
    model_name= 'TFT'
    export_predictions(preds,model_name )
    export_model(model,model_name,tf_model=True )
