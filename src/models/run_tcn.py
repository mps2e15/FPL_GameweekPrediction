# %%
import tensorflow as tf
from tcn import TCN
from src.data.tf_data_utils import prepare_data
from src.models.tf_model_utils import SequenceFlattener, TimeDistributedDense
from src.models.utils import export_predictions,export_model
from src.configs.model_config import MAX_ENCODER_LENGTH,MAX_PREDICTION_LENGTH,RANDOM_SEED
import numpy as np
import optuna

def get_TF_dataset(lag,lead):

    """Function to load TF dataset"""

    file_list = tf.io.matching_files('./data/processed/TF_records/train/*.tfrecord')
    train_ds = prepare_data(file_list,lag=lag,lead=lead,batch_size=32,train=True).cache()

    file_list = tf.io.matching_files('./data/processed/TF_records/val/*.tfrecord')
    val_ds = prepare_data(file_list,lag=lag,lead=lead,batch_size=128,train=False).cache()

    file_list = tf.io.matching_files('./data/processed/TF_records/test/*.tfrecord')
    test_ds = prepare_data(file_list,lag=lag,lead=lead,batch_size=128,train=False).cache()

    return train_ds,val_ds,test_ds


class TCN_Encoder(tf.keras.layers.Layer):
    """TCN Encoder"""

    def __init__(self,seq_len=16, tcn_dim=16,kernel_size=3,dilations=3,dropout=0.2,name="TCN_Encoder", **kwargs):
        super(TCN_Encoder, self).__init__(name=name, **kwargs)
        self.static_repeater = tf.keras.layers.RepeatVector(seq_len)
        self.concatenate_past = tf.keras.layers.Concatenate(name='concat_past')
        self.tcn = TCN(nb_filters=tcn_dim,kernel_size=kernel_size, dilations=[2**i for i in range(dilations)],dropout_rate=dropout,name='tcn_enc')

    def call(self, inputs):
        x = self.static_repeater(inputs[0])
        x = self.concatenate_past([x]+inputs[1:])
        return self.tcn(x)

class TCN_Decoder(tf.keras.layers.Layer):
    """TCN Decoder"""

    def __init__(self,seq_len=16,name="TCN_Decoder", **kwargs):
        super(TCN_Decoder, self).__init__(name=name, **kwargs)
        self.state_repeater = tf.keras.layers.RepeatVector(seq_len)
        self.concatenate_future = tf.keras.layers.Concatenate(name='concat_future')
        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))

    def call(self, future_inputs,encoder_outputs):
        x = self.state_repeater(encoder_outputs)
        x = self.concatenate_future([x,future_inputs])
        return self.out(x)

def build_TCN(tcn_dim=32, dilations=4,kernel_size=3, dropout=0.2,learning_rate=1e-2,seq_past_len=16,seq_fut_len=6):

    #Define inputs
    static_inpt = tf.keras.layers.Input(shape=(4),name='static_input')
    time_varying_known_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,4),name='tv_known_past')
    time_varying_unknown_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,8),name='tv_unknown_past')
    time_varying_known_future_inpt = tf.keras.layers.Input(shape=(seq_fut_len,4),name='tv_known_future')

    #Encoder 
    encoded_outputs = TCN_Encoder(seq_len=seq_past_len, tcn_dim=tcn_dim,dilations=dilations,kernel_size=kernel_size, dropout=dropout)([static_inpt,time_varying_unknown_past_inpt,time_varying_unknown_past_inpt])

    #Decoder
    out = TCN_Decoder(seq_len=seq_fut_len)(time_varying_known_future_inpt,encoded_outputs)

    #Define and compile
    model = tf.keras.Model([static_inpt,time_varying_known_past_inpt,time_varying_unknown_past_inpt,time_varying_known_future_inpt],out,name='CombinedModel')
    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                    optimizer=opt)

    return model



# %%
def objective(trial):

    """Optuna objective"""

    # Params
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    tcn_dim = trial.suggest_categorical('tcn_dim', [8, 16,32])
    dropout = trial.suggest_float('dropout', 0.05, 0.6, log=True)
    dilations = trial.suggest_int('dilations', 2,4, log=True)
    kernel_size = trial.suggest_categorical('kernel_size', [2,3])

    # Build model
    model = build_TCN(tcn_dim=tcn_dim,dilations=dilations, kernel_size=kernel_size,dropout=dropout,learning_rate=learning_rate,seq_past_len=16,seq_fut_len=6)

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
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=5),
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=15, reduction_factor=4
        ),
    )
    study.optimize(objective, n_trials=20)

    #Fut the best model
    model = build_TCN(**study.best_trial.params)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,min_delta=0.0025,restore_best_weights=True)
    model.fit(train_ds,validation_data=val_ds ,callbacks=[es],epochs=100,verbose=True)

    #Export the model and the results
    preds = np.squeeze(model.predict(test_ds))

    dl2ml_test_idx = np.load('./data/interim/dl2ml_test_idx.npy') # reorganised predictions to match ML model index
    preds = preds[dl2ml_test_idx,:]
    model_name= 'TCN'
    export_predictions(preds,model_name )
    export_model(model,model_name,tf_model=True )

