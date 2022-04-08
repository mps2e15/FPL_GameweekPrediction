# %%
import tensorflow as tf
from src.data.tf_data_utils import prepare_data
from src.models.tf_model_utils import LSTM_Decoder,LSTM_Encoder
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


# %%
def build_LSTM(lstm_dim=32, dropout=0.2,learning_rate=1e-2,seq_past_len=16,seq_fut_len=6):

    #Define inputs
    static_inpt = tf.keras.layers.Input(shape=(4),name='static_input')
    time_varying_known_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,4),name='tv_known_past')
    time_varying_unknown_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,8),name='tv_unknown_past')
    time_varying_known_future_inpt = tf.keras.layers.Input(shape=(seq_fut_len,4),name='tv_known_future')

    #Encoder 
    encoded_outputs = LSTM_Encoder(seq_len=16, lstm_dim=lstm_dim, dropout=dropout)([static_inpt,time_varying_unknown_past_inpt,time_varying_unknown_past_inpt])

    #Decoder
    out = LSTM_Decoder(seq_len=6, lstm_dim=lstm_dim, dropout=dropout)(time_varying_known_future_inpt,encoded_outputs)

    #Define and compile
    model = tf.keras.Model([static_inpt,time_varying_known_past_inpt,time_varying_unknown_past_inpt,time_varying_known_future_inpt],out,name='CombinedModel')
    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                    optimizer=opt)

    return model

def objective(trial):

    """Optuna objective"""

    # Params
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    lstm_dim = trial.suggest_categorical('lstm_dim', [8, 16,32])
    dropout = trial.suggest_float('dropout', 0.05, 0.6, log=True)

    # Build model
    model = build_LSTM(lstm_dim=lstm_dim, dropout=dropout,learning_rate=learning_rate,seq_past_len=16,seq_fut_len=6)

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
    model = build_LSTM(**study.best_trial.params)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,min_delta=0.0025,restore_best_weights=True)
    model.fit(train_ds,validation_data=val_ds ,callbacks=[es],epochs=100,verbose=True)

    #Export the model and the results
    preds = np.squeeze(model.predict(test_ds))

    dl2ml_test_idx = np.load('./data/interim/dl2ml_test_idx.npy') # reorganised predictions to match ML model index
    preds = preds[dl2ml_test_idx,:]
    model_name= 'LSTM'
    export_predictions(preds,model_name )
    export_model(model,model_name,tf_model=True )
