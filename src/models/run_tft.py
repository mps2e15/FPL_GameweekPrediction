# %%
import tensorflow as tf
from src.data.tf_data_utils import prepare_data
from src.models.tf_model_utils import StaticEmbeddings,VariableSelection,AddAndNorm, \
    PositionEmbedding, ContextEnrichment,GatedResidualNetwork,TransformerDecoder
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


def build_TFT(tft_units=16, dropout=0.2,learning_rate=1e-2,seq_past_len=16,add_positional_emb=True,seq_fut_len=6):
    #HISTORIC COMPONENTS
    static_inpt = tf.keras.layers.Input(shape=(4),name='static_input')
    static_context_variable_selection, static_context_enrichment,\
        static_context_state_h,static_context_state_c = StaticEmbeddings(proj_size=tft_units)(static_inpt)

    time_varying_known_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,4),name='tv_known_past')
    time_varying_unknown_past_inpt = tf.keras.layers.Input(shape=(seq_past_len,8),name='tv_unknown_past')
    time_varying_past  = tf.keras.layers.Concatenate()([time_varying_known_past_inpt,time_varying_unknown_past_inpt])

    vsn_past = VariableSelection(num_features=12,units=tft_units,dropout_rate=dropout,time_varying=True)(time_varying_past,static_context_variable_selection)
    history_lstm, state_h, state_c = tf.keras.layers.LSTM(tft_units, dropout=dropout,return_state=True,\
                                        return_sequences=True,name='lstm_history')(vsn_past,initial_state=[static_context_state_h,static_context_state_c])
    past_lstm_skip = AddAndNorm()([vsn_past,history_lstm])
    past_rep = ContextEnrichment(seq_past_len)(past_lstm_skip,static_context_enrichment)

    pre_transformer_grn = GatedResidualNetwork(tft_units, dropout_rate=dropout)
    past_rep = pre_transformer_grn(past_rep)

    #FUTURE COMPONENTS
    time_varying_known_future_inpt = tf.keras.layers.Input(shape=(seq_fut_len,4),name='tv_known_future')
    vsn_future= VariableSelection(num_features=4,seq_len=6,units=tft_units,dropout_rate=dropout,time_varying=True)(time_varying_known_future_inpt,static_context_variable_selection)
    future_lstm = tf.keras.layers.LSTM(tft_units, dropout=dropout,return_state=False,\
                                        return_sequences=True,name='lstm_future')(vsn_future,initial_state=[state_h, state_c])
    future_lstm_skip = AddAndNorm(name='future_add_norm')([vsn_future,future_lstm])
    future_rep = ContextEnrichment(seq_fut_len)(future_lstm_skip,static_context_enrichment)
    future_rep = pre_transformer_grn(future_rep)

    #COMBINED COMPONENTS
    combined_rep = tf.keras.layers.Concatenate(axis=1)([past_rep,future_rep])
    if add_positional_emb:
        combined_rep= PositionEmbedding(maxlen=seq_fut_len+seq_past_len,num_hid=tft_units)(combined_rep)
    # out = AddAndNorm(name='positional_add_norm')([combined_rep,pos_emb])


    #DECODER
    transformer_query= PositionEmbedding(maxlen=seq_fut_len,num_hid=tft_units)(future_rep)
    decoded= TransformerDecoder(embed_dim=tft_units, num_heads=4, feed_forward_dim=tft_units, dropout_rate=0.1)(combined_rep,transformer_query)
    decoder_skip = AddAndNorm(name='decoder_skip')([future_rep,decoded])
    future_grn = GatedResidualNetwork(tft_units,dropout_rate=dropout)(decoder_skip )
    decoder_skip = AddAndNorm(name='transformer_skip')([future_lstm_skip,future_grn])
    out = tf.keras.layers.Dense(1)(decoder_skip)
    model = tf.keras.Model([static_inpt,time_varying_known_past_inpt,time_varying_unknown_past_inpt,time_varying_known_future_inpt],out ,name='CombinedModel')
    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                    optimizer=opt)
    return model


def objective(trial):

    """Optuna objective"""

    # Params
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    add_positional_emb = trial.suggest_categorical('add_positional_emb', [True,False])
    tft_units = trial.suggest_categorical('tft_units', [4,8, 16])
    dropout = trial.suggest_float('dropout', 0.05, 0.6, log=True)

    # Build model
    model = build_TFT(tft_units=tft_units, dropout=dropout,learning_rate=learning_rate,\
        add_positional_emb=add_positional_emb,seq_past_len=16,seq_fut_len=6)

    #Define fit params
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,min_delta=0.0025,restore_best_weights=True)
    callbacks = [es,optuna.integration.TFKerasPruningCallback(trial, 'val_loss')]
    model.fit(train_ds,validation_data=val_ds ,callbacks= callbacks,epochs=20,verbose=True)

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

# %%