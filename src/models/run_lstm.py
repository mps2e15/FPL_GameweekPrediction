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

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,min_delta=0.0025,restore_best_weights=True)
model.fit(train_ds,validation_data=val_ds ,callbacks=[es],epochs=100,verbose=True)

# %%
import numpy as np
preds = np.squeeze(model.predict(test_ds))
preds = preds[dl2ml_test_idx,:]
model_name= 'LSTM'
from src.models.utils import export_predictions