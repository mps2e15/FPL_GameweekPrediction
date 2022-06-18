import tensorflow as tf

class TimeDistributedDense(tf.keras.layers.Layer):
    """Time distributed dense head for multi-step prediction"""

    def __init__(self,n_future_steps, dense_dim=32, dropout=0.2,name="distributed_dense_head", **kwargs):
        super(TimeDistributedDense, self).__init__(name=name, **kwargs)
        self.past_repeater = tf.keras.layers.RepeatVector(n_future_steps)
        self.concatenate_future = tf.keras.layers.Concatenate(name='concat_future')
        self.dropout = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout))
        self.dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dense_dim,activation="relu"))
        self.dense_output =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))

    def call(self, inputs):
        x = self.past_repeater(inputs[0])
        x = self.concatenate_future([x,inputs[1]])
        x = self.dropout(x)
        x = self.dense_layer(x)
        return self.dense_output(x)

class SequenceFlattener(tf.keras.layers.Layer):
    """Flattens sequence data"""

    def __init__(self,name="sequence_flattener", **kwargs):
        super(SequenceFlattener, self).__init__(name=name, **kwargs)
        self.flattener = tf.keras.layers.Flatten()
        self.concatenate = tf.keras.layers.Concatenate()


    def call(self, inputs):
        seq_list = []
        for x in inputs:
            x = self.flattener(x)
            seq_list.append(x)
        return self.concatenate(seq_list)



# %%
class LSTM_Encoder(tf.keras.layers.Layer):
    """LSTM Encoder"""

    def __init__(self,seq_len=16, lstm_dim=32, dropout=0.2,name="LSTM_Encoder", **kwargs):
        super(LSTM_Encoder, self).__init__(name=name, **kwargs)
        self.static_repeater = tf.keras.layers.RepeatVector(seq_len)
        self.concatenate_past = tf.keras.layers.Concatenate(name='concat_past')
        self.lstm = tf.keras.layers.LSTM(lstm_dim, dropout=dropout,return_state=True,name='lstm')

    def call(self, inputs):
        x = self.static_repeater(inputs[0])
        x = self.concatenate_past([x]+inputs[1:])
        return self.lstm(x)

class LSTM_Decoder(tf.keras.layers.Layer):
    """LSTM Decoder"""

    def __init__(self,seq_len=16, lstm_dim=32, dropout=0.2,name="LSTM_Decoder", **kwargs):
        super(LSTM_Decoder, self).__init__(name=name, **kwargs)
        self.state_repeater = tf.keras.layers.RepeatVector(seq_len)
        self.concatenate_future = tf.keras.layers.Concatenate(name='concat_future')
        self.lstm = tf.keras.layers.LSTM(lstm_dim, dropout=dropout,return_sequences=True,name='lstm')
        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))

    def call(self, future_inputs,encoder_outputs):
        x = self.state_repeater(encoder_outputs[0])
        x = self.concatenate_future([x,future_inputs])
        x = self.lstm(x,initial_state = encoder_outputs[1:])
        return self.out(x)

class GatedLinearUnit(tf.keras.layers.Layer):
    """Gated Lienar Unit"""
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = tf.keras.layers.Dense(units)
        self.sigmoid = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

class GatedResidualNetwork(tf.keras.layers.Layer):

    """Gated Residual Network"""
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

    """
    Variable Selection Network
    """
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





class StaticEmbeddings(tf.keras.layers.Layer):
    
    """Generates the four static outputs for static fusion"""
    
    def __init__(self,proj_size,dropout_rate=0.0,name="StaticEmbeddings", **kwargs):
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
    """Adds and normalises list of inputs"""

    def __init__(self,name="add_norm", **kwargs):
        super(AddAndNorm, self).__init__(name=name, **kwargs)
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, input):
        x = self.add(input)
        x = self.norm(x)
        return x

class PositionEmbedding(tf.keras.layers.Layer):
    """Generates positional embeddings for input data, adding the embeddins to the input sequence"""
    def __init__(self, maxlen=100, num_hid=8):
        super().__init__()
        self.num_hid = num_hid
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=num_hid)
        self.layernorm = tf.keras.layers.LayerNormalization()
    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        positions = self.layernorm(x+positions)
        return positions

class TransformerDecoder(tf.keras.layers.Layer):
    """
    Model for decoding input sequences (past and future)

    It assumes no causal masking as all information is available at timestep t.    
    
    """
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.self_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.enc_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ffn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )

    def call(self, enc_out, target):
        target_att = self.self_att(target, target)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm

class ContextEnrichment(tf.keras.layers.Layer):
    """
    Layer for 'enriching' the static data i.e. repeating the static data and 
    oncatenating it with the sequence inputs
    """
    def __init__(self, seq_len=16, **kwargs):
        super(ContextEnrichment,self).__init__(**kwargs)
        self.feat_repeat = tf.keras.layers.RepeatVector(seq_len)
        self.concat = tf.keras.layers.Concatenate()
    def call(self, sequence_input,static_input):
        static_repeater = self.feat_repeat(static_input)
        return self.concat([sequence_input,static_repeater])