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