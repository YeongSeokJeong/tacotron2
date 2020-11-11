import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class tacotron_conv_layer(tf.keras.layers.Layer):
    def __init__(self, filter_size, kernel_size, dropout_rate, padding = 'same'):
        super(tacotron_conv_layer, self).__init__()
        
        self.conv1d = Conv1D(filters = filter_size, 
                             kernel_size = kernel_size,
                            padding = padding)
        self.dropout = Dropout(dropout_rate)
        self.batch_norm = BatchNormalization()
        self.activation = tf.keras.activations.relu
        
    def call(self, inp):
        conv_output = self.conv1d(inp)
        conv_output = self.dropout(conv_output)
        batch_output = self.batch_norm(conv_output)
        return self.activation(batch_output)
    
class ZoneoutLSTMCell(tf.keras.layers.LSTMCell):
    """LSTM cell with recurrent zoneout.
    https://arxiv.org/abs/1606.01305
    """
    def __init__(
        self,
        units: int,
        zoneout_h: float = 0,
        zoneout_c: float = 0,
        seed: int = None,
        **kwargs
    ):
        """
        """
        super().__init__(units, **kwargs)
        self.zoneout_h = zoneout_h
        self.zoneout_c = zoneout_c
        self.seed = seed

    def _zoneout(self, t, tm1, rate, training):
        dt = tf.cast(
            tf.random.uniform(t.shape, seed=self.seed) >= rate * training, t.dtype
        )
        return dt * t + (1 - dt) * tm1

    def call(self, inputs, states, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        output, new_states = super().call(inputs, states, training)
        h = self._zoneout(new_states[0], states[0], self.zoneout_h, training)
        c = self._zoneout(new_states[1], states[1], self.zoneout_c, training)
        return h, [h, c]

    def get_config(self):
        config = {
            "zoneout_h": self.zoneout_h,
            "zoneout_c": self.zoneout_c,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}

class Encoder(Model):
    def __init__(self, input_vocab_size, 
                 embedding_dim, 
                 num_of_conv_layer, 
                 filters, 
                 kernel_size,
                 rnn_unit, 
                 zoneout_prob,
                 dropout_rate = 0.1):
        super(Encoder, self).__init__()
        self.rnn_unit = rnn_unit
        self.num_of_conv_layer = num_of_conv_layer
        self.char_embedding = Embedding(input_vocab_size, embedding_dim)
        self.conv_layers = [tacotron_conv_layer(filters, kernel_size, dropout_rate) for _ in range(num_of_conv_layer)]
        self.bi_lstm = Bidirectional(RNN(ZoneoutLSTMCell(rnn_unit, zoneout_prob, zoneout_prob), return_sequences = True))
        
    def call(self, x, lstm_initialize):
        x = self.char_embedding(x)
        for i in range(self.num_of_conv_layer):
            x = self.conv_layers[i](x)
        x = self.bi_lstm(x, initial_state = lstm_initialize)
        return x
    
    def get_initialization(self, batch_size):
        zeros = tf.zeros((batch_size, self.rnn_unit), tf.float32)
        return [zeros, zeros, zeros, zeros]
        
class LocationLayer(Layer):
    def __init__(self, attention_filters, attention_location_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        self.kernel_size = attention_location_kernel_size
        self.loc_conv = tacotron_conv_layer(attention_filters, attention_location_kernel_size, 0.5, padding = 'same')
        self.linear = Dense(attention_dim, activation = 'linear')
        
#     def pad(self, x):
#         pad_size = (int(self.kernel_size)-1)//2
#         padding = tf.constant([[0,0],[pad_size, pad_size], [0,0]])
#         zero_pad = tf.pad(x, padding, 'constant')
#         return zero_pad
    
    def call(self, inp):
        inp = self.loc_conv(inp)
        oup = self.linear(inp)
        return oup


class Loc_sensitive_Attention(Layer):
    def __init__(self, attention_dim, attention_location_filters, attention_location_kernel_size):
        super(Loc_sensitive_Attention, self).__init__()
        self.W1 = Dense(attention_dim)
        self.W2 = Dense(attention_dim)
        self.V = Dense(1)
        self.location_layer = LocationLayer(attention_location_filters, 
                                            attention_location_kernel_size, 
                                            attention_dim)
        
    def get_score(self, query, memory, attention_weights_cum):
        '''
        query : (batch_size, 1, decoder_rnn_hidden_size)
        memory : (batch_size, input_length, encoder_hidden_size)
        attention_weights_cum : (batch_size, 1, attention_weight_dim)
        '''
        
        key = self.W2(memory)
        query = self.W1(tf.expand_dims(query,1))
        location_sensitive_weights = self.location_layer(attention_weights_cum)
        score = self.V(tf.tanh(query + key + location_sensitive_weights))
        return score
    
    def call(self, query, memory, attention_weights_cum):
        alignment = self.get_score(query, memory, attention_weights_cum)
        attention_weight = tf.nn.softmax(alignment)
        context_vector = attention_weight * memory
        return tf.reduce_sum(context_vector, axis = 1), attention_weight
        
        
class PreNet(Layer):
    def __init__(self, dim1, dim2, dropout):
        super(PreNet, self).__init__()
        self.fc1 = Dense(dim1, activation = 'relu')
        self.dropout1 = Dropout(dropout)
        self.fc2 = Dense(dim2, activation = 'relu')
        self.dropout2 = Dropout(dropout)
        
    def call(self, inp):
        inp = self.dropout1(self.fc1(inp))
        oup = self.dropout2(self.fc2(inp))
        return oup

class ConvNorm(Layer):
    def __init__(self, kernel_size, filter_size, padding = 'same'):
        super(ConvNorm, self).__init__()
        self.conv1d = tacotron_conv_layer(filter_size, kernel_size, dropout = 0.1)
        self.batch_norm = BatchNormalization()
    def call(self, x):
        x = self.conv1d(x)
        oup = self.batch_norm(x)
        return x
    
class PostNet(Model):
    def __init__(self, n_mel_channels, postnet_embedding_dim, kernel_size, dropout_rate = 0.1, padding = 'same'):
        super(PostNet, self).__init__()
        self.conv_net = [tacotron_conv_layer(postnet_embedding_dim, kernel_size, dropout_rate = dropout_rate) for _ in range(4)]
        self.pred_mel = tacotron_conv_layer(n_mel_channels, kernel_size, dropout_rate = dropout_rate)
        
    def call(self, x):
        for i in range(len(self.conv_net)):
            x = self.conv_net[i](x)
        output = self.pred_mel(x)
        
        return output
    
class Decoder(Model):
    def __init__(self, 
                 prenet_dim1,
                 prenet_dim2,
                 n_mel_channels,
                 rnn_unit,
                 linear_unit,
                 attention_location_filters,
                 attention_dim,
                 attention_loc_kernel_size,
                 dropout = 0.3):
        
        super(Decoder, self).__init__()
        
        self.rnn_unit = rnn_unit
        self.linear = Dense(linear_unit)
        self.stop_token = Dense(1, activation = 'sigmoid')
        self.loc_attention = Loc_sensitive_Attention(attention_dim, 
                                                     attention_location_filters, 
                                                     attention_loc_kernel_size)
        self.lstm1 = RNN(ZoneoutLSTMCell(rnn_unit), return_sequences = True, return_state = True)
        self.lstm2 = RNN(ZoneoutLSTMCell(rnn_unit), return_sequences = True, return_state = True)
        self.prenet = PreNet(prenet_dim1, prenet_dim2, dropout)
    
    def call(self, x, encoder_output, lstm1_state, lstm2_state, attention_weights_cum):
        x = self.prenet(x)
        '''
        x : mel-input
        '''
        lstm2_hidden_state, lstm2_cell_state = lstm2_state 
        attention_weights_cum = tf.expand_dims(attention_weights_cum, 1)
        context_vector, attention_weights = self.loc_attention(lstm2_hidden_state,
                                                               encoder_output,
                                                               attention_weights_cum)
        # loc sensitive Attention
        
        context_vector = tf.expand_dims(context_vector,1)
        x, lstm1_hidden_state, lstm1_cell_state= self.lstm1(tf.concat([x, context_vector], axis = -1), initial_state = lstm1_state)
        lstm1_state = (lstm1_hidden_state, lstm2_cell_state)
        
        x, lstm2_hidden_state, lstm2_cells_state = self.lstm2(tf.concat([x, context_vector], axis = -1), initial_state = lstm2_state)
        lstm2_state = (lstm2_hidden_state, lstm2_cell_state)
        
        stop_token_pred = self.stop_token(x)
        
        linear_output = self.linear(x)
        
        return linear_output, stop_token_pred, lstm1_state, lstm2_state, tf.reduce_mean(attention_weights, axis = -1)
    
    def get_initialize(self, batch_size):
        zero = tf.zeros((batch_size, self.rnn_unit), tf.float32)
        return [zero,zero],[zero,zero]