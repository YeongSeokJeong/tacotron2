import tensorflow as tf
from tensorflow.keras.layers import *

class tacotron_conv_layer(tf.keras.layers.Layer):
    def __init__(self, filter_size, kernel_size, dropout_rate):
        super(tacotron_conv_layer, self).__init__()
        
        self.conv1d = Conv1D(filters = filter_size, 
                             kernel_size = kernel_size,
                            padding = 'same')
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
            training = keras.backend.learning_phase()
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

class Encoder(tf.keras.models.Moedel):
    def __init__(self, input_vocab_size, 
                 embedding_dim, 
                 num_of_conv_layer, 
                 filters, 
                 kernel_size,
                 rnn_unit, 
                 zoneout_prob,
                 dropout_rate = 0.1):
        super(Encoder, self).__init__()
        self.num_of_conv_layer = num_of_conv_layer
        self.char_embedding = Embedding(input_vocab_size, embedding_dim)
        self.conv_layers = [tacotron_conv_layer(filters, kernel_size, dropout_rate) for _ in range(num_of_conv_layer)]
        self.bi_lstm = Bidirectional(RNN(ZoneoutLSTMCell(rnn_unit, zoneout_prob, zoneout_prob), return_sequences = True))
        
    def cass(self, x):
        x = self.char_embedding(x)
        for i in range(self.num_of_conv_layer):
            x = conv_layers[i](x)
        x = self.bi_lstm(x)
        return x

class LocationLayer(Layer):
    def __init__(self, attention_filters, attention_location_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        self.loc_conv = Conv1D(attention_filters, attention_location_kernel_size, padding = 'same')
        self.linear = Dense(attention_dim, activation = 'linear')
        
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
        key = self.W2(memory)
        query = self.W1(query)
        
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

class PostNet(tf.keras.models.Model):
    def __init__(self, n_mel_channels, postnet_embedding_dim, kernel_size, dropout = 0.1, padding = 'same'):
        super(PostNet, self).__init__()
        self.conv_net = [tacotron_conv_layer(postnet_embedding_dim, kernel_size, dropout = dropout) for _ in range(4)]
        self.pred_mel = tacotron_conv_layer(n_mel_channels, kernel_size, dropout = dropout)
        
    def call(self, x):
        for i in range(len(self.conv_net)):
            x =tf.nn.dropout(self.conv_net[i](x), 0.5)
        output = self.pred_mel(output)
        
        return output
    
class Decoder(tf.keras.models.Model):
    def __init__(self, 
                 prenet_dim1,
                 prenet_dim2,
                 kernel_size, 
                 n_mel_channels,
                 postnet_embedding_dim,
                 rnn_unit,
                 linear_unit,
                 attention_location_filters,
                 attention_dim,
                 attention_loc_kernel_size,
                 dropout = 0.3):
        
        super(Decoder, self).__init__()
        
        self.linear = Dense(linear_unit)
        self.stop_token = Dense(1, activation = 'sigmoid')
        self.loc_attention = Loc_sensitive_Attention(attention_dim, 
                                                     attention_location_filters, 
                                                     attention_loc_kernel_size)
        self.prenet = PreNet(prenet_dim1, prenet_dim2, dropout)
    
    def call(self, x, encoder_output, lstm1_state, lstm2_state, attention_weights_cum):
        x = self.prenet(x)
        '''
        x : mel-input
        '''
        lstm2_hidden_state, lstm2_cell_state = lstm2_state 
        
        context_vector, attention_weights = self.loc_attention(encoder_output,
                                                               lstm2_hidden_state,
                                                               tf.expand_dims(attention_weights_cum, 1))
        # loc sensitive Attention
        attention_weights_cum = attention_weights_cum + attention_weights
        
        context_vector = tf.expand_dims(context_vector)
        x, lstm1_hidden_state, lstm1_cell_state= self.lstm1(tf.concat(x, context_vector), initial_state = lstm1_state)
        lstm1_state = (lstm1_hidden_state, lstm2_cell_state)
        
        x, lstm2_hidden_state, lstm2_cells_state = self.lstm2(tf.concat(x, context_vector), initial_state = lstm2_state)
        lstm2_state = (lstm2_hidden_state, lstm2_cell_state)
        
        stop_token_pred = self.stop_token(x)
        
        linear_output = self.linear(x)
        
        return linear_output, stop_token_pred, lstm1_state, lstm2_state