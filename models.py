import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, GRU
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM
    
    
class CNNModel(tf.keras.Model):
    def __init__(self, embedding_layer):
        super(CNNModel, self).__init__()
        self.embedding_layer = embedding_layer
        self.conv1d_1 = Conv1D(filters=16, kernel_size=3, activation='relu')
        self.maxpooling1d = GlobalMaxPooling1D()
        self.conv1d_2 = Conv1D(filters=16, kernel_size=3, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(100, activation='relu')
        self.dense2 = Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.conv1d_1(x)
        x = self.maxpooling1d(x)
        x = self.conv1d_2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
           

class LSTMModel(tf.keras.Model):
    def __init__(self, embedding_layer):
        super(LSTMModel, self).__init__()
        self.embedding_layer = embedding_layer
        self.lstm_layer = LSTM(32, return_sequences=True)
        self.dropout_layer = Dropout(0.2)
        self.flatten_layer = Flatten()
        self.dense_layer1 = Dense(32, activation='sigmoid')
        self.dense_layer2 = Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.lstm_layer(x)
        x = self.dropout_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_layer1(x)
        return self.dense_layer2(x)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embedding_matrix):
        super().__init__()
        self.token_emb = layers.Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                                embedding_matrix.shape[1], # or EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=200,
                                trainable=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embedding_matrix.shape[1], trainable=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
