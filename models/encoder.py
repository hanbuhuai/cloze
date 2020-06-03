# -*- coding utf-8 -*-
import tensorflow as tf
from tensorflow import keras
"""
vocab_size = ?
embed_dim = 256
units = 1024
"""
class Encoder(keras.Model):
    def __init__(self,batch_size,vocab_size,embed_dim,units,*args, **kwargs):
        super(Encoder,self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.units = units
        self.Embeding = keras.layers.Embedding(vocab_size,embed_dim)
        self.Gru = keras.layers.GRU(units=units,return_sequences=True,return_state=True,
        recurrent_initializer="glorot_uniform")
    def initiallize_state(self):
        return tf.zeros(shape=(self.batch_size,self.units))
    def call(self,x_input,H):
        #shape x_input (128,32) H(128,1024)
        embed_input = self.Embeding(x_input)
        #shape embed_input (128,32,256)
        Eo,H = self.Gru(embed_input,H)
        #shape Eo (128,32,1024) H (128,1024)
        return Eo,H

    