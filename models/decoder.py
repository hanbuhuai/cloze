# -*- coding utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from .attention import buhadanoAttention
class Decoder(keras.Model):
    def __init__(self,batch_size,vocab_size,embed_dim,units,*args, **kwargs):
        super(Decoder,self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.units = units
        self.Attention = buhadanoAttention()
        self.Embeding = keras.layers.Embedding(vocab_size,embed_dim)
        self.Gru = keras.layers.GRU(units=units,recurrent_initializer="glorot_uniform")
        self.vocabDense = keras.layers.Dense(vocab_size)
    def call(self,x_char,Eo,H):
        """
        args
            x_char : tensor shape=(batch_size,1)
            Eo     : tensor shape=(batch_size,timestep,encoding_units)
            H      : tensor shape=(batch_size,encoder_units)
        returns
            output : tensor shape=(batch_size,decoder_units)  
        """
        #shape x_char (128,1) H(128,1024)
        embed_input_char = self.Embeding(x_char)
        #shape embed_input_char (128,1,256)
        context,attention_weight = self.Attention.call(Eo,H)
        #shape context (128,1024) attention_weight(128,32,1)
        context_expand = tf.expand_dims(context,axis=1)
        #shape context_expand (128,1,256)
        combind_x = tf.concat([embed_input_char,context_expand],axis=-1)
        #con_cate shape (128,1,1024+256)
        gru_logic = self.Gru(combind_x)
        #gru_logic shape (128,1024)
        vocab_logic = self.vocabDense(gru_logic)
        #vocab_logic shape (128,vocab_size)
        return vocab_logic,attention_weight

    