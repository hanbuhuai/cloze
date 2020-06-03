# -*- coding utf-8 -*-
import tensorflow as tf
from tensorflow import keras
class buhadanoAttention(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.EoDense = keras.layers.Dense(10)
        self.HDense = keras.layers.Dense(10)
        self.scoreDense = keras.layers.Dense(1)
    def call(self,Eo,H):
        # Eo shape (128,32,1024) H shape(128,1024)
        Eo_d = self.EoDense(Eo)
        H_d = self.HDense(H)
        # Eo shape (128,32,10) H shape(128,10)
        H_d_expand = tf.expand_dims(H_d,axis=1)
        EoH = tf.nn.tanh(tf.add(Eo_d,H_d_expand))
        #EoH shape(128,32,10)
        score = self.scoreDense(EoH)
        #EoH shape(128,32,1)
        weight = tf.nn.softmax(score,axis=1)
        #weight shape(128,32,1)
        context = tf.reduce_sum(Eo*weight,axis=1)
        #context shape(128,32,1024)->[reduce_sum]->shape(128,32)
        return context,weight






