# -*- coding utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from preprocess import *
from models.encoder import Encoder
from models.decoder import Decoder
from progressbar import * 
import os
import pandas as pd
class train():
    def __init__(self,batch_size,vocab_size,embed_dim,units):
        self.Encoder = Encoder(batch_size,vocab_size,embed_dim,units)
        self.Decoder = Decoder(batch_size,vocab_size,embed_dim,units)
        self.batch_size = batch_size
        self.loss_obj = keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = keras.optimizers.Adam(5e-3)
        #ckpt
        self._droot = os.path.abspath(os.path.dirname(__file__))
        self.path_ckpt = os.path.join(self._droot,"ckpt")
        self.path_ckpt_encoder = os.path.join(self._droot,"ckpt/encoder/")
        self.path_ckpt_decoder = os.path.join(self._droot,"ckpt/decoder/")
        self.weight_info = self._load_weight()
    def _loss(self,y_true,logits):
        y_pred = tf.nn.softmax(logits,axis=-1)
        loss = self.loss_obj(y_true,y_pred)
        return loss
    def _accurancy(self,y_true,logits):
        y_pred =tf.argmax(tf.nn.softmax(logits,axis=-1),axis=-1) 
        accu = tf.cast(tf.equal(y_true,y_pred),dtype=tf.float32)
        accu = tf.reduce_mean(accu)
        return accu
    @tf.function
    def _train_step(self,x_input,y_input,H):
        with tf.GradientTape() as tape:
            Eo,H = self.Encoder(x_input,H)
            loss_sum = 0
            accurancy = 0
            x_char = tf.reshape(y_input[:,0],shape=(y_input.shape[0],1))
            logits,attention = self.Decoder(x_char,Eo,H)
            y_true = y_input[:,1]
            loss = self._loss(y_true,logits)
            accu = self._accurancy(y_true,logits)
            variables = self.Encoder.trainable_variables+self.Decoder.trainable_variables
            gradient = tape.gradient(loss,variables)
            self.optimizer.apply_gradients(zip(gradient,variables))
        return loss,accu
    def valuate(self,val_data):
        self.Encoder.trainable = False
        self.Decoder.trainable = False
        H = self.Encoder.initiallize_state()
        if not hasattr(self,"fix_val_data"):
            self.fix_val_data = val_data.take(1)
        for x_input,y_input in self.fix_val_data:
            val_loss,val_accu = self._train_step(x_input,y_input,H)
        else:
            self.Encoder.trainable = True
            self.Decoder.trainable = True
        return val_loss,val_accu
    def train(self,train_data,val_data,epoch,batch_per_epoch):
        for ep in range(epoch):
            batch_loss,val_loss = 0,0
            batch_accu,val_accu = 0,0
            ep_step = ep+1
            widgets = [
                'epoch{ep}/{all}:'.format(ep=ep_step,all=epoch),
                lambda x,y:"[{:03d}/{:03d}]".format(y['value'],y['max_value']),
                lambda x,y:" loss:{:.4f} accu:{:.4f}".format(batch_loss,batch_accu),
                Bar('#'),
                Timer(),
                lambda x,y:" val_loss:{:.4f} val_accu:{:.4f}".format(val_loss,val_accu),
            ]
            with ProgressBar(widgets=widgets,max_value=batch_per_epoch) as bar:
                for batch , (x_input,y_input) in enumerate(train_data.take(batch_per_epoch)):
                    batch_step = batch+1
                    if batch==1 and ep==0 and self._ECkpt:
                        self.Encoder.load_weights(self._ECkpt)
                        self.Decoder.load_weights(self._DCkpt)
                    H = self.Encoder.initiallize_state()
                    batch_loss,batch_accu = self._train_step(x_input,y_input,H)
                    bar.update(batch_step)
                val_loss,val_accu = self.valuate(val_data)
                self.check_point(ep,3,batch_loss,val_loss)
    def check_point(self,epoch,save_num,train_loss,val_loss):
        if not os.path.isdir(self.path_ckpt_decoder):
            os.makedirs(self.path_ckpt_decoder)
        if not os.path.isdir(self.path_ckpt_encoder):
            os.makedirs(self.path_ckpt_encoder)
        step = epoch
        EFP = os.path.join(self.path_ckpt_encoder,"e_weight_%d.h5"%(step%save_num))
        DFP = os.path.join(self.path_ckpt_decoder,"d_weight_%d.h5"%(step%save_num))
        self.Encoder.save_weights(EFP,overwrite=True,save_format="h5")
        self.Decoder.save_weights(DFP,overwrite=True,save_format="h5")
        self.weight_info = self.weight_info.append({
             "epoch":epoch, "epoch":epoch,'encoder_weight':EFP,"decoder_weight":DFP,"loss":train_loss.numpy(),"val_loss":val_loss.numpy()
        },ignore_index=True)
        self.weight_info = self.weight_info.drop_duplicates(
            subset=['encoder_weight'],
            keep="last"
        )
        self.weight_info.to_csv(os.path.join(self.path_ckpt,"ckpt.csv"))
        return self
    def _load_weight(self):
        path_ckpt_info = os.path.join(self.path_ckpt,"ckpt.csv")
        self._ECkpt = False
        self._DCkpt = False
        if os.path.isfile(path_ckpt_info):
            df = pd.read_csv(path_ckpt_info)
            df = df.sort_values(["val_loss"],ascending=True).reset_index(drop=True)
            ECkpt = df.loc[0]['encoder_weight']
            DCkpt = df.loc[0]['decoder_weight']
            self._ECkpt = ECkpt
            self._DCkpt = DCkpt
        else:
            df = pd.DataFrame(columns=[
                "epoch", "batch",'encoder_weight',"decoder_weight","loss","val_loss"]) 
        return df
    
            
if __name__ == "__main__":
    batch_per_epoch = 32
    epoch = train_size//(batch_per_epoch*batch_size)
    mTrain = train(batch_size,vocab_size,256,1024)
    mTrain.train(train_set,val_set,epoch,batch_per_epoch)





# embed_dim = 256
# units = 1024
# Eomodel = Encoder(batch_size,vocab_size,embed_dim,units)
# Demodel = Decoder(batch_size,vocab_size,embed_dim,units)
# H = Eomodel.initiallize_state()
# for x_input,y_input in train_set.take(1):
#     Eo,H = Eomodel(x_input,H)
    
#     for i in range(y_input.shape[-1]):
#         x_char = tf.reshape(y_input[:,i],shape=(y_input.shape[0],1))
#         # print(x_char.shape)
#         logic,attention = Demodel(x_char,Eo,H)
#         print(logic.shape,attention.shape)
    