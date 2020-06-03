# -*- coding -*-
import os,pickle,glob,random
from os import path
import tensorflow as tf
from tensorflow import keras
class dataMaker():
    def __init__(self,set_prefix="train",shuffle=True,dev=False):
        """
        args
            set_prefix : train/val
        """
        self._droot = path.abspath(path.dirname(__file__))
        self._d_dist= path.join(self._droot,"dist")
        if dev:
            self._d_dist= path.join(self._droot,"dev")
        self._record_path = path.join(self._d_dist,"record")
        #添加token
        self.token = self._load_token()
        self.pre_fix = "{set_prefix}*.record.zip".format(set_prefix=set_prefix)
        self.shuffle = shuffle
    def _load_token(self):
        token_file = path.join(self._d_dist,"token.pickle")
        with open(token_file,"rb") as fp:
            token = pickle.load(fp)
        return token
    def _unserialize_example(self,serialized):
        example = tf.io.parse_single_example(serialized,features={
            "input_x":tf.io.FixedLenFeature([32],dtype=tf.int64),
            "input_y":tf.io.FixedLenFeature([3],dtype=tf.int64)
        })
        return example['input_x'],example['input_y']
    def load_data(self,batch_size):
        fpath = path.join(self._record_path,self.pre_fix)
        f_list = glob.glob(fpath)
        data_set = tf.data.TFRecordDataset(f_list,compression_type="GZIP")
        data_set = data_set.map(self._unserialize_example)
        if self.shuffle:
            data_set.shuffle(10000)
        data_set = data_set.batch(batch_size=batch_size,drop_remainder=True)
        return data_set
    def seq2text(self,seq):
        text = self.token.sequences_to_texts(seq)
        return text

#实际运行
batch_size = 128
val_size = 6*500000+23829
train_size = 26*500000+45314
epoch_step = 26
train_model = dataMaker(set_prefix="train")
val_model = dataMaker(set_prefix="val",shuffle=False)
train_set  = train_model.load_data(batch_size)
val_set    = val_model.load_data(batch_size)
vocab_size = len(train_model.token.word_index)+1
"""
开发
"""
# batch_size = 128
# val_size   = 1033
# train_size = 4132
# epoch_step = 1
# train_model = dataMaker(set_prefix="train",dev=True)
# val_model = dataMaker(set_prefix="val",shuffle=False,dev=True)
# train_set  = train_model.load_data(batch_size)
# val_set    = val_model.load_data(batch_size)
# vocab_size = len(train_model.token.word_index)+1





        
    


        


        