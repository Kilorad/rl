#содержит 2-й автоэнкодер для сжатия последовательностей
import sys
import pylab
import random
import re
import numpy as np
import pandas as pd
from collections import deque
from sklearn.metrics import pairwise
import os
import psutil
import random 

import matplotlib.pyplot as plt
sys.path.append('./common/')
import networkx as nx
import itertools
import utils
import copy

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D,Conv3D,Conv2D,TimeDistributed,Flatten, Reshape
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Permute, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Dense, Dropout, GRU, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

import pickle


class AutoencoderRLWrapper2:
    def __init__(self, state_size, action_size, learn_size=4000):
        self.render = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.learn_size = learn_size
        self.batch_size = 150
        self.cancel_train = False
        self.childhood = True #эта переменная меняется вслед (через такт) за cancel_train. И означает, что низлежащему RL можно учиться нормально
        self.reset()
        
    def reset_keras(self):
        K.clear_session()       
    def reset(self):
        # create replay memory using deque
        self.s = deque(maxlen=self.learn_size)
        self.a = deque(maxlen=self.learn_size)
        self.r = deque(maxlen=self.learn_size)
        self.done = deque(maxlen=self.learn_size)
        #учебная память
        self.x_sec_len = 20 #для secondary автоэнкодера
        self.period_sec = 1
        self.embedding_size = 6
        self.learning_rate = 0.001
        #
        self.scale_r = 500
        self.autoencoder_secondary_network = self.make_autoencoder_network()
        self.is_fit=False
    def make_autoencoder_network(self):
        dropout = 0.2
        x_shape = self.state_size + 1
        model = Sequential()
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Dense(700,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)), input_shape=(self.x_sec_len, x_shape)))
        model.add(TimeDistributed(Dropout(rate=dropout)))
        model.add(BatchNormalization())
        model.add(GRU(100, return_sequences=False,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(rate=dropout))
        model.add(BatchNormalization())
        #layer 7:
        #model.add(GRU(self.embedding_size, return_sequences=True,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        floats_per_rnn = 2
        model.add(Dense(self.x_sec_len*floats_per_rnn, kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(rate=dropout/2))
        model.add(BatchNormalization())
        #layer 10:
        model.add(Dense(self.embedding_size, kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(rate=dropout/2))
        
        model.add(BatchNormalization())
        model.add(Dense(self.x_sec_len*floats_per_rnn, kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(rate=dropout/2))
        model.add(Reshape((self.x_sec_len,floats_per_rnn)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(GRU(200, return_sequences=True,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(rate=dropout))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Dense(x_shape,kernel_initializer='he_uniform',
                                        kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(TimeDistributed(Dropout(rate=dropout)))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.reset_keras()
        return model
    
    def make_embedding(self, state_arr,reward_arr,ravel=True, showtime=False):
        if showtime:
            start = pd.Timestamp.now()
        if not self.is_fit:
            self.train_model()
        #сделаю эмбеддинг энкодером    
        if len(state_arr)>self.x_sec_len*self.period_sec:
            state_arr = [state_arr[i] for i in np.arange(-self.x_sec_len*self.period_sec+1,1,self.period_sec)]
            reward_arr = [reward_arr[i]*self.scale_r for i in np.arange(-self.x_sec_len*self.period_sec+1,1,self.period_sec)]
        else:
            x_frames_after_periodizing = int(len(state_arr)/self.period_sec)
            state_arr = [state_arr[i] for i in np.arange(-x_frames_after_periodizing*self.period_sec+1,1,self.period_sec)]
            reward_arr = [reward_arr[i]*self.scale_r for i in np.arange(-x_frames_after_periodizing*self.period_sec+1,1,self.period_sec)]   
        if len(state_arr)<self.x_sec_len:
            #если не хватает ряда, дорисую нулями
            black_square = np.zeros(self.state_size+1)
            how_many_zeros_add = self.x_sec_len-len(state_arr)
            if np.shape(state_arr)[0]>0:
                state_arr = np.hstack((np.array(state_arr)[:,0,:], np.array(reward_arr,ndmin=2).T))
                state_arr = np.vstack((np.array([black_square]*how_many_zeros_add), state_arr))
            else:
                state_arr = np.array([black_square]*how_many_zeros_add)
        else:
            state_arr = np.hstack((np.array(state_arr)[:,0,:], np.array(reward_arr,ndmin=2).T))
        if not self.is_fit:
            embedding = np.zeros((1,self.embedding_size))
        else:
            state_arr = np.reshape(state_arr,(1,self.x_sec_len,self.state_size+1))
            embedding = self.encoder_secondary_network.predict(state_arr)
            embedding = np.reshape(embedding,(1,self.embedding_size))
        if showtime:
            print(pd.Timestamp.now() - start)
        return embedding   
    def decode(self, embedding):
        layer_name = self.autoencoder_secondary_network.layers[9].name
        layer_name_end = self.autoencoder_secondary_network.layers[-1].name
        decoder_network = Model(inputs=self.autoencoder_secondary_network.get_layer(layer_name).input, 
                                              outputs=self.autoencoder_secondary_network.get_layer(layer_name_end).output)
        s_decoded = decoder_network.predict(embedding)
        return s_decoded
    def add_deeper_rl(self,rl_agent_object):
        self.deeper_rl = rl_agent_object
        self.deeper_rl.reset() #эта функция должна быть у любого rl
    
    def get_action(self, state, verbose=0):
        embedding = self.make_embedding(self.s,self.r)
        return self.deeper_rl.get_action(embedding,verbose)
    
    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.s.append(state)
        self.r.append(reward)
        self.done.append(done)
        embedding = self.make_embedding(self.s,self.r)
        self.deeper_rl.append_sample(embedding, action, reward, next_state, done)
    
    def update_target_model(self):
        self.train_model(epochs=20,sub_batch_size=6000,verbose=0)
        if not self.childhood:
            self.deeper_rl.update_target_model()
        
    def make_screenshot(self, frame_number=-1, filename=None):
        if filename is None:
            plt.imshow(self.s[frame_number])
            plt.show()
            
    def train_model(self,epochs=4,sub_batch_size=None,verbose=0,screen_curious=True):
        #сжимало адаптивное
        if not self.cancel_train:
            s = np.array(self.s, dtype=np.float32)
            if np.shape(self.s)[0]<=self.x_sec_len*2:
                return
            r = np.array(self.r, dtype=np.float32, ndmin=2).T
            r *= self.scale_r
            self.sar_arr_x = []
            self.sar_arr_y = []
            s = s[:,0,:]
            sar = np.hstack([s,r])
            if epochs<5:
                count_sequences = 20
                epochs = 4
            else:
                count_sequences = 950
                epochs = 300
            start_arr = (np.random.rand(count_sequences)*(len(self.s) - 2*self.x_sec_len)).astype(int)
            for start in start_arr:
                self.sar_arr_x.append(sar[start:start+self.x_sec_len])
                self.sar_arr_y.append(sar[start+self.x_sec_len:start+2*self.x_sec_len])
            if epochs>5:
                print('fit 2nd autoencoder model',np.shape(self.sar_arr_x),np.shape(self.sar_arr_y))
            self.autoencoder_secondary_network.fit(np.array(self.sar_arr_x),np.array(self.sar_arr_y), batch_size=self.batch_size,
                               epochs=epochs, verbose=False)
            layer_name = self.autoencoder_secondary_network.layers[10].name
            self.encoder_secondary_network = Model(inputs=self.autoencoder_secondary_network.input, 
                                                  outputs=self.autoencoder_secondary_network.get_layer(layer_name).output)
            sar_arr_pred = self.autoencoder_secondary_network.predict(np.array(self.sar_arr_x))
            rel_err = np.mean(np.abs(self.sar_arr_y - sar_arr_pred))/np.mean(np.abs(self.sar_arr_y))
            if epochs>5:
                print('relative error of 2nd autoencoder',rel_err)
            if ((len(self.s)>=900) and (rel_err<0.19))  or ((len(self.s)>=2300) and (rel_err<0.31)):
                self.cancel_train=True
        else:
            if self.childhood:
                print('CHILDHOOD IS OVER')
                self.childhood = False
                self.deeper_rl.reset(partial=True)
        #значит дообучаем низовую модель
        if self.childhood:
            epochs = 3
            #учим, но слабенько, чтобы потом не переучивать на новый язык
        self.deeper_rl.train_model(epochs=epochs, sub_batch_size=sub_batch_size,screen_curious=screen_curious)
        self.is_fit=True
