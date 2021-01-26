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
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

import pickle

_IMAGE_NET_TARGET_SIZE = (224, 224)
MAX_LENGTH = 40


class AutoencoderRLWrapper:
    def __init__(self, state_size, action_size, encoder_layers_count, path_to_network, x_path,y_path, p_add_to_learn=0.0, learn_size=500, x_len=6,y_len=6, period=8,learn_size_raw=100,learn_size_preprocessed=1000,root=''):
        #на входе в автоэнкодер: последовательность из x_len кадров (разный 8-й, если period=8), на выходе - последний из y_len-последовательности
        self.root=root
        self.render = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        
        # create replay memory using deque
        self.period = period
        deque_size = (x_len + y_len)*period
        self.s = deque(maxlen=deque_size)
        self.done = deque(maxlen=deque_size)
        
        #учебная память
        self.x_len = x_len
        self.y_len = y_len
        self.p_add_to_learn = p_add_to_learn #вероятность того, что мы конец self.s запишем в учебную память
        self.learn_size = learn_size
        self.learn_sequences_raw = deque(maxlen=learn_size_raw)
        self.learn_sequences_preprocessed = deque(maxlen=learn_size_preprocessed)
        
        #
        self.encoder_layers_count = encoder_layers_count
        self.path_to_network = path_to_network
        #Это должен быть list!
        self.x_path = x_path
        self.y_path = y_path
        if os.path.isfile(self.path_to_network):
            self.autoencoder_network = self.make_main_autoencoder_network()
            self.autoencoder_network.load_weights(self.path_to_network)
            layer_name = self.autoencoder_network.layers[31].name
            self.encoder_network = Model(inputs=self.autoencoder_network.input, 
                                              outputs=self.autoencoder_network.get_layer(layer_name).output)
        
    def reset_keras(self):
        K.clear_session()       
    def decode(self, embedding):
        layer_name = self.autoencoder_network.layers[31].name
        layer_name_end = self.autoencoder_network.layers[-1].name
        decoder_network = Model(inputs=self.autoencoder_network.get_layer(layer_name).input, 
                                              outputs=self.autoencoder_network.get_layer(layer_name_end).output)
        s_decoded = decoder_network.predict(embedding)
        return s_decoded    
    def make_main_autoencoder_network(self):
        x_subseq_len = self.x_len
        y_subseq_len = self.y_len
        
        droprate = 0.10
        model = Sequential()
        model.add(TimeDistributed(BatchNormalization()))
        model.add(
          TimeDistributed(
              Conv2D(39, (5, 5), padding='same', strides = 1, kernel_regularizer=keras.regularizers.l2(0.0001)),
              input_shape=(None, 224, 224, 3)))
        model.add(Activation('relu'))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same', strides = 2)))

        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2D(72, (5, 5), padding='same', strides = 1,kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(Dropout(droprate))
        model.add(Activation('relu')) 
        model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same', strides = 2)))

        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2D(75, (5, 5), padding='same', strides = 1, kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(Dropout(droprate))
        model.add(Activation('relu'))


        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2D(120, (4, 4), padding='same', strides = 1,kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(Dropout(droprate))
        model.add(Activation('relu'))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same', strides = 2)))  

        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2D(126, (4, 4), padding='same', strides = 1,kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(Dropout(droprate))
        model.add(Activation('relu')) 


        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2D(245, (7, 7), padding='same', strides = 4,kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(Dropout(droprate))
        model.add(Activation('relu'))  

        filters3d = 64
        depth = 3
        model.add(BatchNormalization())
        #[x_seq,7,7,256]
        model.add(Conv3D(filters3d, (4,3,3), padding='same', strides=(depth,1,1), kernel_regularizer=keras.regularizers.l2(0.0001)))
        #[x_seq/3,7,7,filters3d]
        model.add(Dropout(droprate))
        model.add(Activation('relu')) 

        filters3d_new = 26
        model.add(BatchNormalization())
        #сжатие для решейпа
        model.add(Conv3D(filters3d_new, (int(np.ceil(x_subseq_len/depth)),3,3), padding='same', strides=(x_subseq_len,1,1), kernel_regularizer=keras.regularizers.l2(0.0001)))
        #[1,7,7,filters3d_new]
        #model.add(Dropout(droprate))
        model.add(Activation('relu')) 

        #поворот для решейпа
        model.add(Reshape((7,7,filters3d_new)))

        filters = 500
        kernel_size = (16,16)
        strides = (1,1)
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(input_shape=(7,7,x_subseq_len*2),filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(droprate))
        model.add(Activation('relu')) 

        filters = 260
        kernel_size = (7,7)
        strides = (4,4)
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(droprate))

        filters = 260
        kernel_size = (4,4)
        strides = (2,2)
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(droprate))

        filters = 180
        kernel_size = (5,5)
        strides = (1,1)
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(droprate))

        filters = 150
        kernel_size = (5,5)
        strides = (1,1)
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(droprate))

        filters = 150
        kernel_size = (5,5)
        strides = (2,2)
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(droprate))

        filters = 125
        kernel_size = (5,5)
        strides = (2,2)
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(droprate))

        filters = 3
        kernel_size = (1,1)
        strides = (1,1)
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
        #model.compile(loss=keras.losses.mean_squared_error,
        #              optimizer=Adam(learning_rate=0.0001))
        #Какой взять коэффициент? Я хочу, чтобы 10 от 5 отличалось (по логике mape), как 100 от 150 (по mse). 100 от 150 по mse отличается где-то на 2500, а по mape на 5 от 10 отличается на 1.
        #Очень дохрена. Пусть будет 200 к 1
        model.compile(loss=[keras.losses.mean_squared_error,keras.losses.mape],
                    loss_weights=[1,200] , optimizer=Adam(learning_rate=0.001))

        with open(self.root + self.x_path[0], 'rb') as f:
            x_movies = pickle.load(f)
        with open(self.root + self.y_path[0], 'rb') as f:
            y_movies = pickle.load(f)
        self.reset_keras()
        x_movies = x_movies[-2:]
        y_movies = y_movies[-2:]
        model.fit(
            x_movies,
            y_movies,
            epochs=1,
            verbose=2,
            batch_size=1
        )
        return model
        
    def make_embedding(self, state_arr,ravel=True, showtime=False):
        if showtime:
            start = pd.Timestamp.now()
        #сделаю эмбеддинг энкодером
        #если не тот размер картинки, то отмасштабирую и докрашу чёрным лишнее
        #print("np.arange(-self.x_len*self.period,0,self.period)", np.arange(-self.x_len*self.period,0,self.period))
        if len(state_arr)>self.x_len*self.period:
            state_arr = [state_arr[i] for i in np.arange(-self.x_len*self.period+1,1,self.period)]
        else:
            x_frames_after_periodizing = int(len(state_arr)/self.period)
            state_arr = [state_arr[i] for i in np.arange(-x_frames_after_periodizing*self.period+1,1,self.period)]
        if len(state_arr)>0:
            shp = np.shape(state_arr[-1])
        else:
            shp = (0,0,3)
        delta_x = int((_IMAGE_NET_TARGET_SIZE[0]- shp[0])/2)
        delta_y = int((_IMAGE_NET_TARGET_SIZE[1]- shp[1])/2)
        for idx in range(len(state_arr)):
            if np.shape(state_arr[idx])!=(_IMAGE_NET_TARGET_SIZE[0],_IMAGE_NET_TARGET_SIZE[1],3):
                img = np.zeros((_IMAGE_NET_TARGET_SIZE[0],_IMAGE_NET_TARGET_SIZE[1],3))#чёрный квадрат
                img[delta_x:delta_x+shp[0],delta_y:delta_y+shp[1],:] = state_arr[idx]#поверх рисуем кадр
                state_arr[idx] = img
                if np.shape(state_arr[idx])!=(_IMAGE_NET_TARGET_SIZE[0],_IMAGE_NET_TARGET_SIZE[1],3):
                    print(np.shape(state_arr[idx]))
                    1/0
        if len(state_arr)<self.x_len:
            #если не хватает ряда, дорисую чёрными квадратами
            black_square = np.zeros((_IMAGE_NET_TARGET_SIZE[0],_IMAGE_NET_TARGET_SIZE[1],3))
            how_many_squares_add = self.x_len-len(state_arr)
            if np.shape(state_arr)[0]>0:
                state_arr = np.vstack((np.array([black_square]*how_many_squares_add), state_arr))
            else:
                state_arr = np.array([black_square]*how_many_squares_add)
        state_arr = np.array([state_arr])
        embedding = self.encoder_network.predict(state_arr)
        if ravel:
            shp = np.shape(embedding)
            embedding = np.reshape(embedding,(shp[0],-1))
        if showtime:
            print(pd.Timestamp.now() - start)
        return embedding

    def add_deeper_rl(self,rl_agent_object):
        self.deeper_rl = rl_agent_object
        self.deeper_rl.reset() #эта функция должна быть у любого rl
    
    def get_action(self, state, verbose=0):
        embedding = self.make_embedding(self.s)
        return self.deeper_rl.get_action(embedding,verbose)
    
    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.s.append(state)
        self.done.append(done)
        embedding = self.make_embedding(self.s)
        self.deeper_rl.append_sample(embedding, action, reward, next_state, done)
        if np.random.rand()<self.p_add_to_learn:
            #записать последовательность в self.learn_sequences_raw
            self.learn_sequences_raw.append()
    
    def update_target_model(self):
        #self.train_model(epochs=120,sub_batch_size=6000,verbose=0)
        self.deeper_rl.update_target_model()
        
    def test_model(self,model,X,Y,show_result=False):
        Y_pred = model.predict(X)
        mse = np.mean((Y_pred-Y)**2)
        if show_result:
            return mse, Y_pred
        else:
            return mse
    def make_screenshot(self, frame_number=-1, filename=None):
        if filename is None:
            plt.imshow(self.s[frame_number])
            plt.show()
            
    def train_model(self,epochs=4,sub_batch_size=None,verbose=0,screen_curious=True):
        #значит дообучаем низовую модель
        self.deeper_rl.train_model(epochs=epochs, sub_batch_size=sub_batch_size,screen_curious=screen_curious)
        if 0:
            if len(self.learn_sequences_raw)>=self.learn_size:
                #значит, пора переобучать автоэнкодер
                #
                pass
