import gym
import sys
import pylab
import random
import aa_gun
import numpy as np
from collections import deque
import keras
from keras.layers import Dense, Dropout, Conv2D,Conv3D,MaxPooling3D, MaxPooling2D, Input, concatenate,Flatten
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
sys.path.append('./common/')
import utils
import operator


# double SARSA agent (sr+sar)
class SarsaConvolAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.995
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 9000
        self.sub_batch_size=500
        self.train_start = 3000
        self.reward_part_need = 0.1
        self.planning_horison = 210
        # create replay memory using deque
        memsize=10000
        self.memory = deque(maxlen=memsize)

        # create main model and target model
        self.model_sr = self.build_model('sr')
        self.model_sar = self.build_model('sar')
        
        self.s=deque(maxlen=memsize)
        self.a=deque(maxlen=memsize)
        self.r=deque(maxlen=memsize)

        # initialize target model
        self.train_model(epochs=1)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self,type):
        inputIMG = Input(shape=self.state_size)
        img_mod=Conv2D(filters=8, kernel_size=(4,4), strides=(2, 2), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(inputIMG)
        img_mod=BatchNormalization()(img_mod)
        img_mod=Dropout(rate=0.5)(img_mod)
        img_mod=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(img_mod)
        img_mod = Flatten()(img_mod)
        img_model_compiled = Model(inputs=inputIMG, outputs=img_mod)
        if type=='sar':
            # the first branch operates on the first input
            inputActions = Input(shape=(self.action_size,))
            act_mod=Dense(50, activation='relu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.001))(inputActions)
            act_mod=BatchNormalization()(act_mod)
            act_mod=Dropout(rate=0.5)(act_mod)
            act_mod = Model(inputs=inputActions, outputs=act_mod)
            combined = concatenate([img_model_compiled.output, act_mod.output])
        else:
            combined=img_mod

        z = Dense(160, activation="relu",kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.001))(combined)
        z=BatchNormalization()(z)
        z=Dropout(rate=0.5)(z)
        z = Dense(1, activation="linear",kernel_regularizer=keras.regularizers.l2(0.001))(z)
        
        if type=='sar':
            model = Model(inputs=[img_model_compiled.input, act_mod.input], outputs=z)
        elif type=='sr':
            model = Model(inputs=img_model_compiled.input, outputs=z)
            
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    # get action from model using epsilon-greedy policy
    def get_action(self, state, verbose=0,extended=0):
        if np.random.rand() <= self.epsilon or len(self.r)<self.train_start:
            return random.randrange(self.action_size)
        else:
            #Перебрать все A, для них предсказать дельта R
            r_predict_array = []
            for a in range(self.action_size):
                a_one_hot = np.zeros(self.action_size)
                a_one_hot[a]=1
                sa_current=np.concatenate((state[0,:],a_one_hot))
                sa_current=np.array(sa_current,ndmin=2)
                #sar-модель работает с дельта r
                r_predict_array.append(self.model_sar.predict(sa_current)[0][0])
            if verbose:
                print('r_predict_array',r_predict_array)
            return np.argmax(r_predict_array)

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        a_one_hot = np.zeros(self.action_size)
        a_one_hot[action]=1
        
        self.s.append(state)
        self.a.append(a_one_hot)
        self.r.append(reward)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def make_discounted_rewards(self):
        self.r_disco = utils.exp_smooth(np.array(self.r),self.discount_factor,self.planning_horison)

    # pick samples randomly from replay memory (with batch_size)
    def rebalance_data(self,s,action,reward):
        mean = np.mean(reward)
        idx_big = np.array(reward)>mean
        if any(idx_big):
            idx_big_num = np.where(idx_big)[0]
            s_add = operator.itemgetter(*idx_big_num)(self.s)
            action_add = operator.itemgetter(*idx_big_num)(self.a)
            reward_add = operator.itemgetter(*idx_big_num)(self.r)

            initial_part = np.mean(idx_big)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1

            #action.extend([action_add]*multiplication_coef)
            #reward.extend([reward_add]*multiplication_coef)
            for i in range(multiplication_coef):
                s.append(s_add)
                action.append(action_add)
                reward.append(reward_add)
            #print('initial_part',initial_part,'mean',mean,'multiplication_coef',multiplication_coef)
        #аугментировать по экшнам. Сделать частоту каждого экшна в выборке >= 0.3/число экшнов
        return (s,action,reward)
    def update_target_model(self):
        self.train_model(epochs=40,sub_batch_size=9000,verbose=0)
        self.train_model(epochs=1,sub_batch_size=9000,verbose=1)
    def train_model(self,epochs=1,sub_batch_size=None,verbose=0):
        if len(self.r) < self.train_start:
            return
        if sub_batch_size is None:
            sub_batch_size = self.sub_batch_size
        self.make_discounted_rewards()
        batch_size = max(self.batch_size,sub_batch_size)
        batch_size = min(batch_size, len(self.r))
        for i in range(6):
            mini_batch = np.random.randint(low=0,high=len(self.r),size=len(self.r))
            #я хочу, чтобы в батч попали награды, чтобы было, за что цепляться
            r = self.r_disco[mini_batch]
            if np.max(r)!=np.min(r):
                break
        s = operator.itemgetter(*mini_batch)(self.s) 
        #s = self.s[mini_batch]
        #a = self.a[mini_batch]
        a = operator.itemgetter(*mini_batch)(self.a) 
        #    
        (s,a,r) = self.rebalance_data(s,a,r)
        #
        batch_size = min(sub_batch_size, len(self.r))
        mini_batch = np.random.randint(low=0,high=len(self.s),size=batch_size)
        s = list(operator.itemgetter(*mini_batch)(s))
        a = list(operator.itemgetter(*mini_batch)(a))
        r=r[mini_batch]

        if len(self.r) < self.train_start*1.05:
            epochs*=10
        #print(s)
        #print(s.shape)
        print(s[0].shape,s[2].shape,s[20].shape)
        s=np.concatenate(s[:])
        print(s.shape)
        self.model_sr.fit(s, r, batch_size=self.batch_size,
                       epochs=epochs, verbose=verbose)
        #проверка на неправильные s
        for i in len(s):
            print(i,s[i].shape)
            if i>0 and s[i].shape!=s[i-1].shape:
                print('________________________________________')
        #
        r_sr_predicted = self.model_sr.predict(s)
        #Предсказать дельту
        delta_r = r-r_sr_predicted
        self.model_sar.fit([s,np.concatenate(a)], delta_r, batch_size=self.batch_size,
                       epochs=epochs, verbose=verbose)