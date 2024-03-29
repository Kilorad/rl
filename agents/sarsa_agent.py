import sys
import pylab
import random
import re
import numpy as np
import pandas as pd
from collections import deque
from sklearn.metrics import pairwise
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, GRU, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
sys.path.append('./common/')
import networkx as nx
import itertools
import utils
import copy
import matplotlib.pyplot as plt

class SarsaAgent:
    def __init__(self, state_size, action_size, layers_size=[450,450],deque_len=2000):
        self.render = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        # hyper parameters for the Double SARSA
        self.discount_factor = 0.99 #0.98**15 ~ 0.75
        self.learning_rate = 0.0001
        self.epsilon_decay = 0.992
        #0.6*0.996^1000 = 0.01, то есть за 1000 тактов перестанем рандомить, а за 600 тактов сведём рандом к 10% от исходного
        self.epsilon_min = 0.01
        self.batch_size = 290
        self.sub_batch_size=1000
        self.train_start = 220
        self.reward_part_need = 0.3
        self.planning_horison = 400
        self.curiosity = 0.15
        self.layers_size = layers_size
        self.deque_len = deque_len
        self.multy_sar = 3 #в sar можно больше одного s использовать
        self.multy_sar_period = 30
        self.reset()
        self.is_fit = False
        
        
    def reset(self):
        # create replay memory using deque
        self.epsilon = 0.6
        self.s = deque(maxlen=self.deque_len)
        self.r = deque(maxlen=self.deque_len)
        #действия в формате one_hot:
        self.a = deque(maxlen=self.deque_len)
        self.done = deque(maxlen=self.deque_len)
        # SR модель - для оценки value
        self.model_sr = self.build_model('sr')
        # SAR модель - для оценки advantage
        self.model_sar = self.build_model('sar')
        # SAR модель - для оценки advantage
        self.model_sar_err = self.build_model('sar_err')
        #инициализировать модельки
        self.train_model(epochs=1)
        
    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self,type_mdl):
        if type_mdl=='sar':
            input_dim = self.state_size*self.multy_sar + self.action_size 
            out_dim = 1
        elif type_mdl=='sr':
            input_dim = self.state_size*self.multy_sar 
            out_dim = 1
        elif type_mdl=='sar_err':
            input_dim = self.state_size*self.multy_sar  + self.action_size 
            out_dim = 1
        if type_mdl=='sar_err':
            layer_size_multiplier = 1
        else:
            layer_size_multiplier = 1
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(int(self.layers_size[0]*layer_size_multiplier), input_dim=input_dim, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(rate=0.3))
        model.add(BatchNormalization())
        model.add(Dense(int(self.layers_size[1]*layer_size_multiplier), activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(rate=0.3))
        model.add(Dense(out_dim, activation='linear',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.is_fit = False
        #model.summary()
        return model
    
    
    # get action from model using epsilon-greedy policy
    def get_action(self, state, verbose=0):
        if (np.random.rand() <= self.epsilon) or (len(self.s)<self.train_start) or (self.is_fit==False):
            return random.randrange(self.action_size)
        else:
            #Перебрать все A, для них предсказать дельта R
            #r_predict_array = []
            sa_current_array = []
            for a in range(self.action_size):
                a_one_hot = np.zeros(self.action_size)
                a_one_hot[a]=1
                sa_current=state[0,:]
                for multy_num in range(1,self.multy_sar):
                    sa_current = np.concatenate((sa_current,self.s[(-multy_num)*self.multy_sar_period]))
                sa_current=np.concatenate((sa_current, a_one_hot))
                sa_current=np.array(sa_current, ndmin=2)
                #sar-модель работает с дельта r
                #r_predict_array.append(self.model_sar.predict(sa_current)[0][0])
                sa_current_array.append(sa_current)
            r_predict_array = self.model_sar.predict(np.array(sa_current_array)[:, 0, :])
            r_predict_err_array = self.model_sar_err.predict(np.array(sa_current_array)[:, 0, :])
            #наложить шум, чтобы из двух равнозначных вариантов выбирать разный
            random_ampl = np.std(r_predict_array)*0.2
            d_r_random = np.random.rand(len(r_predict_array))*random_ampl
            #и добавить любопытство
            d_r_curiosity = np.ravel(r_predict_err_array*self.curiosity)
            r_predict_array_augmented = np.ravel(r_predict_array) + d_r_random + d_r_curiosity
            if verbose:
                print('r_predict_array_augmented',r_predict_array_augmented)
            return np.argmax(r_predict_array_augmented)
    
    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        a_one_hot = np.zeros(self.action_size)
        a_one_hot[action]=1
        self.s.append(state[0,:])
        self.a.append(a_one_hot)
        self.r.append(reward)
        self.done.append(done)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def make_discounted_rewards(self):
        #derivative - брать ли за награду скорость движения к цели
        idx_borders = np.array(self.done)
        #borders - границы между эпизодами
        self.r_disco = utils.exp_smooth(self.r,self.discount_factor,self.planning_horison,idx_borders)
    
    #аугментация данных. Придётся выбросы размножить, потому что реворды очень на них завязаны
    #ну и данные выложим в виде (s,a,r_disco)
    def rebalance_data(self,s,action,reward):      
        mean = np.mean(reward)
        #mean75 = (np.mean(reward)+np.percentile(reward,0.97))*0.5
        #mean25 = (np.mean(reward)+np.percentile(reward,0.03))*0.5
        mean75 = np.mean(reward)
        mean25 = np.mean(reward)
        #размножь большие
        idx_big = reward>mean75
        if any(idx_big):
            idx_big_num = np.where(idx_big)[0]
            s_add = list(s[idx_big_num])
            action_add = list(action[idx_big_num])
            reward_add = list(reward[idx_big_num])
    
            initial_part = np.mean(idx_big)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1
    
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                action = np.concatenate((action,action_add))
                reward = np.concatenate((reward,reward_add))
        #размножь мелкие
        idx_small = reward<mean25
        if any(idx_small):
            idx_small_num = np.where(idx_small)[0]
            s_add = list(s[idx_small_num])
            action_add = list(action[idx_small_num])
            reward_add = list(reward[idx_small_num])
    
            initial_part = np.mean(idx_small)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1
    		
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                action = np.concatenate((action,action_add))
                reward = np.concatenate((reward,reward_add))
        #аугментировать по экшнам. Сделать частоту каждого экшна в выборке >= 0.3/число экшнов
        freq_arr=np.zeros(self.action_size)
        for corrections_count in range(6):
            corrections_flag=False
            for i in range(self.action_size):
                freq_arr[i]=np.mean(action[:,i]==1)
                if freq_arr[i]<0.3/self.action_size and freq_arr[i]>0:
                    corrections_flag=True
                    #размножаем
                    idx_act_num = np.where(action[:,i]==1)[0]
                    s_add = list(s[idx_act_num])
                    action_add = list(action[idx_act_num])
                    reward_add = list(reward[idx_act_num])
                    s=np.vstack((s,s_add))
                    action = np.concatenate((action,action_add))
                    reward = np.concatenate((reward,reward_add))
            corrections_count+=1
            if not corrections_flag:
                break
        return (s,action,reward)
    
    def update_target_model(self):
        self.train_model(epochs=310,sub_batch_size=6000,verbose=False)
        self.train_model(epochs=1,sub_batch_size=6000,verbose=False, draw=True)
        
    def test_model(self,model,X,Y,show_result=False):
        Y_pred = model.predict(X)
        mse = np.mean((Y_pred-Y)**2)
        if show_result:
            return mse, Y_pred
        else:
            return mse
    
    def train_model(self,epochs=4,sub_batch_size=None,verbose=False): 
        if len(self.s) < self.train_start:
            return
        self.ss = np.array(self.s)
        for multy_num in range(1, self.multy_sar):
            self.ss = np.hstack([self.ss[:-self.multy_sar_period], np.array(self.s)[multy_num*self.multy_sar_period:]])
            
        if sub_batch_size is None:
            sub_batch_size = self.sub_batch_size
        #награды - это дистанции до цели
        self.make_discounted_rewards()
        batch_size = max(self.batch_size,sub_batch_size)
        batch_size = min(batch_size, len(self.ss))
        #batch_size - это или self.batch_size, или кастомный sub_batch_size, если он больше, или длина дека, когда она меньше
        #batch_size применяется, чтобы отобрать часть экземпляров для обучения, и это же размер батча для нейронки
        len_ss = np.shape(self.ss)[0]
        a = np.array(self.a, dtype=np.float32)[:len_ss]
        r = np.array(self.r_disco, dtype=np.float32)[:len_ss]
        for i in range(6):
            mini_batch = np.random.randint(low=0,high=len(self.ss),size=len(self.ss))
            #я хочу, чтобы в батч попали награды, чтобы было, за что цепляться
            r = r[mini_batch]
            if np.max(r)!=np.min(r):
                break
        
        s = np.array(self.ss, dtype=np.float32)[mini_batch,:]
        a = np.array(a, dtype=np.float32)[mini_batch,:]
        #    
        (s,a,r) = self.rebalance_data(s,a,r)
        if np.random.rand()<0.06:
            #написать дамп в файл, потом разберу
            data = np.hstack((np.array(self.ss),np.array(np.argmax(np.array(a),axis=1),ndmin=2).T,np.array(r,ndmin=2).T,np.array(r_disco,ndmin=2).T))
            columns = []
            for i in range(len(self.ss[0])):
                columns.append('s'+str(i))

            columns.append('a')
            columns.append('r')
            columns.append('r_disco')
                
            report = pd.DataFrame(data=data,columns=columns)
            report.to_csv('report.csv')
        #Это уже минибатч. Размноженный
        if np.std(r)==0:
            #Нечего тут учить
            return
    
        if len(self.ss) < self.train_start*1.05:
            #инициализация
            verbose = True
            epochs*=2
        for i in range(3):
            if verbose:
                print('sr_fit')           
            self.model_sr.fit(s, r, batch_size=self.batch_size,epochs=epochs, verbose=verbose)
            mse = self.test_model(self.model_sr,s,r)
            if epochs==1:
                break
            if np.std(r)==0:
                break
            if mse/np.std(r)<=1: #обучать до тех пор, пока не станет хорошо
                break
        print('sr rmse', np.sqrt(mse),'sr rmse/std', np.sqrt(mse)/np.std(r), 'sr rmse/mean', np.sqrt(mse)/np.mean(np.abs(r)), 's.shape', np.shape(s))
        r_sr_predicted = self.model_sr.predict(s)
        #Предсказать дельту
        delta_r = np.array(np.ravel(r-r_sr_predicted[:,0]),ndmin=2,dtype=np.float32).T
        sa = np.hstack((s,a))
        for i in range(3):
            if verbose:
                print('sar_fit')
            self.model_sar.fit(sa, delta_r, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
            mse, Y_pred = self.test_model(self.model_sar,sa,delta_r, True)
            error_abs = np.abs(Y_pred-delta_r)
            if epochs==1:
                break
            if np.std(delta_r)==0:
                break
            if mse/np.mean(np.abs(delta_r))<=0.35: #обучать до тех пор, пока не станет хорошо
                break
        print('sar', np.sqrt(mse),' rmse/std', np.sqrt(mse)/np.std(delta_r), 'sar rmse/mean', np.sqrt(mse)/np.mean(np.abs(delta_r)))
        self.model_sar_err.fit(sa, error_abs, batch_size=self.batch_size,
                           epochs=epochs*2, verbose=verbose)
        mse = self.test_model(self.model_sar_err,sa,error_abs)
        print('sar_err', np.sqrt(mse),' rmse/std', np.sqrt(mse)/np.std(error_abs), 'sar rmse/mean', np.sqrt(mse)/np.mean(np.abs(error_abs)))
        self.is_fit = True


import xgboost as xgb
class SarsaAgentTree(SarsaAgent):
    def build_model(self,type_mdl):
        if type_mdl=='sar':
            input_dim = self.state_size*self.multy_sar  + self.action_size 
            out_dim = 1
        elif type_mdl=='sr':
            input_dim = self.state_size*self.multy_sar 
            out_dim = 1
        elif type_mdl=='sar_err':
            input_dim = self.state_size*self.multy_sar  + self.action_size 
            out_dim = 1
        if type_mdl=='sar_err':
            layer_size_multiplier = 0.5
        else:
            layer_size_multiplier = 1
            
        xgbparams = {
            'booster':'gbtree',
            'metric':'mse',
            #'objective':'reg:squarederror',
            'verbosity':0,
            'max_depth': 6,
            'n_estimators': 700,
            'eta': 0.05,
            'nthreads': 2,
            'seed':0
        }    
        model = xgb.XGBRegressor(**xgbparams)
        self.is_fit = False
        return model
    def train_model(self,epochs=4,sub_batch_size=None,verbose=0): 
        if len(self.s) < self.train_start:
            return
        if sub_batch_size is None:
            sub_batch_size = self.sub_batch_size
        #награды - это дистанции до цели
        self.make_discounted_rewards()
        batch_size = max(self.batch_size,sub_batch_size)
        batch_size = min(batch_size, len(self.s))
        #batch_size - это или self.batch_size, или кастомный sub_batch_size, если он больше, или длина дека, когда она меньше
        #batch_size применяется, чтобы отобрать часть экземпляров для обучения, и это же размер батча для нейронки
        for i in range(6):
            mini_batch = np.random.randint(low=0,high=len(self.s),size=len(self.s))
            #я хочу, чтобы в батч попали награды, чтобы было, за что цепляться
            r = self.r_disco[mini_batch]
            if np.max(r)!=np.min(r):
                break
        s = np.array(self.s, dtype=np.float32)[mini_batch,:]
        a = np.array(self.a, dtype=np.float32)[mini_batch,:]
        #    
        (s,a,r) = self.rebalance_data(s,a,r)
        if np.random.rand()<0.06:
            #написать дамп в файл, потом разберу
            data = np.hstack((np.array(self.s),np.array(np.argmax(np.array(self.a),axis=1),ndmin=2).T,np.array(self.r,ndmin=2).T,np.array(self.r_disco,ndmin=2).T))
            columns = []
            for i in range(len(self.s[0])):
                columns.append('s'+str(i))

            columns.append('a')
            columns.append('r')
            columns.append('r_disco')
                
            report = pd.DataFrame(data=data,columns=columns)
            report.to_csv('report.csv')
        #Это уже минибатч. Размноженный
        if np.std(r)==0:
            #Нечего тут учить
            return
    
        if len(self.s) < self.train_start*1.05:
            #инициализация
            verbose = True
        if verbose:
            print('sr_fit')
        eval_border = int(np.shape(s)[0]*0.75)
        self.model_sr.fit(s[:eval_border], r[:eval_border], eval_set=[(s[eval_border:], r[eval_border:])], verbose=False)
        mse = self.test_model(self.model_sr,s,r)
        print('sr rmse', np.sqrt(mse),'sr rmse/std', np.sqrt(mse)/np.std(r), 'sr rmse/mean', np.sqrt(mse)/np.mean(np.abs(r)), 's.shape', np.shape(s))
        r_sr_predicted = self.model_sr.predict(s)
        #Предсказать дельту
        delta_r = r-r_sr_predicted
        sa = np.hstack((s,a))
        self.model_sar.fit(sa[:eval_border], r[:eval_border], eval_set=[(sa[eval_border:], r[eval_border:])], verbose=False)
        mse, Y_pred = self.test_model(self.model_sar,sa,delta_r, True)
        error_abs = np.abs(Y_pred-delta_r)   
        print('sar', np.sqrt(mse),' rmse/std', np.sqrt(mse)/np.std(delta_r), 'sar rmse/mean', np.sqrt(mse)/np.mean(np.abs(delta_r)))
        self.model_sar_err.fit(sa[:eval_border], error_abs[:eval_border], eval_set=[(sa[eval_border:], error_abs[eval_border:])], verbose=False)
        mse = self.test_model(self.model_sar_err,sa,error_abs)
        print('sar_err', np.sqrt(mse),' rmse/std', np.sqrt(mse)/np.std(error_abs), 'sar rmse/mean', np.sqrt(mse)/np.mean(np.abs(error_abs)))
        self.is_fit = True
        
    # get action from model using epsilon-greedy policy
    def get_action(self, state, verbose=0):
        if (np.random.rand() <= self.epsilon) or (len(self.s)<self.train_start) or (self.is_fit==False):
            return random.randrange(self.action_size)
        else:
            #Перебрать все A, для них предсказать дельта R
            #r_predict_array = []
            sa_current_array = []
            for a in range(self.action_size):
                a_one_hot = np.zeros(self.action_size)
                a_one_hot[a]=1
                sa_current=np.concatenate((state[0,:],a_one_hot))
                sa_current=np.array(sa_current, ndmin=2)
                for multy_num in range(1,self.multy_sar):
                    sa_current = np.concatenate((sa_current,self.s[(-multy_num)*self.multy_sar_period]))
                #sar-модель работает с дельта r
                #r_predict_array.append(self.model_sar.predict(sa_current)[0][0])
                sa_current_array.append(sa_current)
            r_predict_array = self.model_sar.predict(np.array(sa_current_array).reshape([self.action_size,self.action_size+self.state_size]))
            r_predict_err_array = self.model_sar_err.predict(np.array(sa_current_array).reshape([self.action_size,self.action_size+self.state_size]))
            #наложить шум, чтобы из двух равнозначных вариантов выбирать разный
            random_ampl = np.std(r_predict_array)*0.2
            d_r_random = np.random.rand(len(r_predict_array))*random_ampl
            #и добавить любопытство
            d_r_curiosity = np.ravel(r_predict_err_array*self.curiosity)
            r_predict_array_augmented = np.ravel(r_predict_array) + d_r_random + d_r_curiosity
            if verbose:
                print('r_predict_array_augmented',r_predict_array_augmented)
            return np.argmax(r_predict_array_augmented)
        
class SarsaAgent_RND(SarsaAgent):
    def __init__(self, state_size, action_size, layers_size=[300,300],deque_len=2000):
        self.hash_size = 14
        self.hash_curiosity = 0.6#+mean награды за 1.6-кратно выбивающуюся ошибку
        super().__init__(state_size=state_size, action_size=action_size, layers_size=layers_size,deque_len=deque_len)
    def reset(self):
        self.epsilon = 0.6
        # create replay memory using deque
        self.s = deque(maxlen=self.deque_len)
        self.r = deque(maxlen=self.deque_len)
        #действия в формате one_hot:
        self.a = deque(maxlen=self.deque_len)
        self.done = deque(maxlen=self.deque_len)
        # SR модель - для оценки value
        self.model_sr = self.build_model('sr')
        # SAR модель - для оценки advantage
        self.model_sar = self.build_model('sar')
        # SAR err модель - для оценки неожиданности advantage
        self.model_sar_err = self.build_model('sar_err')
        # rand модель - для рандомного хеширования состояний
        self.model_rand = self.build_model('rand')
        # State->hash модель - для прогноза хешей, для оценки новизны локации
        self.model_sh = self.build_model('sh')
        #инициализировать модельки
        self.train_model(epochs=1)    
    def build_model(self,type_mdl):
        if type_mdl=='sar':
            input_dim = self.state_size*self.multy_sar  + self.action_size 
            out_dim = 1
        elif type_mdl=='sr':
            input_dim = self.state_size*self.multy_sar 
            out_dim = 1
        elif type_mdl=='sar_err':
            input_dim = self.state_size*self.multy_sar  + self.action_size 
            out_dim = 1
        elif type_mdl=='rand':
            input_dim = self.state_size*self.multy_sar  
            out_dim = self.hash_size
        elif type_mdl=='sh':
            input_dim = self.state_size*self.multy_sar  
            out_dim = self.hash_size
        if (type_mdl=='sar_err')or(type_mdl=='rand'):
            layer_size_multiplier = 0.5
        else:
            layer_size_multiplier = 1
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(int(self.layers_size[0]*layer_size_multiplier), input_dim=input_dim, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(rate=0.3))
        model.add(BatchNormalization())
        model.add(Dense(int(self.layers_size[1]*layer_size_multiplier), activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(Dropout(rate=0.3))
        model.add(Dense(out_dim, activation='linear',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.is_fit = False
        #model.summary()
        return model
    def train_model(self,epochs=5,sub_batch_size=None,verbose=0,draw=False,diaposon=[-200,-1],screen_curious=True):
        if len(self.s) < self.train_start:
            return
        self.ss = np.array(self.s)
        for multy_num in range(1, self.multy_sar):
            self.ss = np.hstack([self.ss[:-self.multy_sar_period], np.array(self.s)[multy_num*self.multy_sar_period:]])
        if sub_batch_size is None:
            sub_batch_size = self.sub_batch_size
        #награды - это дистанции до цели
        self.make_discounted_rewards()
        batch_size = max(self.batch_size,sub_batch_size)
        batch_size = min(batch_size, len(self.ss))
        #batch_size - это или self.batch_size, или кастомный sub_batch_size, если он больше, или длина дека, когда она меньше
        #batch_size применяется, чтобы отобрать часть экземпляров для обучения, и это же размер батча для нейронки
        len_ss = np.shape(self.ss)[0]
        a = np.array(self.a, dtype=np.float32)[:len_ss]
        r = np.array(self.r_disco_curiosity, dtype=np.float32)[:len_ss]
        for i in range(6):
            mini_batch = np.random.randint(low=0,high=len(self.ss),size=len(self.ss))
            #я хочу, чтобы в батч попали награды, чтобы было, за что цепляться
            r = r[mini_batch]
            if np.max(r)!=np.min(r):
                break
        s = np.array(self.ss, dtype=np.float32)[mini_batch,:]
        a = np.array(self.a, dtype=np.float32)[mini_batch,:]
        #    
        (s,a,r) = self.rebalance_data(s,a,r)
        if np.random.rand()<0.06:
            #написать дамп в файл, потом разберу
            a2write = np.array(np.argmax(np.array(self.a),axis=1),ndmin=2)
            a2write = a2write.T[:len(self.ss)]
            r2write = np.array(r,ndmin=2).T[:len(self.ss)]
            data = np.hstack((np.array(self.ss),a2write,r2write))
            columns = []
            for i in range(len(self.ss[0])):
                columns.append('s'+str(i))

            columns.append('a')
            columns.append('r_disco_curiosity')
            #columns.append('r_disco')
            #columns.append('r_disco_curiosity')
                
            report = pd.DataFrame(data=data,columns=columns)
            report.to_csv('report.csv')
        #Это уже минибатч. Размноженный
        if np.std(r)==0:
            #Нечего тут учить
            return
    
        if len(self.ss) < self.train_start*1.05:
            #инициализация
            verbose = True
            epochs*=2
        for i in range(5):
            if verbose:
                print('sr_fit')           
            self.model_sr.fit(s, r, batch_size=self.batch_size,epochs=epochs, verbose=verbose)
            mse = self.test_model(self.model_sr,s,np.array(np.ravel(r),ndmin=2,dtype=np.float32).T)
            if epochs==1:
                break
            if np.std(r)==0:
                break
            if mse/np.std(r)<=1: #обучать до тех пор, пока не станет хорошо
                break
        if epochs>5:
            print('sr rmse', np.sqrt(mse),'sr rmse/std', np.sqrt(mse)/np.std(r), 'sr rmse/mean', np.sqrt(mse)/np.mean(np.abs(r)), 's.shape', np.shape(s))
        r_sr_predicted = self.model_sr.predict(s)
        #Предсказать дельту
        delta_r = r-r_sr_predicted[:,0]
        delta_r = np.array(delta_r, ndmin=2).T
        sa = np.hstack((s,a))
        for i in range(7):
            if verbose:
                print('sar_fit')
            self.model_sar.fit(sa, delta_r, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
            mse, Y_pred = self.test_model(self.model_sar,sa,delta_r, True)
            Y_pred = self.model_sar.predict(sa)
            error_abs = np.abs(np.ravel(Y_pred)-np.ravel(delta_r))
            if epochs==1:
                break
            if np.std(delta_r)==0:
                break
            if mse/np.mean(np.abs(delta_r))<=0.35: #обучать до тех пор, пока не станет хорошо
                break
        if epochs>5:
            print('sar', np.sqrt(mse),' rmse/std', np.sqrt(mse)/np.std(delta_r), 'sar rmse/mean', np.sqrt(mse)/np.mean(np.abs(delta_r)))
        if draw:
            print('sar fact, frc')
            plt.plot(np.ravel(delta_r[diaposon[0]:diaposon[1]]))
            plt.plot(np.ravel(Y_pred[diaposon[0]:diaposon[1]]))
            plt.show()
        self.model_sar_err.fit(sa, error_abs, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
        error_abs = np.array(np.ravel(error_abs),ndmin=2).T
        mse, Y_pred = self.test_model(self.model_sar_err,sa,error_abs, True)
            
        if epochs>5:
            print('sar_err', np.sqrt(mse),' rmse/std', np.sqrt(mse)/np.std(error_abs), 'sar rmse/mean', np.sqrt(mse)/np.mean(np.abs(error_abs)))
        if draw:
            print('sar_err fact, frc')
            Y_pred = self.model_sar_err.predict(sa)
            plt.plot(np.ravel(error_abs)[diaposon[0]:diaposon[1]])
            plt.plot(np.ravel(Y_pred)[diaposon[0]:diaposon[1]])
            plt.show()
        hash_arr = self.model_rand.predict(s)
        self.model_sh.fit(s, hash_arr, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
        mse, Y_pred = self.test_model(self.model_sh,s,hash_arr, True)
        if draw:
            print('model_sh fact, frc')
            plt.plot(np.ravel(hash_arr)[diaposon[0]:diaposon[1]])
            plt.plot(np.ravel(Y_pred)[diaposon[0]:diaposon[1]])
            print('mean delta = ', np.mean(np.abs(np.ravel(Y_pred) - np.ravel(hash_arr))))
            plt.show()
            print('error in prediction hash *self.hash_curiosity')
            plt.plot(np.power(np.mean(np.abs(Y_pred - hash_arr),1)/np.mean(np.abs(Y_pred - hash_arr)),2)*self.hash_curiosity*np.mean(np.abs(self.r)))
            plt.show()
        if screen_curious:
            count_frames_curious = 60 #~1 сек
            s = np.array(self.ss)
            hash_fact = self.model_rand.predict(s)
            hash_pred = self.model_sh.predict(s)
            hash_prediction_error = np.power(np.mean(np.abs(hash_pred - hash_fact),1)/np.mean(np.abs(hash_pred - hash_fact)),2)
            if np.max(hash_prediction_error[-count_frames_curious:]*self.hash_curiosity*np.mean(np.abs(self.r)))>0.18:
                amax = np.argmax(hash_prediction_error[-count_frames_curious:])
                print('frame curiosity:',np.round(np.max(hash_prediction_error[-count_frames_curious:]*self.hash_curiosity*np.mean(np.abs(self.r))),2))
                self.parent.make_screenshot(frame_number=-count_frames_curious+amax)
        self.is_fit = True
    
    def make_discounted_rewards(self):
        self.ss = np.array(self.s)
        for multy_num in range(1, self.multy_sar):
            self.ss = np.hstack([self.ss[:-self.multy_sar_period], np.array(self.s)[multy_num*self.multy_sar_period:]])
        idx_borders = np.array(self.done)
        #borders - границы между эпизодами
        self.r_disco = utils.exp_smooth(self.r,self.discount_factor,self.planning_horison,idx_borders)
        hash_fact = self.model_rand.predict(np.array(self.ss, dtype=np.float32))
        mse, hash_pred = self.test_model(self.model_sh,np.array(self.ss,dtype=np.float32),hash_fact,True)
        self.hash_prediction_error = np.power(np.mean(np.abs(hash_pred - hash_fact),1)/np.mean(np.abs(hash_pred - hash_fact)),2)
        self.hash_prediction_error = np.hstack((self.hash_prediction_error, np.zeros(len(self.r)-len(self.hash_prediction_error))))
        hash_prediction_error_disco = utils.exp_smooth(self.hash_prediction_error,self.discount_factor,self.planning_horison,idx_borders)
        self.r_disco_curiosity = hash_prediction_error_disco*self.hash_curiosity*np.mean(np.abs(self.r)) + self.r_disco
        
class SarsaAgent_RND_dreamworlds(SarsaAgent_RND):
    def __init__(self, state_size, action_size, layers_size=[450,450],deque_len=2000):
        self.x_len = 50
        self.y_len = 50
        super().__init__(state_size=state_size, action_size=action_size, layers_size=layers_size,deque_len=deque_len)
    def reset(self, partial=False):
        self.epsilon = 0.6
        #partial - значит, не с концами всё грохнуть, а оставить N последних чисел
        if partial:
            how_much_old = 10
            self.s = deque(list(np.array(self.s)[-how_much_old:]),maxlen=self.deque_len)
            self.r = deque(list(np.array(self.r)[-how_much_old:]),maxlen=self.deque_len)
            self.a = deque(list(np.array(self.a)[-how_much_old:]),maxlen=self.deque_len)
            self.done = deque(list(np.array(self.done)[-how_much_old:]),maxlen=self.deque_len)
        else:
            # create replay memory using deque
            self.s = deque(maxlen=self.deque_len)
            self.r = deque(maxlen=self.deque_len)
            #действия в формате one_hot:
            self.a = deque(maxlen=self.deque_len)
            self.done = deque(maxlen=self.deque_len)
        # SR модель - для оценки value
        self.model_sr = self.build_model('sr')
        # SAR модель - для оценки advantage
        self.model_sar = self.build_model('sar')
        # SAR err модель - для оценки неожиданности advantage
        self.model_sar_err = self.build_model('sar_err')
        # rand модель - для рандомного хеширования состояний
        self.model_rand = self.build_model('rand')
        # State->hash модель - для прогноза хешей, для оценки новизны локации
        self.model_sh = self.build_model('sh')
        # SA>SAR-SAR-SAR модель - для генерации будущей истории, включая наши ходы
        self.model_sargen = self.build_model('sargen')
        #инициализировать модельки
        self.train_model(epochs=1)    
    def build_model(self,type_mdl):
        if type_mdl=='sar':
            input_dim = self.state_size*self.multy_sar  + self.action_size 
            out_dim = 1
        elif type_mdl=='sr':
            input_dim = self.state_size*self.multy_sar 
            out_dim = 1
        elif type_mdl=='sar_err':
            input_dim = self.state_size*self.multy_sar  + self.action_size 
            out_dim = 1
        elif type_mdl=='rand':
            input_dim = self.state_size*self.multy_sar

            out_dim = self.hash_size
        elif type_mdl=='sh':
            input_dim = self.state_size*self.multy_sar  
            out_dim = self.hash_size
        elif type_mdl=='sargen':
            input_dim = self.state_size + self.action_size + 1
            out_dim = self.state_size + self.action_size + 1
        if (type_mdl=='sar_err')or(type_mdl=='rand'):
            layer_size_multiplier = 0.5
        else:
            layer_size_multiplier = 1
        model = Sequential()
        if type_mdl!='sargen':
            model.add(BatchNormalization())
            model.add(Dense(int(self.layers_size[0]*layer_size_multiplier), input_dim=input_dim, activation='relu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0003)))
            model.add(Dropout(rate=0.3))
            model.add(BatchNormalization())
            model.add(Dense(int(self.layers_size[1]*layer_size_multiplier), activation='relu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0003)))
            model.add(Dropout(rate=0.3))
            model.add(Dense(out_dim, activation='linear',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0003)))
        else:
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Dense(1000,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0003)), input_shape=(self.x_len, input_dim)))
            model.add(TimeDistributed(Dropout(rate=0.2)))
            model.add(BatchNormalization())
            model.add(GRU(550, return_sequences=True,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0003)))
            #model.add(Dense(1000, kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(Dropout(rate=0.2))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Dense(input_dim,kernel_initializer='he_uniform',
                                            kernel_regularizer=keras.regularizers.l2(0.0003))))
            model.add(TimeDistributed(Dropout(rate=0.1)))
            
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.is_fit = False
        #model.summary()
        return model    
        
    def make_world_model(self):
        #модель для генерации sar
        s = np.array(self.s, dtype=np.float32)
        a = np.array(self.a, dtype=np.float32)
        r = np.array(self.r, dtype=np.float32, ndmin=2).T
        done = np.array(self.done, dtype=np.float32, ndmin=2).T
        self.scale_r = 10000#np.mean(np.abs(s))/(np.mean(np.abs(r))+0.001)*20
        r *= self.scale_r
        self.sar_arr_x = []
        self.sar_arr_y = []
        sar = np.hstack([s,a,r])
        count_sequences = 900
        epochs = 250
        start_arr = (np.random.rand(count_sequences)*(len(self.s) - self.x_len - self.y_len)).astype(int)
        for start in start_arr:
            while np.sum(done[start:start+2*self.x_len])>0:
                #перебрось кубы, хреновый это start
                start = int(np.random.rand()*(len(self.s) - self.x_len - self.y_len))
                if np.random.rand()<0.01:
                    print('trying to change start')
            self.sar_arr_x.append(sar[start:start+self.x_len])
            self.sar_arr_y.append(sar[start+self.x_len:start+self.x_len+self.y_len])
        print('fit world_model')
        if np.shape(self.sar_arr_x[0])[0]==0:
            print('self.self.sar_arr_x[0] is empty')
            return
        self.model_sargen.fit(np.array(self.sar_arr_x),np.array(self.sar_arr_y),batch_size=self.batch_size,
                           epochs=epochs, verbose=False)
        
    def update_target_model(self):
        border = 0.052
        #border = 0.7
        min_deque = 1100
        if len(self.r)>min_deque:
            for i in range(33):
                self.make_world_model()
                print('predicting sargen...', np.shape(np.array(self.sar_arr_x)), pd.Timestamp.now())
                #self.sar_arr_x = self.sar_arr_x[:40]
                #self.sar_arr_y = self.sar_arr_y[:40]
                Y_pred = self.model_sargen.predict(np.array(self.sar_arr_x))
                err = Y_pred - np.array(self.sar_arr_y)
                mse = np.mean(np.power(err,2))
                rmse_std = np.sqrt(mse)/np.std(np.array(self.sar_arr_y, dtype=np.float32))
                print('world model mse', np.sqrt(mse),' rmse/std', rmse_std, 'rmse/mean', np.sqrt(mse)/np.mean(np.abs(np.array(self.sar_arr_y))), pd.Timestamp.now(),flush=True)
                Y_pred = Y_pred[:,-1]
                r = np.array(self.sar_arr_y)[:,-1]/self.scale_r
                Y_pred /= self.scale_r
                mse = np.mean(np.power((Y_pred - r),2))
                print('world model reward mse', np.sqrt(mse),' rmse/std', np.sqrt(mse)/np.std(r), 'rmse/mean', np.sqrt(mse)/np.mean(np.abs(r)),'r mean',np.mean(np.abs(r)),'Y_pred mean',np.mean(np.abs(Y_pred)), flush=True)
                if rmse_std<border:
                    break
                else:
                    print('try more',rmse_std,border)
            self.test_dream()
        super().update_target_model()
        if len(self.r)>min_deque:
            if rmse_std<border:
                self.dream()
            else:
                print('dont dream: too bad model')
    def decode_to_print(self, s):
        s = np.array(s,ndmin=2)
        print(np.shape(s))
        s = self.higher_level.decode(s)
        print(np.shape(s))
        s = s[0]
        print(np.shape(s))
        img = self.parent.decode(s)
        print(np.shape(img))
        plt.imshow(img)
        plt.show()
        
    def dream(self, publish=False):
        print('dreaming')
        temperature = 0.1
        if np.std(self.r)==0:
            print('nothing to dream')
            return
        #положить заменить sar и done буфера в архив, а тем временем заменить их на новые версии
        s_archived = self.s
        a_archived = self.a
        r_archived = self.r
        done_archived = self.done
        #create sar-tables
        sar_arr = []
        count_dreams = 70
        len_dream = 60 #60*30=1800 - это почти целый раунд
        print('dreaming sar tables seeds')
        for i in range(count_dreams):
            #выбрать начало истории
            start = int((len(self.s)-self.x_len-1)*np.random.rand())
            s_start = np.array(s_archived, dtype=np.float32)[start:start+self.x_len]
            print('dream ',i,'start',start,'s',s_start[:2,:4], flush=True)
            a_start = np.array(a_archived, dtype=np.float32)[start:start+self.x_len]
            r_start = np.array(r_archived, dtype=np.float32, ndmin=2).T[start:start+self.x_len]
            done_start = np.array(done_archived)[start:start+self.x_len]
            #наложить искажения
            rand = np.std(s_start)*temperature*np.random.normal(self.x_len)
            s_start += rand
            print('dream ',i,'noise',np.shape(rand),'s_noised',s_start[:2,:4], flush=True)
            sar_start = np.hstack([s_start,a_start,r_start])
            sar_arr.append(sar_start)
        sar_arr = np.array(sar_arr, dtype=np.float32)
        print('dreaming sar tables stories', flush=True)
        sar_arr_out_arr = []
        #dream_num x dream_episode_len x state_size
        for i in range(len_dream):
            sar_arr_out = self.model_sargen.predict(sar_arr,batch_size=5)
            actions_dream = sar_arr_out[:,:,self.state_size:self.state_size+self.action_size]
            actions_dream = np.round(actions_dream)
            for idx0 in range(round(np.shape(actions_dream)[0])):
                #делаем argmax action
                actions_dream_idx0 = actions_dream[idx0]
                actions_dream_idx0 = np.eye(np.shape(actions_dream_idx0)[0],np.shape(actions_dream_idx0)[1])[np.argmax(actions_dream_idx0,axis=1)]
                actions_dream[idx0] = actions_dream_idx0   
            sar_arr_out[:,:,self.state_size:self.state_size+self.action_size] = actions_dream
            sar_arr_out_arr.append(sar_arr_out)
            sar_arr = sar_arr_out
        print('sar_arr_out',np.shape(sar_arr_out), f'must be dream_numXlenXstate+action+1, len must be {self.x_len}')
        print('sar_arr_out_arr orig',np.shape(sar_arr_out_arr),'must be batchХdream_numXlenXstate+action+1')
        sar_arr_out_arr = np.hstack(sar_arr_out_arr)
        print('sar_arr_out_arr hstack',np.shape(sar_arr_out_arr),f'must be batchXlenXstate+action+1, len must be {len_dream}, batch must be {count_dreams}')
        sar_arr_out_arr = np.vstack(list(sar_arr_out_arr))
        print('sar_arr_out_arr vstack',np.shape(sar_arr_out_arr),f'must be lenXstate+action+1, len must be {count_dreams*len_dream*self.x_len}')
        #можно было бы выбрать наиболее удачные, а можно просто получить их фильтрацией
        self.s = list(sar_arr_out_arr[:,:self.state_size])
        self.a = list(sar_arr_out_arr[:,self.state_size:self.state_size+self.action_size])
        self.r = list(sar_arr_out_arr[:,-1]/self.scale_r)
        self.done = list(np.zeros(count_dreams*len_dream*self.x_len))
        #диагностика
        print('checking difference between stories')
        s = np.array(self.s)
        a = np.array(self.a)
        r = np.array(self.r)
        for num_embedding_component in [0,1,2]:
            for num_dream in [0,2,4]:
                print(f'component {num_embedding_component}, dream {num_dream}', s[len_dream*(1+num_dream)*self.x_len-5:len_dream*(1+num_dream)*self.x_len,num_embedding_component])
        for a_component in [0,1,2,4]:
            for num_dream in [0,2,4]:
                print(f'a-component {a_component}, dream {num_dream}', a[len_dream*(1+num_dream)*self.x_len-7:len_dream*(1+num_dream)*self.x_len,a_component])
        print('checking reward sums for dreams')
        for num_dream in [0,1,2]:
            print(f'dream {num_dream} sum',np.sum(r[len_dream*(num_dream)*self.x_len:len_dream*(1+num_dream)*self.x_len]),'min',np.min(r[len_dream*(num_dream)*self.x_len:len_dream*(1+num_dream)*self.x_len]),'max',np.max(r[len_dream*(num_dream)*self.x_len:len_dream*(1+num_dream)*self.x_len]))
        #показать самые интересные кадры из снов
        amax = np.argmax(r)
        amin = np.argmin(r)
        print('r=',np.max(r))
        if 0:
            for i in np.arange(-6,6,2):
                try:
                    self.decode_to_print(s[amax+i])
                except Exception:
                    pass
        print('r=',np.min(r))
        if 0:
            for i in np.arange(-6,6,2):
                try:
                    self.decode_to_print(s[amax+i])
                except Exception:
                    pass   
            
        for i in np.arange(0,count_dreams*len_dream*self.x_len,len_dream*self.x_len):
            self.done[i] = 1
        #обучаться не только на снах
        self.done = np.vstack([np.array(done_archived,ndmin=2).T,np.array(np.ravel(self.done),ndmin=2).T])
        self.r = np.ravel(np.vstack([np.array(r_archived,ndmin=2).T,np.array(np.ravel(r),ndmin=2).T]))
        self.s = np.vstack([np.array(s_archived,ndmin=2),self.s])
        self.a = np.vstack([np.array(a_archived,ndmin=2),self.a])
        ##
        print('rewards in dreams')
        plt.plot(self.r)
        plt.show()
        print('dones in dreams')
        plt.plot(self.done)
        plt.show()
        print('rewards in dreams')
        plt.plot(self.r[-1200:])
        plt.show()
        print('dones in dreams')
        plt.plot(self.done[-1200:])
        plt.show()
        
        self.train_model(epochs=180,sub_batch_size=60000)
        if publish:
            self.s_dream = self.s
            self.a_dream = self.a
            self.r_dream = self.r
            self.done_dream = self.done
        #вернуть как было
        self.s = s_archived
        self.a = a_archived
        self.r = r_archived
        self.done = done_archived
        
class SarsaAgent_RND_dreamworlds_actor(SarsaAgent_RND_dreamworlds):
    #через dreamworlds сгенерить последовательности sar, переиграть кучу раз, найти для каждого старта наилучшую последовательность ходов
    #сделать отображение sar->a, зарядить в нейронку.
    #что на входе? Ну, текущий s мы проигнорируем, так как на него нет a, r.
    def reset(self, partial=False):
        self.count_dreams = 15
        self.len_dream = 900
        self.try_count = 6
        
        self.sa_nn_fit = False      
        self.epsilon = 0.6
        #partial - значит, не с концами всё грохнуть, а оставить N последних чисел
        if partial:
            how_much_old = 10
            self.s = deque(list(np.array(self.s)[-how_much_old:]),maxlen=self.deque_len)
            self.r = deque(list(np.array(self.r)[-how_much_old:]),maxlen=self.deque_len)
            self.a = deque(list(np.array(self.a)[-how_much_old:]),maxlen=self.deque_len)
            self.done = deque(list(np.array(self.done)[-how_much_old:]),maxlen=self.deque_len)
        else:
            # create replay memory using deque
            self.s = deque(maxlen=self.deque_len)
            self.r = deque(maxlen=self.deque_len)
            #действия в формате one_hot:
            self.a = deque(maxlen=self.deque_len)
            self.done = deque(maxlen=self.deque_len)
        # SR модель - для оценки value
        self.model_sr = self.build_model('sr')
        # SAR модель - для оценки advantage
        self.model_sar = self.build_model('sar')
        # SAR err модель - для оценки неожиданности advantage
        self.model_sar_err = self.build_model('sar_err')
        # rand модель - для рандомного хеширования состояний
        self.model_rand = self.build_model('rand')
        # State->hash модель - для прогноза хешей, для оценки новизны локации
        self.model_sh = self.build_model('sh')
        # SA>SAR-SAR-SAR модель - для генерации будущей истории, включая наши ходы
        self.model_sargen = self.build_model('sargen')
        # SA модель - для генерации actions
        self.model_sara = self.build_model('sara')
        #инициализировать модельки
        self.train_model(epochs=1) 
    def build_model(self,type_mdl):
        if type_mdl=='sara':
            input_dim = self.state_size  + self.action_size + 1
            out_dim = self.action_size
            model = Sequential()
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Dense(900,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)), input_shape=(self.x_len, input_dim)))
            model.add(TimeDistributed(Dropout(rate=0.15)))
            model.add(BatchNormalization())
            model.add(GRU(450, return_sequences=False,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            #model.add(Dense(1000, kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(Dropout(rate=0.1))
            model.add(BatchNormalization())
            model.add(Dense(out_dim,kernel_initializer='he_uniform',
                                            kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model
        elif type_mdl=='sargen':
            input_dim = self.state_size + self.action_size + 1
            out_dim = self.state_size + self.action_size + 1
            model = Sequential()
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Dense(900,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0002)), input_shape=(self.x_len, input_dim)))
            model.add(TimeDistributed(Dropout(rate=0.15)))
            model.add(BatchNormalization())
            model.add(GRU(500, return_sequences=False,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0002)))
            #model.add(Dense(1000, kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(Dropout(rate=0.12))
            model.add(BatchNormalization())
            model.add(Dense(out_dim,kernel_initializer='he_uniform',
                                            kernel_regularizer=keras.regularizers.l2(0.0002)))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model
        else:
            return super().build_model(type_mdl)
    
    # get action from model using epsilon-greedy policy
    # OR get action from sara model
    def get_action(self, state,  verbose=0):
        #state - это склейка последних нескольких state. На каком этапе делается эта склейка? Не помню, возможно, на каком-то из автоэнкодеров. Да какая разница.
        if not self.sa_nn_fit:
            return super().get_action(state,  verbose)
        else:
            if (np.random.rand() <= self.epsilon) or (len(self.s)<self.train_start) or (self.is_fit==False):
                #может, порандомим?
                return random.randrange(self.action_size)
            else:
                s = np.array(self.s,ndmin=2)
                s = np.reshape(s,(np.shape(s)[0],np.shape(s)[1]))
                a = np.array(self.a,ndmin=2)
                a = np.reshape(a,(np.shape(a)[0],np.shape(a)[1]))
                r = np.array(self.r,ndmin=2).T*self.scale_r
                sar = np.array([np.hstack([s,a,r])[-self.x_len:,:]],ndmin=3)
                a = self.model_sara.predict(sar)
            if verbose:
                print(a)
            return np.argmax(a)
        
    def make_world_model(self):
        #модель для генерации sar. На 1 такт вперёд, не более!
        s = np.array(self.s, dtype=np.float32)
        a = np.array(self.a, dtype=np.float32)
        r = np.array(self.r, dtype=np.float32, ndmin=2).T
        done = np.array(self.done, dtype=np.float32, ndmin=2).T
        self.scale_r = 100#np.mean(np.abs(s))/(np.mean(np.abs(r))+0.001)*20
        r *= self.scale_r
        self.sar_arr_x = []
        self.sar_arr_y = []
        sar = np.hstack([s,a,r])
        count_sequences = 1300
        epochs = 250
        start_arr = (np.random.rand(count_sequences)*(len(self.s) - self.x_len - 1)).astype(int)
        for start in start_arr:
            while np.sum(done[start:start+2*self.x_len])>0:
                #перебрось кубы, хреновый это start
                start = int(np.random.rand()*(len(self.s) - self.x_len - 1))
                if np.random.rand()<0.01:
                    print('trying to change start')
            self.sar_arr_x.append(sar[start:start+self.x_len])
            self.sar_arr_y.append(np.ravel(sar[start+self.x_len:start+self.x_len+1]))
        print('fit world_model')
        if np.shape(self.sar_arr_x[0])[0]==0:
            print('self.self.sar_arr_x[0] is empty')
            return
        self.model_sargen.fit(np.array(self.sar_arr_x),np.array(self.sar_arr_y),batch_size=self.batch_size,
                           epochs=epochs, verbose=False)
        
    def dream(self, publish=False):
        #здесь смысл dream другой. Сгенерировать несколько историй таким образом, получая каждое действие через get_action и генерируя ровно на один ход вперёд.
        #сгенерировать так несколько раз с одного seed, учитывая, что у нас есть рандом в get_action.
        #cделать так несколько раз, отобрать траекторию, давшую больше всего профита
        print('dreaming')
        temperature = 0.05
        if np.std(self.r)==0:
            print('nothing to dream')
            return
        #положить заменить sar и done буфера в архив, а тем временем заменить их на новые версии
        s_archived = self.s
        a_archived = self.a
        r_archived = self.r
        done_archived = self.done
        sar_arr = []#список историй
        count_dreams = self.count_dreams
        len_dream = self.len_dream #длина в тактах (~20 секунд)
        try_count = self.try_count #число переигрываний одной истории
        #Это всё займёт процессорного времени 6*6*9*5 с = 1650 с для sara
        print('dreaming sar tables seeds',pd.Timestamp.now())
        epsilon_archived = self.epsilon
        self.epsilon = 0.07

        for i in range(count_dreams):
            #выбрать начало истории
            start = int((len(self.s)-self.x_len-1)*np.random.rand())
            s_start = np.array(s_archived, dtype=np.float32)[start:start+self.x_len]
            print('dream ',i,'start',start,'s',s_start[:2,:4], flush=True)
            a_start = np.array(a_archived, dtype=np.float32)[start:start+self.x_len]
            r_start = np.array(r_archived, dtype=np.float32, ndmin=2).T[start:start+self.x_len]
            done_start = np.array(done_archived)[start:start+self.x_len]
            #наложить искажения
            rand = np.std(s_start)*temperature*np.random.normal(self.x_len)
            s_start += rand
            sar_start = np.hstack([s_start,a_start,r_start])
            #
            best_reward = -1e100
            best_idx = 0
            sar_arr_best_trial = []
            for j in range(try_count):
                sar_arr_current_trial = np.array(sar_start,ndmin=3)
                for k in range(len_dream):
                    self.s = list(sar_arr_current_trial[0,:,:self.state_size])
                    self.a = list(sar_arr_current_trial[0,:,self.state_size:self.state_size+self.action_size])
                    self.r = list(sar_arr_current_trial[0,:,-1]/self.scale_r)
                    a = self.get_action(state=sar_arr_current_trial[:,-1,:self.state_size])
                    a_onehot = np.zeros([1,self.action_size])
                    a_onehot[:,a] = 1
                    sar_arr_out = self.model_sargen.predict(sar_arr_current_trial[:,-self.x_len:,:],batch_size=5)
                    sar_arr_out[0,self.state_size:self.state_size+self.action_size] = a_onehot
                    sar_arr_current_trial = np.hstack([sar_arr_current_trial,np.array(sar_arr_out[0:1,:],ndmin=3)])
                reward = np.sum(sar_arr_current_trial[:,:,-1])

                ss = sar_arr_current_trial[0,:,:self.state_size]
                #hash_fact = self.model_rand.predict(np.array(ss, dtype=np.float32))
                #mse, hash_pred = self.test_model(self.model_sh,np.array(ss,dtype=np.float32),hash_fact,True)
                #self.hash_prediction_error = np.power(np.mean(np.abs(hash_pred - hash_fact),1)/np.mean(np.abs(hash_pred - hash_fact)),2)
                #reward_curiosity = np.sum(self.hash_prediction_error*self.hash_curiosity*np.mean(np.abs(self.r)))
                #reward = reward + reward_curiosity*self.scale_r

                if reward>=best_reward:
                    best_idx = j
                    best_reward = reward
                    #, 'reward_curiosity', reward_curiosity,
                    print('try',j,'reward',reward/self.scale_r, pd.Timestamp.now(), flush=True)
                    sar_arr_best_trial = np.array(sar_arr_current_trial)
            sar_arr.append(sar_arr_best_trial)
        len_one_series = np.shape(sar_arr_best_trial)[1]
        sar_arr = np.hstack(sar_arr)
        self.s = list(sar_arr[0,:,:self.state_size])
        self.a = list(sar_arr[0,:,self.state_size:self.state_size+self.action_size])
        self.r = list(sar_arr[0,:,-1]/self.scale_r)
        self.done = list(np.zeros(len(self.r)))
        for i in np.arange(0,len(self.r),len_one_series):
            self.done[i] = 1
        self.epsilon = epsilon_archived
        #диагностика
        print('checking difference between stories')
        s = np.array(self.s)
        a = np.array(self.a)
        r = np.array(self.r)
        for num_embedding_component in [0,1,2]:
            for num_dream in [0,2,4]:
                print(f'component {num_embedding_component}, dream {num_dream}', s[len_dream*(1+num_dream)-5:len_dream*(1+num_dream),num_embedding_component])
        for a_component in [0,1,2,4]:
            for num_dream in [0,2,4]:
                print(f'a-component {a_component}, dream {num_dream}', a[len_dream*(1+num_dream)-7:len_dream*(1+num_dream),a_component])
        print('checking reward sums for dreams')
        for num_dream in [0,1,2]:
            print(f'dream {num_dream} sum',np.sum(r[len_dream*(num_dream):len_dream*(1+num_dream)]),'min',np.min(r[len_dream*(num_dream):len_dream*(1+num_dream)]),'max',np.max(r[len_dream*(num_dream):len_dream*(1+num_dream)]))
        #показать самые интересные кадры из снов
        amax = np.argmax(r)
        amin = np.argmin(r)
        print('r=',np.min(r),np.max(r),pd.Timestamp.now())
        if 0:
            for i in np.arange(-6,6,2):
                try:
                    self.decode_to_print(s[amax+i])
                except Exception:
                    pass
        print('r=',np.min(r))
        if 0:
            for i in np.arange(-6,6,2):
                try:
                    self.decode_to_print(s[amax+i])
                except Exception:
                    pass   

        self.make_sara_model()
        if publish:
            self.s_dream = self.s
            self.a_dream = self.a
            self.r_dream = self.r
            self.done_dream = self.done
        #вернуть как было
        self.s = s_archived
        self.a = a_archived
        self.r = r_archived
        self.done = done_archived
        
    def test_dream(self, publish=False):
        #нужно, чтобы взять пару эпизодов и предсказать будущее на сколько-то ходов вперёд. И посмотреть, насколько прогноз с фактом разошёлся
        print('Test_dreaming')
        temperature = 0.0
        if np.std(self.r)==0:
            print('nothing to dream')
            return
        #положить заменить sar и done буфера в архив, а тем временем заменить их на новые версии
        s_archived = self.s
        a_archived = self.a
        r_archived = self.r
        done_archived = self.done
        sar_arr = []#список историй
        count_dreams = 3
        len_dream = 120 #длина в тактах (~20 секунд)
        try_count = 1 #число переигрываний одной истории

        for i in range(count_dreams):
            #выбрать начало истории
            start = int((len(self.s)-self.x_len-1-len_dream)*np.random.rand())
            if start<0:
                print('dream break',i,'start',start,'s',s_start[:2,:4], flush=True)
                continue
            s_start = np.array(s_archived, dtype=np.float32)[start:start+self.x_len]
            print('dream ',i,'start',start,'s',s_start[:2,:4], flush=True)
            a_start = np.array(a_archived, dtype=np.float32)[start:start+self.x_len]
            r_start = np.array(r_archived, dtype=np.float32, ndmin=2).T[start:start+self.x_len]
            done_start = np.array(done_archived)[start:start+self.x_len]
            #наложить искажения
            rand = np.std(s_start)*temperature*np.random.normal(self.x_len)
            s_start += rand
            sar_start = np.hstack([s_start,a_start,r_start])
            #
            best_reward = -1e100
            best_idx = 0
            for j in range(try_count):
                sar_arr_current_trial = np.array(sar_start,ndmin=3)
                for k in range(len_dream):
                    self.s = list(sar_arr_current_trial[0,:,:self.state_size])
                    self.a = list(sar_arr_current_trial[0,:,self.state_size:self.state_size+self.action_size])
                    self.r = list(sar_arr_current_trial[0,:,-1]/self.scale_r)
                    a_onehot = np.array(a_archived, dtype=np.float32)[start+self.x_len+k:start+self.x_len+k+1]

                    sar_arr_out = self.model_sargen.predict(sar_arr_current_trial[:,-self.x_len:,:],batch_size=5)
                    sar_arr_out[0,self.state_size:self.state_size+self.action_size] = a_onehot
                    sar_arr_current_trial = np.hstack([sar_arr_current_trial,np.array(sar_arr_out[0:1,:],ndmin=3)])
                r_predict = sar_arr_current_trial[:,-len_dream:,-1]/self.scale_r
                s_predict = sar_arr_current_trial[:,-len_dream:,:self.state_size]
                a_predict = sar_arr_current_trial[:,-len_dream:,self.state_size:self.state_size+self.action_size]
                r_fact = np.array(r_archived, dtype=np.float32)[start+self.x_len:start+self.x_len+len_dream]
                s_fact = np.array(s_archived, dtype=np.float32)[start+self.x_len:start+self.x_len+len_dream]
                a_fact = np.array(a_archived, dtype=np.float32)[start+self.x_len:start+self.x_len+len_dream]
                rmse_r = np.sqrt(np.mean((r_predict-r_fact)**2))
                rmse_s = np.sqrt(np.mean((s_predict-s_fact)**2))
                rmse_a = np.sqrt(np.mean((a_predict-a_fact)**2))
                print('rmse s,a,r',rmse_s,rmse_a,rmse_r)
                rmse_r_arr = np.sqrt((r_predict-r_fact)**2)
                rmse_s_arr = np.sqrt(np.mean((s_predict-s_fact)**2, axis=2))
                rmse_a_arr = np.sqrt(np.mean((a_predict-a_fact)**2, axis=2))
                print('s mean=',np.mean(np.abs(s_fact)))
                plt.plot(rmse_s_arr[0])
                plt.show()
                print('r mean=',np.mean(np.abs(r_fact)))
                plt.plot(rmse_r_arr[0])
                plt.show()
        if publish:
            self.s_dream = self.s
            self.a_dream = self.a
            self.r_dream = self.r
            self.done_dream = self.done
        #вернуть как было
        self.s = s_archived
        self.a = a_archived
        self.r = r_archived
        self.done = done_archived
        
    def make_sara_model(self):
        #генерирует actions по последовательностям sar
        #то есть у нас есть sar, а мы из него нарезаем датасет и учим нейронку
        s = np.array(self.s, dtype=np.float32)
        a = np.array(self.a, dtype=np.float32)
        r = np.array(self.r, dtype=np.float32, ndmin=2).T
        done = np.array(self.done, dtype=np.float32, ndmin=2).T
        self.sar_arr_x = []
        self.a_arr_y = []
        sar = np.hstack([s,a,r])
        count_sequences = 1200
        epochs = 190
        start_arr = (np.random.rand(count_sequences)*(len(self.s) - self.x_len - 1)).astype(int)
        for start in start_arr:
            while np.sum(done[start:start+self.x_len+1])>0:
                #перебрось кубы, хреновый это start
                start = int(np.random.rand()*(len(self.s) - self.x_len - 1))
                if np.random.rand()<0.01:
                    print('trying to change start')
            self.sar_arr_x.append(sar[start:start+self.x_len])
            self.a_arr_y.append(np.ravel(a[start+self.x_len:start+self.x_len+1]))
        print('fit sara_model')
        if np.shape(self.sar_arr_x[0])[0]==0:
            print('self.self.sar_arr_x[0] is empty')
            return
        self.model_sara.fit(np.array(self.sar_arr_x),np.array(self.a_arr_y),batch_size=self.batch_size,
                           epochs=epochs, verbose=False)
        self.sa_nn_fit = True
        y_pred = self.model_sara.predict(np.array(self.sar_arr_x))
        mse = np.mean((y_pred-np.array(self.a_arr_y))**2)
        print('mse',mse)
        y_pred_amax = np.argmax(y_pred,axis=1)
        a_arr_y_amax = np.argmax(self.a_arr_y,axis=1)
        print('success a pred sum',np.sum(y_pred_amax==a_arr_y_amax),'success a pred mean',np.sum(y_pred_amax==a_arr_y_amax)/np.shape(y_pred_amax)[0], flush=True)