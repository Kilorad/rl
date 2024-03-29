import gym
import sys
import pylab
import random
import numpy as np
from collections import deque
import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
sys.path.append('./common/')
import utils


# double SARSA agent (sr+sar)
class SarsaAgent:
    def __init__(self, state_size, action_size, layers_size=[100,100], maxlen=10000):
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
        self.batch_size = 3000
        self.sub_batch_size=500
        self.train_start = 2000
        self.reward_part_need = 0.3
        self.planning_horison = 810
        self.layers_size = layers_size
        # create replay memory using deque
        self.memory = deque(maxlen=maxlen)

        # create main model and target model
        self.model_sr = self.build_model('sr')
        self.model_sar = self.build_model('sar')

        # initialize target model
        self.train_model(epochs=1)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self,type):
        if type=='sar':
            input_dim = self.state_size + self.action_size 
            out_dim = 1
        elif type=='sr':
            input_dim = self.state_size
            out_dim = 1
        model = Sequential()
        model.add(Dense(self.layers_size[0], input_dim=input_dim, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        model.add(Dense(self.layers_size[1], activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        model.add(Dense(out_dim, activation='linear',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.001)))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model


    # get action from model using epsilon-greedy policy
    def get_action(self, state, verbose=0,extended=0):
        if np.random.rand() <= self.epsilon or len(self.memory)<self.train_start:
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
        sar_curr = np.concatenate((state[0,:],a_one_hot,[reward],[done]))
        self.memory.append(sar_curr)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def make_discounted_rewards(self):
        arr_mem = np.array(self.memory)
        self.s = arr_mem[:,:self.state_size]
        self.a = arr_mem[:,self.state_size:self.state_size+self.action_size]
        self.r = arr_mem[:,self.state_size+self.action_size:self.state_size+self.action_size+1]
        self.d = arr_mem[:,self.state_size+self.action_size+1:self.state_size+self.action_size+2]
        self.r_disco = utils.exp_smooth(self.r,self.discount_factor,self.planning_horison,self.d)

    # pick samples randomly from replay memory (with batch_size)
    def rebalance_data(self,s,action,reward):
        mean = np.mean(reward)
        #размножь большие
        idx_big = reward>mean
        if any(idx_big):
            idx_big_num = np.where(idx_big)[0]
            s_add = list(s[idx_big_num])
            action_add = list(action[idx_big_num])
            reward_add = list(reward[idx_big_num])

            initial_part = np.mean(idx_big)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1

            #action.extend([action_add]*multiplication_coef)
            #reward.extend([reward_add]*multiplication_coef)
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                action = np.concatenate((action,action_add))
                reward = np.concatenate((reward,reward_add))
            #print('initial_part',initial_part,'mean',mean,'multiplication_coef',multiplication_coef)
        #размножь мелкие
        idx_small = reward<mean
        if any(idx_small):
            idx_small_num = np.where(idx_small)[0]
            s_add = list(s[idx_small_num])
            action_add = list(action[idx_small_num])
            reward_add = list(reward[idx_small_num])

            initial_part = np.mean(idx_small)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1

            #action.extend([action_add]*multiplication_coef)
            #reward.extend([reward_add]*multiplication_coef)
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                action = np.concatenate((action,action_add))
                reward = np.concatenate((reward,reward_add))
            #print('initial_part',initial_part,'mean',mean,'multiplication_coef',multiplication_coef)
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
                    #action.extend([action_add]*multiplication_coef)
                    #reward.extend([reward_add]*multiplication_coef)
                    s=np.vstack((s,s_add))
                    action = np.concatenate((action,action_add))
                    reward = np.concatenate((reward,reward_add))
            corrections_count+=1
            if not corrections_flag:
                break
        return (s,action,reward)
    def update_target_model(self):
        self.train_model(epochs=10,sub_batch_size=6000,verbose=0)
        self.train_model(epochs=1,sub_batch_size=6000,verbose=0)
    def test_model(self,model,X,Y):
        mse = np.mean((model.predict(X)-Y)**2)
        return mse
    def train_model(self,epochs=1,sub_batch_size=None,verbose=0):
        if len(self.memory) < self.train_start:
            return
        if sub_batch_size is None:
            sub_batch_size = self.sub_batch_size
        self.make_discounted_rewards()
        batch_size = max(self.batch_size,sub_batch_size)
        batch_size = min(batch_size, len(self.memory))
        for i in range(6):
            mini_batch = np.random.randint(low=0,high=self.s.shape[0],size=len(self.memory))
            #я хочу, чтобы в батч попали награды, чтобы было, за что цепляться
            r = self.r_disco[mini_batch,:]
            if np.max(r)!=np.min(r):
                break
        s = self.s[mini_batch,:]
        a = self.a[mini_batch,:]
        #    
        (s,a,r) = self.rebalance_data(s,a,r)
        if np.std(r)==0:
            return
        #
        batch_size = min(sub_batch_size, len(self.memory))
        mini_batch = np.random.randint(low=0,high=self.s.shape[0],size=batch_size)
        s=s[mini_batch,:]
        a=a[mini_batch,:]
        r=r[mini_batch]

        if len(self.memory) < self.train_start*1.05:
            verbose = True
            epochs*=2
        for i in range(4):
            self.model_sr.fit(s, r, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
            mse = self.test_model(self.model_sr,s,r)
            if epochs==1:
                break
            if np.std(r)==0:
                break
            if mse/np.std(r)<=1: #обучать до тех пор, пока не станет хорошо
                break
        r_sr_predicted = self.model_sr.predict(s)
        #Предсказать дельту
        delta_r = r-r_sr_predicted
        
        for i in range(4):
            self.model_sar.fit(np.hstack((s,a)), delta_r, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
            mse = self.test_model(self.model_sar,np.hstack((s,a)),delta_r)
            if epochs==1:
                break
            if np.std(delta_r)==0:
                break
            if mse/np.std(delta_r)<=0.35: #обучать до тех пор, пока не станет хорошо
                break
                
        if verbose:
            print('delta_r',np.std(delta_r),'r',np.std(r))