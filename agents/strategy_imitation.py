import gym
import pylab
import random
import aa_gun
import numpy as np
from collections import deque
import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

def exp_smooth(data,alpha,steps):
    data_to_mod = np.copy(data)
    for i in range(1,steps):
        roll = np.roll(data,-i)
        roll[:i+1]=np.median(data)
        data_to_mod=data_to_mod+roll*(alpha**i)
    #data_to_mod = data_to_mod/np.std(data_to_mod)
    return(data_to_mod)
# Goal-based Agent for the Cartpole
class ImitAgent:
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
        self.epsilon_decay = 0.9998
        self.epsilon_min = 0.01
        self.batch_size = 9000
        self.sub_batch_size=800
        self.train_start = 3000
        self.reward_part_need = 0.4
        self.planning_horison = 270
        # create replay memory using deque
        self.memory = deque(maxlen=10000)

        # create main model and target model
        self.model_sr = self.build_model('sr')
        self.model_sar = self.build_model('sar')
        self.model_sra = self.build_model('sra')

        # initialize target model
        self.train_model(epochs=1)
        if self.load_model:
            self.model.load_weights("./save_model/cartpole_ddqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self,type):
        if type=='sra':
            input_dim = self.state_size + 1
            out_dim = self.action_size 
        elif type=='sar':
            input_dim = self.state_size + self.action_size 
            out_dim = 1
        elif type=='sr':
            input_dim = self.state_size
            out_dim = 1
        model = Sequential()
        model.add(Dense(160, input_dim=input_dim, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.3))
        model.add(Dense(160, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.3))
        model.add(Dense(out_dim, activation='linear',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.01)))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    # get action from model using epsilon-greedy policy
    def get_action(self, state, verbose=0,extended=0):
        if np.random.rand() <= self.epsilon or len(self.memory)<self.train_start:
            return random.randrange(self.action_size)
        else:
            #Предсказать возможное R
            #Затем чуть завысить R и выдать A (сделать так для 2-3 разных R), записать в массив
            #Затем предсказать R сарсой
            r_predict = self.model_sr.predict(state)
            r_predict_abs = abs(r_predict)
            r_addition_array = [r_predict_abs*(-0.1),r_predict_abs*0.1,r_predict_abs*0.25,r_predict_abs*0.5,r_predict_abs*1,np.percentile(self.r_disco,90)-r_predict]
            if extended:
                r_addition_array.extend([r_predict_abs*2,r_predict_abs*0.01,np.percentile(self.r_disco,50)-r_predict,np.percentile(self.r_disco,60)-r_predict,np.percentile(self.r_disco,20)-r_predict,np.percentile(self.r_disco,95)-r_predict])
            a_array = []
            for r_addition in r_addition_array:
                #sra-модель работает с дельта r
                sr_current=np.hstack((state,r_addition))
                raw=self.model_sra.predict(sr_current)
                a_predict = np.argmax(raw)
                a_array.append(a_predict)
            a_array = np.unique(np.array(a_array))
            #предсказать R через sarsa
            r_predict_array = []
            for a in a_array:
                a_one_hot = np.zeros(self.action_size)
                a_one_hot[a]=1
                sa_current=np.concatenate((state[0,:],a_one_hot))
                sa_current=np.array(sa_current,ndmin=2)
                #sar-модель работает с дельта r
                r_predict_array.append(self.model_sar.predict(sa_current)[0][0])
            if verbose:
                print('a_array',a_array,'r_predict',r_predict,'r_predict_array',r_predict_array)
            return a_array[np.argmax(r_predict_array)]

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        a_one_hot = np.zeros(self.action_size)
        a_one_hot[action]=1
        sar_curr = np.concatenate((state[0,:],a_one_hot,[reward]))
        self.memory.append(sar_curr)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def make_discounted_rewards(self):
        arr_mem = np.array(self.memory)
        self.s = arr_mem[:,:self.state_size]
        self.a = arr_mem[:,self.state_size:self.state_size+self.action_size]
        self.r = arr_mem[:,self.state_size+self.action_size:self.state_size+self.action_size+1]
        self.r_disco = exp_smooth(self.r,self.discount_factor,self.planning_horison)

    # pick samples randomly from replay memory (with batch_size)
    def rebalance_data(self,s,action,reward):
        mean = np.mean(reward)
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
        self.train_model(epochs=100,sub_batch_size=9000,verbose=0)
        self.train_model(epochs=1,sub_batch_size=9000,verbose=1)
    def train_model(self,epochs=1,sub_batch_size=None,verbose=0):
        if len(self.memory) < self.train_start:
            return
        if sub_batch_size is None:
            sub_batch_size = self.sub_batch_size
        self.make_discounted_rewards()
        batch_size = max(self.batch_size,sub_batch_size)
        batch_size = min(batch_size, len(self.memory))
        for i in range(7):
            mini_batch = np.random.randint(low=0,high=self.s.shape[0],size=len(self.memory))
            #я хочу, чтобы в батч попали награды, чтобы было, за что цепляться
            r = self.r_disco[mini_batch,:]
            if np.max(r)!=np.min(r):
                break
        s = self.s[mini_batch,:]
        a = self.a[mini_batch,:]
        #    
        (s,a,r) = self.rebalance_data(s,a,r)
        #
        batch_size = min(sub_batch_size, len(self.memory))
        mini_batch = np.random.randint(low=0,high=self.s.shape[0],size=batch_size)
        s=s[mini_batch,:]
        a=a[mini_batch,:]
        r=r[mini_batch]

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        if len(self.memory) < self.train_start*1.05:
            epochs*=4
        self.model_sr.fit(s, r, batch_size=self.batch_size,
                       epochs=epochs, verbose=verbose)
        r_sr_predicted = self.model_sr.predict(s)
        #Предсказать дельту
        delta_r = r-r_sr_predicted
        self.model_sar.fit(np.hstack((s,a)), delta_r, batch_size=self.batch_size,
                       epochs=epochs, verbose=verbose)
        #А здесь сделать аугментацию!
        idx_good_decisions = np.concatenate((np.where(delta_r>np.percentile(delta_r,60))[0],np.where(delta_r>np.percentile(delta_r,90))[0],np.where(delta_r>np.percentile(delta_r,96))[0]))
        self.model_sra.fit(np.hstack((s,delta_r))[idx_good_decisions,:], a[idx_good_decisions,:], batch_size=self.batch_size,
                       epochs=int(epochs*1.2), verbose=verbose)