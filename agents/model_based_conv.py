import gym
import sys
import pylab
import random
import aa_gun
import numpy as np
from collections import deque
import keras
from keras.layers import Dense, Dropout, UpSampling1D, Conv1D, concatenate,MaxPooling1D,AveragePooling1D,Flatten,Reshape
from keras.optimizers import Adam
from keras.models import Sequential, Model, Input
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from sklearn import metrics
sys.path.append('./common/')
import utils

#Model Based, но прогноз будущего основан на свёрточной нейронке

class ModelBasedConvAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Model Based
        self.discount_factor = 0.995
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 3000 #сколько пар используем в обучении
        self.train_start = 1000
        
        self.reward_part_need = 0.45
        self.planning_horison = 125*3
        self.history_conv_len = 25*3
        
        
        self.count_plans = 32
        self.actions_count = 10
        self.plan_len = 70
        # create replay memory using deque
        memlen = 7000

        # create main model and target model
        self.model_ss = self.build_model('ss')
        
        self.s=deque(maxlen=memlen)
        self.ns=deque(maxlen=memlen)
        self.r=deque(maxlen=memlen)
        self.a=deque(maxlen=memlen)
        self.d=deque(maxlen=memlen)

        # initialize target model
        self.train_model(epochs=1)

    def build_model(self,type):
        if type=='ss':
            input_dim_s = self.state_size + self.action_size + 1
            input_dim_a = self.action_size
                 
        #сюда влетает временной ряд наблюдений        
        input_s = Input(batch_shape=(1,self.history_conv_len,input_dim_s))
        k_size=5
        x = Conv1D(filters=32,kernel_size=k_size, activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(input_s)
        x1 = MaxPooling1D(k_size)(x)
        x15=BatchNormalization()(x1)
        x2 = Conv1D(filters=32,kernel_size=k_size, activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(x15)
        x3 = MaxPooling1D(k_size)(x2)
        x4 = AveragePooling1D()(x3)
        #25-кратное сжатие
        flat = Flatten()(x4)
        encoded = Dense(50,kernel_regularizer=keras.regularizers.l2(0.001))(flat)
        s_preprocessor = Model(inputs=input_s, outputs=encoded)
        
        #сюда влетает план действий
        input_a = Input(batch_shape=(1,self.planning_horison,input_dim_a))
        k_size=5
        x = Conv1D(filters=32,kernel_size=k_size, activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(input_a)
        x1 = MaxPooling1D(k_size)(x)
        x15=BatchNormalization()(x1)
        x2 = Conv1D(filters=32,kernel_size=k_size, activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(x15)
        x3 = MaxPooling1D(k_size)(x2)
        x4 = AveragePooling1D()(x3)
        #25-кратное сжатие
        flat = Flatten()(x4)
        encoded = Dense(50,kernel_regularizer=keras.regularizers.l2(0.001))(flat)
        a_preprocessor = Model(inputs=input_a, outputs=encoded)
        
        
        combined = concatenate([s_preprocessor.output, a_preprocessor.output])
        
        #выход
        size_auto = int(self.planning_horison*input_dim_s/(k_size*k_size*k_size))
        k_count=16 #число ядер
        d1 = Dense( (size_auto)*k_count)(combined)
        d2 = Reshape((size_auto,k_count))(d1)
        d3 = Conv1D(k_count,input_dim_s,strides=k_size, activation='relu', padding='same')(d2)
        d4 = UpSampling1D(k_size)(d3)
        d5 = Conv1D(k_count,input_dim_s,strides=k_size, activation='relu', padding='same')(d4)
        d6 = UpSampling1D(k_size)(d5)
        d7 = Conv1D(k_count,input_dim_s,strides=k_size, activation='relu', padding='same')(d6)
        d8 = UpSampling1D(k_size)(d7)
        d9 = BatchNormalization()(d8)
        decoded = Conv1D(1,input_dim_s,strides=1, activation='sigmoid', padding='same')(d9)
        
        #Склеиваем все 3 куска воедино
        model= Model(inputs=[s_preprocessor.input, a_preprocessor.input], outputs=decoded)
        model.summary()
        model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))

        return model


    # get action from model using epsilon-greedy policy
    def get_action(self, state, verbose=0,extended=0):
        #ПОКА НЕ ГОТОВО
        if np.random.rand() <= self.epsilon or len(self.s)<self.train_start:
            if verbose:
                print('r_predict_array random')
            return random.randrange(self.action_size)
        else:
            #Перебрать разные последовательности A, для них предсказать R
            r_predict_array = []
            plans = self.generate_plans(self.count_plans,self.actions_count,self.plan_len)
            for plan in plans:
                (reward_mean,s,a,r)=self.estimate_plan(state,plan)
                r_predict_array.append(reward_mean)
            argmx = np.argmax(r_predict_array)
            plan = plans[argmx]
            if verbose:
                print('r_predict_array',r_predict_array,plan)
            return int(plan[0])
    def generate_plan(self,actions_count,plan_len):
        plan = np.zeros(plan_len)
        for i in range(plan_len):
            plan[i] = random.randrange(self.action_size)
        return plan
    def generate_plans(self,count_plans,actions_count,plan_len):
        plans = [self.generate_plan(actions_count,self.planning_horison) for i in range(count_plans) ]
        return plans
    def estimate_plan(self, state, action_list):
        #ПОКА НЕ ГОТОВО
        #-1 - значит придумай действие сам
        s = []
        a = []
        r = []
        action_by_model_based = random.randrange(self.action_size)
        for act in action_list:
            if act==-1:
                act=action_by_model_based
            if (act<0) or (act>=self.action_size):
                act = random.randrange(self.action_size)
            act = int(act)
            a_onehot=np.zeros(self.action_size)
            a_onehot[act]=1
            sa = np.hstack((np.array(state,ndmin=2),np.array(a_onehot,ndmin=2)))
            nsar=self.model_ss.predict(sa)
            s.append(nsar[:self.state_size])
            action_by_model_based_one_hot=nsar[0,self.state_size+1:self.state_size+self.action_size+1]
            action_by_model_based = np.argmax(action_by_model_based_one_hot)
            a.append(action_by_model_based)
            rew=nsar[0][self.state_size+self.action_size:self.state_size+self.action_size+1]
            r.append(rew)
        r_disco = utils.exp_smooth(r,self.discount_factor,len(r))
        if len(r_disco)>0:
            reward_mean = np.mean(r_disco)
        else:
            reward_mean = 0
        return(reward_mean,s,a,r)
    # save sample <s,a,r,s',d> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        a_one_hot = np.zeros(self.action_size)
        a_one_hot[action]=1
        self.s.append(state)
        self.ns.append(next_state)
        self.r.append(reward)
        self.a.append(a_one_hot)
        self.d.append(done)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay       

    # pick samples randomly from replay memory (with batch_size)
    def rebalance_data(self,mini_batch):
        #ПОКА НЕ НАДО
        
        #idx_num = np.where(idx)[0]
        #mini_batch - циферки
        s = [self.s[idx] for idx in mini_batch]
        ns = [self.ns[idx] for idx in mini_batch]
        d = [self.d[idx] for idx in mini_batch]
        a = [self.a[idx] for idx in mini_batch]
        r = [self.r[idx] for idx in mini_batch]
        mean = np.mean(r)
        #размножь большие
        idx_big = self.r>mean
        
        if any(idx_big):
            idx_big_num = np.where(idx_big)[0]
            s_add = [self.s[idx] for idx in idx_big_num]
            ns_add = [self.ns[idx] for idx in idx_big_num]
            d_add = [self.d[idx] for idx in idx_big_num]
            a_add = [self.a[idx] for idx in idx_big_num]
            r_add = [self.r[idx] for idx in idx_big_num]

            initial_part = np.mean(idx_big)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1


            for j in range(multiplication_coef):
                s.extend(s_add)
                ns.extend(ns_add)
                d.extend(d_add)
                a.extend(a_add)
                r.extend(r_add)
                
        #размножь мелкие
        idx_small = self.r<mean
        if any(idx_small):
            idx_small_num = np.where(idx_small)[0]
            s_add = [self.s[idx] for idx in idx_small_num]
            ns_add = [self.ns[idx] for idx in idx_small_num]
            d_add = [self.d[idx] for idx in idx_small_num]
            a_add = [self.a[idx] for idx in idx_small_num]
            r_add = [self.r[idx] for idx in idx_small_num]

            initial_part = np.mean(idx_small)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1

            for j in range(multiplication_coef):
                s.extend(s_add)
                ns.extend(ns_add)
                d.extend(d_add)
                a.extend(a_add)
                r.extend(r_add)
                
        #аугментировать по экшнам. Сделать частоту каждого экшна в выборке >= 0.3/число экшнов
        freq_arr=np.zeros(self.action_size)
        for corrections_count in range(6):
            corrections_flag=False
            for i in range(self.action_size):
                freq_arr[i]=np.mean(np.array(self.a)[:,i]==1)
                if freq_arr[i]<0.3/self.action_size and freq_arr[i]>0:
                    corrections_flag=True
                    #размножаем
                    idx_act_num = np.where(np.array(self.a)[:,i]==1)[0]
                    #s_add = list(s[idx_act_num])
                    s_add = [self.s[idx] for idx in idx_act_num]
                    ns_add = [self.ns[idx] for idx in idx_act_num]
                    d_add = [self.d[idx] for idx in idx_act_num]
                    a_add = [self.a[idx] for idx in idx_act_num]
                    r_add = [self.r[idx] for idx in idx_act_num]
                    
                    s.extend(s_add)
                    ns.extend(ns_add)
                    d.extend(d_add)
                    a.extend(a_add)
                    r.extend(r_add)
                             
            corrections_count+=1
            if not corrections_flag:
                break
        return (s,a,r,ns,d)
    def update_target_model(self):
        self.train_model(epochs=30,sub_batch_size=6000,verbose=0)
        self.train_model(epochs=1,sub_batch_size=6000,verbose=1)
    def test_model(self,model,X,Y,draw=False):
        #ПОКА НЕ ГОТОВО
        #нормированный rmse по всем выходам сети
        Y_pred = model.predict(X)
        rmse_raw = np.sqrt(metrics.mean_squared_error(Y_pred,Y,multioutput='raw_values'))
        std_arr = np.std(Y,axis=0)
        std_arr[std_arr==0]=1
        rmse_std = rmse_raw/std_arr
        mse = np.mean(rmse_std)
        if draw:
            count = 200
            mini_batch = np.zeros(len(self.s))
            mini_batch[-count:]=1
            mini_batch = np.where(mini_batch)[0]
            s = [self.s[idx] for idx in mini_batch]
            ns = [self.ns[idx] for idx in mini_batch]
            d = [self.d[idx] for idx in mini_batch]
            a = [self.a[idx] for idx in mini_batch]
            r = [self.r[idx] for idx in mini_batch]
            s_arr=np.array(s)
            nsar=np.hstack((np.array(ns)[:,0,:],np.array(a),np.array(r,ndmin=2).T))
            sa=np.hstack((np.array(s)[:,0,:],np.array(a)))
            X = sa
            Y = nsar
            Y_pred = model.predict(X)
            for i in range(Y.shape[1]):
                print('Y'+str(i))
                plt.plot(Y_pred[:,i])
                plt.plot(Y[:,i])
                plt.show()
        return mse
    def test_model_recursive(self,tact_count=70,draw=True):
        #НЕ НУЖНО
        #перемотать время на tact_count тактов назад
        #нормированный rmse по всем выходам сети
        X = np.hstack((np.array(self.ns)[:,0,:],np.array(self.a),np.array(self.r,ndmin=2).T))
        nsar = X[-tact_count:-tact_count+1,:]
        s = []
        a = []
        r = []
        action_by_model_based_one_hot=nsar[0,self.state_size+1:self.state_size+self.action_size+1]
        action_by_model_based = np.argmax(action_by_model_based_one_hot)
        for i in range(tact_count):
            state = nsar[:,:self.state_size]
            a_onehot=np.zeros(self.action_size)
            a_onehot[action_by_model_based]=1
            sa = np.hstack((np.array(state,ndmin=2),np.array(a_onehot,ndmin=2)))
            nsar=self.model_ss.predict(sa)
            s.append(nsar[:self.state_size])
            action_by_model_based_one_hot=nsar[0,self.state_size+1:self.state_size+self.action_size+1]
            action_by_model_based = np.argmax(action_by_model_based_one_hot)
            a.append(action_by_model_based)
            rew=nsar[0][self.state_size+self.action_size:self.state_size+self.action_size+1]
            r.append(rew)
        r_disco = utils.exp_smooth(r,self.discount_factor,len(r))
        if len(r_disco)>0:
            reward_mean_predict = np.mean(r_disco)
        else:
            reward_mean_predict = 0
        reward_mean_fact = np.mean(utils.exp_smooth(X[:,self.state_size+1],self.discount_factor,len(r)))
            
        s = np.array(s)[:,0,:]
        #rmse_raw = np.sqrt(metrics.mean_squared_error(s,X[-tact_count:,:self.state_size+1],multioutput='raw_values'))
        #std_arr = np.std(Y,axis=0)
        #std_arr[std_arr==0]=1
        #rmse_std = rmse_raw/std_arr
        #mse = np.mean(rmse_std)
        
        if draw:
            for i in range(s.shape[1]):
                print('Y'+str(i),(np.mean(s[:,i]),len(s[:,i])),(np.mean(X[-tact_count:,i]),len(X[-tact_count:,i])))
                plt.plot(s[:,i])
                plt.plot(X[-tact_count:,i])
                plt.show()
        return (reward_mean_fact,reward_mean_predict)
    def train_model(self,epochs=1,sub_batch_size=None,verbose=0):
        #ТУТ НАДО ОЧЕНЬ КРУТО НА БАТЧИ РАЗБИТЬ
        if len(self.s) < self.train_start:
            return
        if sub_batch_size is None:
            sub_batch_size = self.batch_size
        batch_size = max(self.batch_size,sub_batch_size)
        batch_size = min(batch_size, len(self.s))
        
        sar = np.hstack((self.s,self.a,self.r))
        
        X1=[]
        Y=[]
        X2=[]
        max_start = self.s.shape[0] - self.planning_horison - self.history_conv_len - 1
        for i in range(batch_size):
            start = int(np.random.rand()*max_start)
            x1 = sar[start:start+self.history_conv_len,:]
            y = sar[start+self.history_conv_len:start+self.history_conv_len+self.planning_horison,:]
            x2 = self.a[start+self.history_conv_len:start+self.history_conv_len+self.planning_horison,:]
            X1.append(x1)
            Y.append(y)
            X2.append(x2)
        
        #тут можно было бы размножить наиболее удачные кейсы    

        if len(self.s) < self.train_start*1.05:
            verbose = True
            epochs*=2
        for i in range(4):
            self.model_ss.fit([np.array(X1),np.array(X2)], Y, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
            if sa.shape[0]==0:
                print(sa.shape[0]==0)
                mse = 500
            else:
                mse = self.test_model(self.model_ss,sa,nsar,draw=False)
            if epochs==1:
                break
            if np.std(r)==0:
                break
            if mse<=0.3: #обучать до тех пор, пока не станет хорошо
                break 
        if verbose:
            if verbose==2:
                self.test_model(self.model_ss,sa,nsar,draw=True)
            print('self-test',mse)