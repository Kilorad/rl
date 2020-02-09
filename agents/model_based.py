import gym
import sys
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
from sklearn import metrics
sys.path.append('./common/')
import utils


class ModelBasedAgent:
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
        self.batch_size = 3000
        self.sub_batch_size=500
        self.train_start = 2000
        self.reward_part_need = 0.3
        self.planning_horison = 810
        
        
        self.count_plans = 17
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
            input_dim = self.state_size + self.action_size 
            out_dim = self.state_size + self.action_size + 1
        model = Sequential()
        model.add(Dense(260, input_dim=input_dim, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        model.add(Dense(260, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        model.add(Dense(260, activation='relu',
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
        if np.random.rand() <= self.epsilon or len(self.s)<self.train_start:
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
        for i in range(actions_count):
            plan[random.randrange(plan_len-1)] = random.randrange(self.action_size)
        plan[0]=random.randrange(self.action_size)
        return plan
    def generate_plans(self,count_plans,actions_count,plan_len):
        plans = [self.generate_plan(actions_count,plan_len) for i in range(count_plans) ]
        return plans
    def estimate_plan(self, state, action_list):
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
            rew=nsar[self.state_size+self.action_size:self.state_size+self.action_size+1]
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
        #нормированный rmse по всем выходам сети
        Y_pred = model.predict(X)
        rmse_raw = np.sqrt(metrics.mean_squared_error(model.predict(X),Y,multioutput='raw_values'))
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
    def train_model(self,epochs=1,sub_batch_size=None,verbose=0):
        if len(self.s) < self.train_start:
            return
        if sub_batch_size is None:
            sub_batch_size = self.sub_batch_size
        batch_size = max(self.batch_size,sub_batch_size)
        batch_size = min(batch_size, len(self.s))
            
        mini_batch = np.random.randint(low=0,high=len(self.s),size=len(self.s))
        #    
        (s,a,r,ns,d) = self.rebalance_data(mini_batch)
        if np.std(r)==0:
            return
        #                   
        batch_size = min(sub_batch_size, len(self.s))
        mini_batch = np.random.randint(low=0,high=len(self.s),size=batch_size)
                             
        s = [self.s[idx] for idx in mini_batch]
        ns = [self.ns[idx] for idx in mini_batch]
        d = [self.d[idx] for idx in mini_batch]
        a = [self.a[idx] for idx in mini_batch]
        r = [self.r[idx] for idx in mini_batch]
        

        if len(self.s) < self.train_start*1.05:
            verbose = True
            epochs*=2
        for i in range(10):
            s_arr=np.array(s)
            nsar=np.hstack((np.array(ns)[:,0,:],np.array(a),np.array(r,ndmin=2).T))
            sa=np.hstack((np.array(s)[:,0,:],np.array(a)))
            self.model_ss.fit(sa, nsar, batch_size=self.batch_size,
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
            self.test_model(self.model_ss,sa,nsar,draw=True)
            print('self-test',mse)