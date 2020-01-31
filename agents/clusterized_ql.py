import gym
import sys
import pylab
import random
import numpy as np
from collections import deque
import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import metrics
sys.path.append('./common/')
import utils


class ClusterQLAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.994
        self.epsilon_min = 0.01
        self.batch_size = 3000
        self.sub_batch_size=500
        self.train_start = 2000
        self.reward_part_need = 0.3
        self.planning_horison = 200
        self.n_clusters=15

        # create replay memory using deque
        memlen = 15000

        # create main model and target model
        self.clusterizer = None

        self.s=deque(maxlen=memlen)
        self.ns=deque(maxlen=memlen)
        self.r=deque(maxlen=memlen)
        self.a=deque(maxlen=memlen)
        self.d=deque(maxlen=memlen)

        # initialize target model
        self.train_model()


    # get action from model using epsilon-greedy policy
    def get_action(self, state, verbose=0):
        if np.random.rand() <= self.epsilon or len(self.s)<self.train_start:
            return random.randrange(self.action_size)
        else:
            # A, для них предсказать Q
            q_predict_array = []
            s=self.clusterizer.predict(np.array(state,ndmin=2))
            a = np.argmax(self.saq_mn[s,:])
            if verbose:
                print('q',self.saq_mn[s,a],a)
            return a
    #далее: q-матрицу составь
    def select_clusterizer(self):
        if self.clusterizer is None:
            n_clusters=self.n_clusters
            self.clusterizer=self.make_clusterizer(n_clusters=n_clusters)
        quality_max=0
        for i in range(10):
            n_clusters = self.clusterizer.n_clusters
            n_clusters += int(n_clusters*np.random.normal(0,0.07))
            clusterizer=self.make_clusterizer(n_clusters)
            (sans_mn,nsr_mn) = self.make_matrix(clusterizer)
            quality=self.check_matrix(sans_mn,nsr_mn)
            if quality>quality_max:
                quality_max = quality
                self.clusterizer = clusterizer
                self.sans_mn=sans_mn
                self.nsr_mn=nsr_mn
    def make_clusterizer(self,n_clusters=8):    
        clusterizer=cluster.KMeans(n_clusters=n_clusters, max_iter=200, verbose=0, random_state=None, n_jobs=5)
        clusterizer.fit(self.s)
        return clusterizer
    def make_matrix(self,clusterizer):
        a_arr = np.array(self.a)
        s_clusters = clusterizer.predict(self.s)
        ns_clusters = clusterizer.predict(self.ns)
        #составить матрицы переходов sa->ns
        sans_mn=[]
        nsr_mn=[]
        for action in range(self.action_size):
            sns_mx = np.zeros((clusterizer.n_clusters,clusterizer.n_clusters))
            for s in range(clusterizer.n_clusters):
                for ns in range(clusterizer.n_clusters):
                    znamen=np.sum((s_clusters==s)&(a_arr==action))
                    if znamen>0:
                        p = np.sum((s_clusters==s)&(ns_clusters==ns)&(a_arr==action))/znamen
                    else:
                        p = np.nan
                    sns_mx[s,ns]=p
            sans_mn.append(sns_mx)
        nsr_mn=np.zeros(clusterizer.n_clusters) + np.nan
        for ns in range(clusterizer.n_clusters):
            idx_full = np.where((ns_clusters==ns))[0]
            r_lst = [self.r[idx] for idx in idx_full] 
            if len(r_lst)>0:
                r_mean=np.mean(r_lst)
                nsr_mn[ns]=r_mean
        return(sans_mn,nsr_mn)
                               
    def check_matrix(self,sans_mn,nsr_mn,diff=False):
        #проверить качество матриц
        sans_quality = 0
        nsr_quality = 0
        for action in range(self.action_size):
            sns_mx = sans_mn[action]
            mean_p = 1/self.action_size #средняя вероятность
            sans_quality += np.nanmean((np.array(sns_mx)-mean_p)**2) #насколько чётко определены переходы
            #+1 редко, -1/7 часто - в идеале. Больше - лучше
        nsr_mn_centered=np.array(nsr_mn)-np.mean(nsr_mn)
        nsr_mn_normed=nsr_mn_centered/(2*np.std(nsr_mn_centered))#+-0.5. Больше - лучше
        nsr_quality=np.mean(nsr_mn_normed**2)#посмотреть, насколько разные r в разных state
        if diff:
            return (sans_quality,nsr_quality)
        else:
            return nsr_quality+sans_quality
    def make_q_iter(self,sans_mn,nsr_mn,saq_mn=None,disco=0.99):
        #в nsr_mn может лежать Q для других итераций
        #Тут нахрен всё неверно
        if (saq_mn is None):
            saq_mn=np.zeros((self.clusterizer.n_clusters,self.action_size))#в ячейках проставим реворды s,a->q
        saq2_mn=np.zeros((self.clusterizer.n_clusters,self.action_size))
        
        for action in range(self.action_size):
            for s in range(self.clusterizer.n_clusters):
                #print(saq_mn.shape,sans_mn[action][s,:].shape,np.dot(sans_mn[action][s,:],saq_mn).shape)
                saq2_mn[s,action] = np.nanmean(np.dot(sans_mn[action][s,:],saq_mn))
                #взвешенная по вероятностям сумма q за следующий такт
        #print(saq2_mn)
        #sar_mn: (s,a) -> action
        maxq = np.max(saq2_mn,axis=0)#argamax(s,a->среднее r за следующий такт)  
        for action in range(self.action_size):
            for s in range(self.clusterizer.n_clusters):
                #print(saq2_mn.shape,nsr_mn.shape)
                #print(nsr_mn.shape,s,maxq.shape,action)
                saq2_mn[s,action]=nsr_mn[s]+maxq[action]*disco  #докинуть ещё r за этот такт
        return saq2_mn
    
    def make_q_full(self,sans_mn,nsr_mn,disco=0.99,horison=10):   
        saq_mn = None
        for i in range(horison):
            saq_mn = self.make_q_iter(sans_mn,nsr_mn,saq_mn,self.discount_factor)
        return saq_mn
                                   
   
    # save sample <s,a,r,s',d> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        a_one_hot = np.zeros(self.action_size)
        a_one_hot[action]=1
        self.s.append(state[0,:])
        self.ns.append(next_state[0,:])
        self.r.append(reward)
        self.a.append(action)
        self.d.append(done)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay       

    
    def update_target_model(self):
        self.train_model(verbose=1,stop=False)

    def train_model(self,stop=True,verbose=0):
        if stop:#костыль
            return
        if len(self.s) < self.train_start:
            return

        self.select_clusterizer()
        (sans_mn,nsr_mn)=self.make_matrix(self.clusterizer)
        self.saq_mn=self.make_q_full(sans_mn,nsr_mn,self.discount_factor,self.planning_horison)
        print ('quality',self.check_matrix(sans_mn,nsr_mn,diff=True))