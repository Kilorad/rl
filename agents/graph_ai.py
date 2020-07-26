import sys
import pylab
import random
import re
import numpy as np
import pandas as pd
from collections import deque
from sklearn.metrics import pairwise
import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
sys.path.append('./common/')
import networkx as nx
import itertools
import utils
import copy

#RL вида (S,S’,A)->R. А R как рассчитывать? По идее, через близость S-после-A и S’. 
#Близость - в смысле мы нормируем все циферки и рассчитываем… Ну пусть косинусную меру.
#Базируется на двойном sarsa - то есть предсказывает и оптимизирует значение advantage

def str_to_array(string):
    numeric_const_pattern = r"""
        [-+]? # optional sign
        (?:
            (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
            |
            (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
        )
        # followed by optional exponent part if desired
        (?: [Ee] [+-]? \d+ ) ?
        """
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    #string like this '[1,2,  3]' to array like this [1,2,3]
    #lst = re.findall('\d+\.\d+',string[1])
    lst = rx.findall(string)
    #распарсить из графа
    lst = np.array([float(l) for l in lst])
    return lst 

def arr_str_to_array(lst):
    lst = [str_to_array(string) for string in lst]
    return np.array(lst)

def arr_to_str(lst):
    if type(lst)==type('abc'):
        return lst
    try:
        lst_new = [lst[i] for i in range(len(lst))]
        #for i in range(len(lst)):
        #    lst[i] += 0.
        #    lst[i] = str(lst[i])
        #    if (lst[i][-2:]=='.0') and (len(lst[i])>2):
        #        lst[i] = lst[i][:-2]
    except Exception:
        pass
    #for i in range(len(lst)):
    #    lst[i] += 0.
    s_lst = str(lst_new)
    s_lst = s_lst.replace('.0 ', '.').replace('.0]', '.]')
    return s_lst.replace(' ',',').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',').replace('\n','').replace('[','').replace(']','')
        
class GoalAgent:
    def __init__(self, state_size, action_size, layers_size=[300,300],deque_len=2000):
        self.render = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # hyper parameters for the Double SARSA
        self.discount_factor = 0.98 #0.98**15 ~ 0.75
        self.learning_rate = 0.0001
        self.epsilon = 0.6
        self.epsilon_decay = 0.996
        #0.6*0.996^1000 = 0.01, то есть за 1000 тактов перестанем рандомить
        self.epsilon_min = 0.01
        self.batch_size = 3000
        self.sub_batch_size=1000
        self.train_start = 1000
        self.reward_part_need = 0.3
        self.planning_horison = 40
        self.layers_size = layers_size
        # create replay memory using deque
        self.s = deque(maxlen=deque_len)
        #self.r - это self.make_cos_distances(s_arr,s_g_arr)
        #целевые состояния:
        self.s_g = deque(maxlen=deque_len)
        #действия в формате one_hot:
        self.a = deque(maxlen=deque_len)
        self.done = deque(maxlen=deque_len)

        # SSR модель - для оценки value
        self.model_ssr = self.build_model('ssr')
        # SSAR модель - для оценки advantage
        self.model_ssar = self.build_model('ssar')

        #инициализировать модельки
        self.train_model(epochs=1)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self,type_mdl):
        if type_mdl=='ssar':
            input_dim = 2*self.state_size + self.action_size 
            out_dim = 1
        elif type_mdl=='ssr':
            input_dim = 2*self.state_size
            out_dim = 1
        model = Sequential()
        model.add(Dense(self.layers_size[0], input_dim=input_dim, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        model.add(Dense(self.layers_size[1], activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        model.add(Dense(out_dim, activation='linear',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model


    # get action from model using epsilon-greedy policy
    def get_action(self, state,state_goal, verbose=0):
        if np.random.rand() <= self.epsilon or len(self.s)<self.train_start:
            return random.randrange(self.action_size)
        else:
            #Перебрать все A, для них предсказать дельта R
            #r_predict_array = []
            ssa_current_array = []
            for a in range(self.action_size):
                a_one_hot = np.zeros(self.action_size)
                a_one_hot[a]=1
                ssa_current=np.concatenate((state[0,:],state_goal[0:],a_one_hot))
                ssa_current=np.array(ssa_current,ndmin=2)
                #ssar-модель работает с дельта r
                #r_predict_array.append(self.model_ssar.predict(ssa_current)[0][0])
                ssa_current_array.append(ssa_current)
            r_predict_array = self.model_ssar.predict(np.array(ssa_current_array)[:, 0, :])
            if verbose:
                print('r_predict_array',r_predict_array)
                
            action_names = ['vx++','vx--','vy++','vy--']
            print(f'go from {np.ravel(state)[1:3]} to {np.ravel(state_goal)[1:3]}, action {action_names[np.argmax(r_predict_array)]}')
            
            return np.argmax(r_predict_array)

    # save sample <s,s_g,a,r,s'> to the replay memory
    def append_sample(self, state, state_goal, action, reward, next_state, done):
        a_one_hot = np.zeros(self.action_size)
        a_one_hot[action]=1
        self.s.append(state)
        self.s_g.append(state_goal)
        self.a.append(a_one_hot)
        self.done.append(done)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def make_cos_distances(self,s_arr,s_g_arr):
        delta_s = s_arr - s_g_arr
        mean_values = np.mean(s_arr,axis=1)
        std_values = np.std(s_arr,axis=1)
        std_values[std_values==0] = 1
        #косинусные меры расстояний между текущим и желаемым state
        #r = pairwise.cosine_similarity(((s_arr.T - mean_values)/std_values).T,((s_g_arr.T - mean_values)/std_values).T)
        r = utils.cosine_similarity(((s_arr.T - mean_values)/std_values),((s_g_arr.T - mean_values)/std_values))
        #какая проблема? Все координаты равнозначны. Важны только распределения - большие дельты могут выйти лишь при сильной неравномерности по одной из координат.
        #не выйдет ли так, что реально важна лишь одна (малое подмножество) координата, а мы оптимизируем все, и той одной пренебрегаем?
        return r
    
    #по смыслу эти награды - расстояния от целевого state до текущего.
    def make_discounted_rewards(self, derivative=True):
        #derivative - брать ли за награду скорость движения к цели
        s_arr = np.array(self.s)
        s_g_arr = np.array(self.s_g)
        self.r = self.make_cos_distances(s_arr.T,s_g_arr.T)
        if derivative:
            self.r[:-1] = self.r[1:] - self.r[:-1]
            self.r[-1] = 0
        
        idx_borders = np.array(self.done)
        #borders - границы между эпизодами и между инструментальными целями
        idx_goal_changed = np.any((s_g_arr - np.roll(s_g_arr,1, axis=0))!=0,axis=1) 
        #цель сменилась. Типа такого: 00001111, и мы выбираем первую единичку
        idx_borders = idx_borders | idx_goal_changed
        self.r_disco = utils.exp_smooth(self.r,self.discount_factor,self.planning_horison,idx_borders)
        #print('self.r_disco', self.r_disco)
        #print('self.r', self.r)

        
    #аугментация данных. Придётся выбросы размножить, потому что реворды очень на них завязаны
    #ну и данные выложим в виде (s,s_g,a,r_disco)
    def rebalance_data(self,s,s_g,action,reward):
        #reward = np.array(self.r_disco)
        #s = np.array(self.s)
        #s_g = np.array(self.s_g)
        #action = np.array(self.a)
        
        mean = np.mean(reward)
        #размножь большие
        idx_big = reward>mean
        if any(idx_big):
            idx_big_num = np.where(idx_big)[0]
            s_add = list(s[idx_big_num])
            s_g_add = list(s_g[idx_big_num])
            action_add = list(action[idx_big_num])
            reward_add = list(reward[idx_big_num])

            initial_part = np.mean(idx_big)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1

            #action.extend([action_add]*multiplication_coef)
            #reward.extend([reward_add]*multiplication_coef)
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                s_g=np.vstack((s_g,s_g_add))
                action = np.concatenate((action,action_add))
                reward = np.concatenate((reward,reward_add))
            #print('initial_part',initial_part,'mean',mean,'multiplication_coef',multiplication_coef)
        #размножь мелкие
        idx_small = reward<mean
        if any(idx_small):
            idx_small_num = np.where(idx_small)[0]
            s_add = list(s[idx_small_num])
            s_g_add = list(s_g[idx_small_num])
            action_add = list(action[idx_small_num])
            reward_add = list(reward[idx_small_num])

            initial_part = np.mean(idx_small)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1

            #action.extend([action_add]*multiplication_coef)
            #reward.extend([reward_add]*multiplication_coef)
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                s_g=np.vstack((s_g,s_g_add))
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
                    s_g_add = list(s_g[idx_act_num])
                    action_add = list(action[idx_act_num])
                    reward_add = list(reward[idx_act_num])
                    #action.extend([action_add]*multiplication_coef)
                    #reward.extend([reward_add]*multiplication_coef)
                    s=np.vstack((s,s_add))
                    s_g=np.vstack((s_g,s_g_add))
                    action = np.concatenate((action,action_add))
                    reward = np.concatenate((reward,reward_add))
            corrections_count+=1
            if not corrections_flag:
                break
        return (s,s_g,action,reward)
    
    def update_target_model(self):
        self.train_model(epochs=30,sub_batch_size=6000,verbose=0)
        self.train_model(epochs=1,sub_batch_size=6000,verbose=1)
        
    def test_model(self,model,X,Y,show_result=False):
        Y_pred = model.predict(X)
        mse = np.mean((Y_pred-Y)**2)
        if show_result:
            return mse, Y_pred
        else:
            return mse
    
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
        s = np.array(self.s)[mini_batch,:]
        s_g = np.array(self.s_g)[mini_batch,:]
        a = np.array(self.a)[mini_batch,:]
        #    
        (s,s_g,a,r) = self.rebalance_data(s,s_g,a,r)
        if np.random.rand()<0.06:
            #написать дамп в файл, потом разберу
            data = np.hstack((np.array(self.s),np.array(self.s_g), np.array(np.argmax(np.array(self.a),axis=1),ndmin=2).T,np.array(self.r,ndmin=2).T,np.array(self.r_disco,ndmin=2).T))
            columns = []
            for i in range(len(self.s[0])):
                columns.append('s'+str(i))
            for i in range(len(self.s_g[0])):
                columns.append('s_g'+str(i))

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
            epochs*=2
        for i in range(7):
            if verbose:
                print('ssr_fit')
            self.model_ssr.fit(np.hstack((s,s_g)), r, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose, )
            mse = self.test_model(self.model_ssr,np.hstack((s,s_g)),r)
            if epochs==1:
                break
            if np.std(r)==0:
                break
            if mse/np.std(r)<=1: #обучать до тех пор, пока не станет хорошо
                break
        print('ssr rmse', np.sqrt(mse),'ssr rmse/std', np.sqrt(mse)/np.std(r), 'ssr rmse/mean', np.sqrt(mse)/np.mean(np.abs(r)), 's.shape', np.shape(s))
        r_ssr_predicted = self.model_ssr.predict(np.hstack((s,s_g)))
        #Предсказать дельту
        delta_r = r-r_ssr_predicted[:,0]
        
        for i in range(5):
            if verbose:
                print('ssar_fit')
            self.model_ssar.fit(np.hstack((s,s_g,a)), delta_r, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
            mse = self.test_model(self.model_ssar,np.hstack((s,s_g,a)),delta_r)
            if epochs==1:
                break
            if np.std(delta_r)==0:
                break
            if mse/np.std(delta_r)<=0.35: #обучать до тех пор, пока не станет хорошо
                break
        print('ssar', np.sqrt(mse),' rmse/std', np.sqrt(mse)/np.std(delta_r), 'ssar rmse/mean', np.sqrt(mse)/np.mean(np.abs(r)))
                

            
#Bычислитель дистанции. (S,S’)->dist. Работает по принципу: ввели 2 состояния, он посчитал число тактов между ними плюс, возможно, штрафы. Сделать метод “обновить” и метод “рассчитать”.
class DistanceMeasure:
    def __init__(self, state_size, layers_size=[300,300], use_bins=False):
        # get size of state and action
        self.state_size = state_size
        self.learning_rate = 0.0001
        self.batch_size = 3000
        self.layers_size = layers_size
        # максимальное теоретическое расстояние между 2 точками. В кадрах
        # если сделать слишком мелким, то не сможем понять, когда из одной точки в другую нельзя попасть
        self.max_distance = 400 
        self.max_thresh = 200 #всё, что больше этого числа, заклипать
        self.pairs_count = 2000
        #штраф за длину ребра. Идеальный маршрут - из коротких рёбер (пусть их и много), 
        #так что мы к длинам рёбер добавляем квадрат длины, умноженный на коэффициент
        #400 + k*400^2 = 400 + k*160000
        #чтобы 3 по 200 перевесило 1 по 400, надо, чтобы 400 + k*160000 = 3*(200 + k*40000), 
        #то есть k=0.005
        self.length_penalty_coef = 0.5
        
        #моделька. Посчитать время между двумя точками.
        input_dim = 2*self.state_size
        if use_bins:
            self.bin_count = 10
            out_dim = self.bin_count
        else:
            out_dim = 1
        model = Sequential()
        model.add(Dense(self.layers_size[0], input_dim=input_dim, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        model.add(Dense(self.layers_size[1], activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        if use_bins:
            model.add(Dense(out_dim, activation='sigmoid',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        else:
            model.add(Dense(out_dim, activation='linear',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.summary()
        if use_bins:
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        else:
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.use_bins = use_bins
        self.model = model
        
    def fit(self,s,done,epochs=1,verbose=0):
        #берём последовательность s
        #берём рандомные пары и рассчитываем время между ними
        done=np.array(done)
        s = np.array(s)
        idx_start = np.random.randint(low=0,high=len(s),size=self.pairs_count)
        #
        idx_len = np.random.randint(low=0,high=self.max_distance,size=self.pairs_count)
        max_idx = int(2*len(idx_len)/3)
        idx_len[0:max_idx] = np.random.randint(low=0,high=self.max_thresh,size=max_idx)#размножить короткие дистанции
        idx_end = idx_start + idx_len
        idx_len[idx_len>=self.max_thresh] = self.max_thresh #клипаем
        idx_end[idx_end>len(s)-1] = len(s)-1
        #выкинуть ситуации, где done попало в кадр
        done_sums = np.array([sum(done[idx_start[i]:idx_end[i]]) for i in range(len(idx_start))])
        idx_start = idx_start[done_sums==0]
        idx_end = idx_end[done_sums==0]
        idx_len = idx_len[done_sums==0]
        
        s1 = s[idx_start,:]
        s2 = s[idx_end,:]
        if verbose:
            print('(s,goal)->len fit')
        for i in range(40):
            self.model.fit(np.hstack((s1,s2)), idx_len, batch_size=1000, epochs=epochs*20, verbose=verbose)
            dist_predict = self.model.predict(np.hstack((s1,s2)))
            mse = np.mean((dist_predict - idx_len)**2)
            mse_mean = np.sqrt(mse)/np.mean(np.abs(idx_len))
            if mse_mean<0.5:
                break
        print('DistanceMeasure rmse', np.sqrt(mse),'rmse/mean = ', np.sqrt(mse)/np.mean(np.abs(idx_len)), 'rmse/std = ', np.sqrt(mse)/np.std(idx_len), 'shape', len(idx_len))
        
    def predict(self,s1,s2,epochs=1,verbose=0):
        #сразу адаптируем под вектора
        pred = self.model.predict(np.hstack((s1,s2)))
        pred[pred<0] = self.max_distance #так как явная ошибка
        pred = pred + self.length_penalty_coef*pred**2 #штрафуем за то, что длинные блоки
        return pred

#Cборщик графов. Он берёт n рандомных состояний и рассчитывает расстояние (направленное!) между каждыми двумя. Есть опция “добавить вершину” (с расчётом всех рёбер) и “удалить вершину”. А так же пересчитать рёбра. Текущий state добавляется, целевые state должны быть добавлены изначально.
class StatesGraph:
    def __init__(self,state_size):
        self.graph = nx.DiGraph()
        self.dist_meas = DistanceMeasure(state_size)
        self.goal_states = []
        self.current_goal = None
        self.edges = set()
        self.nodes = set()
        #Рёбра. Пруним граф: рёбра только таких размеров и используем.
        self.filtration_min_rel = 0.001
        self.filtration_max_rel = 0.2
        self.filtration_min_abs = 5
        self.filtration_max_abs = 60
        
        self.count_additional_edges = 20
        self.part_additional_edges = 0.15
        
        #логирование
        self.time_logging = {'routing':[], 'learning':[], 'acting':[], 'node_adding':[]}
    def fit_dist_measure(self,s,done,epochs=1,verbose=0):
        self.dist_meas.fit(s,done,epochs,verbose)
    #def add_node_list_safety(self,state_list,max_points_abs=500, max_points_part=0.2,filtration_min_rel=0.05,filtration_min_abs=5):
    def add_node_list_safety(self,state_list,max_points_abs=1e2, max_points_part=1):
        #зарядить сюда все вершины. Единственно безопасный вариант!
        #max_points - сколько максимум вершин добавляем из выборки (абсолютное и относительное значение)
        t_start = pd.Timestamp.now()
        l = len(state_list)
        idx = random.sample(range(l), int(min(int(l*max_points_part),max_points_abs)))
        state_list_sparce = [state_list[i] for i in idx]
        self.add_node_list(state_list_sparce)
        self.time_logging['node_adding'].append((pd.Timestamp.now()-t_start)/pd.Timedelta(seconds=1))
        
    def add_node_list(self,state_list, edges_drop_part=0.7):
        if len(state_list)==0:
            print('add_node_list error: empty state_list')
            return
        #Опасно! Может повесить комп!
        #filtration_min - минимальное расстояние, относительное и абсолютное
        #ноды должны быть hashable. А значит - ну пусть string
        old_nodes = list(self.graph.nodes)
        old_nodes = arr_str_to_array(old_nodes)
        if (len(state_list) > 0) and (type(state_list[0])==type('ab')):
            state_list = arr_str_to_array(state_list)
        if (len(old_nodes) > 0) and (type(old_nodes[0])==type('ab')):
            #old_nodes - строки. Надо распотрошить в arrays
            old_nodes = arr_str_to_array(old_nodes)
        #затем пробежаться по всем новым state и всем старым и создать пары старый-новый
        edges_coord = list(itertools.product(old_nodes, state_list)) + list(itertools.product(state_list, old_nodes)) + list(itertools.product(state_list, state_list))
        edges_coord_np = np.array(edges_coord)
        while 1:
            idx_drop = np.random.rand(edges_coord_np.shape[0])<edges_drop_part
            whr = np.where(~idx_drop)[0]
            if len(whr)>0:
                break
            print('empty ~idx_drop, retry')
            if np.random.rand()<0.002:
                print('empty ~idx_drop, random break',edges_coord_np)
                break
        if np.mean(idx_drop)==1:
            #значит, надо дропнуть всё. Это плохо. Отмена
            idx_drop = np.random.rand(edges_coord_np.shape[0])<0
            whr = np.where(~idx_drop)[0]
        if len(whr)==0:
            print('old_nodes',old_nodes)
            print('state_list',state_list)
                
        print('predicting distances',pd.Timestamp.now())
        #try:
        #    print('state_list', np.array(state_list).shape)
        #except Exception:
        #    pass
        #try:
        #    print('old_nodes', np.array(old_nodes).shape)
        #except Exception:
        #    pass
        print('edges_coord_np', edges_coord_np[0,0])
        edges_values = self.dist_meas.predict(edges_coord_np[whr,0,:],edges_coord_np[whr,1,:])
        print('edges_values predicted: min max std avg', np.min(edges_values), np.max(edges_values), np.std(edges_values), np.mean(edges_values),'len edges_values', len(edges_values), pd.Timestamp.now())
        #индексы рёбер, которые приходят в целевые точки
        idx_goal_edges = []
        if len(edges_coord_np[whr,1,:].shape)==2:
            edges_coord_np_whr = edges_coord_np[whr,1,:]
        for state in self.goal_state_list:
            if len(idx_goal_edges)==0:
                idx_goal_edges = state==edges_coord_np_whr
            else:
                idx_goal_edges = (idx_goal_edges)|(state==edges_coord_np_whr)
        
        idx = (edges_values>self.filtration_min_abs) &  (edges_values<self.filtration_max_abs) & (edges_values>np.percentile(edges_values,self.filtration_min_rel*100)) & (edges_values<np.percentile(edges_values,100-self.filtration_max_rel*100))|(idx_goal_edges)
        idx = np.where(idx)[0]
        #print('add2node state_list',state_list)
        for i in idx:
            #print('str(edges_coord[i][0])',arr_to_str(edges_coord[i][0]))
            #print('str(edges_coord[i][1])',arr_to_str(edges_coord[i][1]))
            self.graph.add_node(arr_to_str(edges_coord[i][0]))
            self.graph.add_node(arr_to_str(edges_coord[i][1]))
            self.graph.add_edge(arr_to_str(edges_coord[i][0]),arr_to_str(edges_coord[i][1]),length=edges_values[i][0])
        print('nodes_count', len(self.graph.nodes), 'edges_count', len(self.graph.edges))
    def rewrite_edges(self):
        #провести новые рёбра, обновить веса рёбер и дропнуть короткие рёбра
        #filtration_min - минимальное расстояние, относительное и абсолютное
        nodes = list(self.graph.nodes)
        if len(nodes)==0:
            return
        nodes = arr_str_to_array(nodes)
        edges_coord = list(itertools.product(nodes, nodes))
        edges_coord_np = np.array(edges_coord)
        edges_values = self.dist_meas.predict(edges_coord_np[:,0,:],edges_coord_np[:,1,:])
        
        #индексы рёбер, которые приходят в целевые точки
        idx_goal_edges = []
        for state in self.goal_state_list:
            if len(idx_goal_edges)==0:
                idx_goal_edges = state==edges_coord_np[:,1,:]
            else:
                idx_goal_edges = (idx_goal_edges)|(state==edges_coord_np[:,1,:])
        
        idx = (edges_values>self.filtration_min_abs) &  (edges_values<self.filtration_max_abs) & (edges_values>np.percentile(edges_values,self.filtration_min_rel*100)) & (edges_values<np.percentile(edges_values,100-self.filtration_max_rel*100))|(idx_goal_edges)
        #очистить граф
        self.graph_new = nx.DiGraph()
        for i in np.where(idx)[0]:
            try:
                self.graph_new.add_node(arr_to_str(edges_coord_np[i,0,:]))
                self.graph_new.add_node(arr_to_str(edges_coord_np[i,1,:]))
                self.graph_new.add_edge(arr_to_str(edges_coord_np[i,0,:]),arr_to_str(edges_coord_np[i,1,:]),length=edges_values[i,:])
            except Exception as e:
                print(e)
                print('edges_coord_np.shape', edges_coord_np.shape)
                print('edges_values', edges_values.shape)
                print('error in graph.add_node or graph.add_edge',e,edges_coord_np[i,0,:],edges_coord_np[i,1,:],edges_values[i,:])
        self.graph = self.graph_new
        return edges_values
    
    def add_goal_states(self,state_list):
        #добавить целевые состояния
        self.goal_states.append(state_list)
        self.add_node_list(self,state_list)
    
    def add_cur_state(self,state):
        self.cur_state = arr_to_str(state)
    
    def make_route_to_all_states(self,s1,s2_list):
        #посчитать пути ко всем целям
        t_start_local = pd.Timestamp.now()
        s1_num = s1
        s1 = arr_to_str(np.ravel(np.array(s1)))
        nodes = list(self.graph.nodes)
        if not (s1 in nodes):
            self.add_node_list([s1])
        nodes = list(self.graph.nodes)    
        path_list = []
        path_length_list = []
        for s2 in s2_list:
            s2_num = s2
            s2 = arr_to_str(s2)
            if not (s1 in nodes):
                self.add_node_list([s1])
            if not (s2 in nodes):
                self.add_node_list([s2])
            nodes = list(self.graph.nodes)
            #print('s1',s1)
            #print('s1 in graph', np.isin(s1,self.graph.nodes))
            #print('s2',s2)
            #print('s2 in graph', np.isin(s2,self.graph.nodes))
            try:
                print('s1,s2',s1, s2)
                path = nx.algorithms.shortest_path(self.graph,s1,s2,weight='length')
            except Exception:
                #по какой-то непонятной причине у нас эти 2 точки не связаны никак. 
                print('cannot find path. s1 in graph:', s1 in nodes, 's2 in graph:', s2 in nodes)
                #if not s2 in nodes:
                #    print('nodes', nodes)
                #    print(s2)
                #выбрать рандомные несколько точек
                count_additional_edges = np.max([self.count_additional_edges,self.part_additional_edges*len(nodes)])
                count_additional_edges = int(count_additional_edges)
                #проложить путь от нас до них и от них до цели
                print('making new edges to s2, count_additional_edges', count_additional_edges, 'nodes', len(nodes))
                for i in range(count_additional_edges):
                    id_node_middle = int(np.random.rand()*(len(nodes) - 1))
                    node_middle = nodes[id_node_middle]
                    node_middle = str_to_array(node_middle)
                    dist = self.dist_meas.predict(np.array(node_middle,ndmin=2),np.array(s2_num,ndmin=2) )
                    dist = dist[0][0]
                    self.graph.add_edge(arr_to_str(node_middle),s2,
                                    length=dist)
                    dist = self.dist_meas.predict(np.array(s1_num,ndmin=2),np.array(node_middle,ndmin=2) )
                    dist = dist[0][0]
                    self.graph.add_edge(s1,arr_to_str(node_middle),
                                    length=dist)
                    #сделали дугу с 1 точкой, сделаем теперь с 2
                    id_node_middle_2 = int(np.random.rand()*(len(nodes) - 1))
                    node_middle_2 = nodes[id_node_middle_2]
                    node_middle_2 = str_to_array(node_middle_2)
                    dist = self.dist_meas.predict(np.array(node_middle,ndmin=2),np.array(node_middle_2,ndmin=2) )
                    dist = dist[0][0]
                    self.graph.add_edge(arr_to_str(node_middle),arr_to_str(node_middle_2),
                                    length=dist)
                    dist = self.dist_meas.predict(np.array(node_middle_2,ndmin=2),np.array(s2_num,ndmin=2) )
                    dist = dist[0][0]
                    self.graph.add_edge(arr_to_str(node_middle_2),s2,
                                    length=dist)
                try:
                    print('s1,s2',s1, s2)
                    path = nx.algorithms.shortest_path(self.graph,s1,s2,weight='length')
                    print('path ready')
                except Exception as e:
                    print('cannot!', e)
                    #Ок, делаем edge напрямую из одной точки в другую
                    dist = self.dist_meas.predict(np.array(s1_num,ndmin=2),np.array(s2_num,ndmin=2) )
                    dist = dist[0][0]
                    self.graph.add_node(s1)
                    self.graph.add_node(s2)
                    self.graph.add_edge(s1,s2,
                                        length=dist)
                    print('s1,s2',s1, s2)
                    path = nx.algorithms.shortest_path(self.graph,s1,s2,weight='length')
            
            #здесь можно провести какую-нибудь хитрую проверку 
            #на предмет "а не проходит ли граф через какое-то совсем плохое состояние"
            #
            #Дальше рассчитываем длины маршрутов
            path_length = nx.algorithms.shortest_path_length(self.graph,s1,s2,weight='length')
            path_list.append(path)
            path_length_list.append(path_length)
        try:
            argmin = np.argmin(np.array(path_length_list)) #кратчайший маршрут
            path = path_list[argmin]
        except Exception:
            #делаем тупой наивный маршрут
            print('cannot find route')
            path = [s1,s2_list[int(np.random.rand()*(len(s2_list)))]]
        self.time_logging['routing'].append((pd.Timestamp.now()-t_start_local)/pd.Timedelta(seconds=1))
        return path

#сам ИИ
#состоит из графа, RL и простенькой схемы, маршутизирующей всё это
#методы:  сделать действие, добавить новые данные, train_model (каждый такт), update_target_model (каждый эпизод)
class GraphAI(StatesGraph):
    def __init__(self,state_size, action_size,layers_size=[200,200],goal_state_list=[]):
        self.render = False
        self.action_size=action_size
        self.state_size=state_size
        #цель, которую мы записываем при логировании. По смыслу то же, что и current goal. Но она должна
        #быть определена, если current goal не определена, и должна меняться не каждый ход, а раз в несколько ходов
        self.write_goal = None 
        
        #turn - это про то, как добавлять точки каждый ход. Вероятность и число
        self.p_add_points_turn=0.06
        self.max_points_abs_turn=5
        self.max_points_part_turn=0.01
        #episode - это про то, как добавлять точки каждый ход. Вероятность и число
        self.p_add_points_episode=1
        self.max_points_abs_episode=20
        self.max_points_part_episode=0.2

        
        #окно измерения скорости сближения
        self.window_speed_measure = 8
        
        #если скорость сближения меньше, чем эта, то меняем цель
        #0.02 - значит, тактов за 50 идеальное совпадение получим с нуля, но я бы на это вообще не рассчитывал
        #скорее, это детектор того, что мы уже не чётко сближаемся, а в шумы воткнулись
        #минимальная допустимая скорость движения по ребру (иначе меняем цель)
        self.min_window_cos_speed = 0.02
        #минимальное время попыток достигать этой локальной цели
        self.min_try_time = 7
        
        super().__init__(state_size)
        deque_len = 2000
        self.rl = GoalAgent(state_size, action_size, layers_size,deque_len=deque_len)
        
        
        
        #целевые состояния. Куда наводиться.
        self.goal_state_list=goal_state_list
        
        #реворды. У нас будет альтернативный способ поиска целевых состояний.
        
        self.reward = deque(maxlen=deque_len)
        self.cos_distance = deque(maxlen=deque_len) #насколько хорошо мы приближаемся к цели
        self.goal_changed = deque(maxlen=deque_len) #сменилась ли локальная цель
        
        #среднее и std по состояниям. Нужно, чтобы рассчитывать нормированное косинусное расстояние
        self.mean_std_s = None
        #иногда надо обновлять
        self.frequency_mean_std_update = 0.02
        
        self.auto_aiming_reward_quantile = 1 #1 - значит, мы обрубаем эту фичу
        
        #частота, с которой мы проводим ревизию верхнеуровневых goal
        self.drop_goals_by_reward = 0.01
        
        
    def get_action(self, state, verbose=0):
        #Использовать rl
        t_start_local = pd.Timestamp.now()
        goal_changed = False
        for key in self.time_logging.keys():
            print(key, np.mean(self.time_logging[key]))
        if 't_start' in globals():
            print('A0',pd.Timestamp.now()-globals()['t_start'])
        globals()['t_start'] = pd.Timestamp.now()
        #print('AA',pd.Timestamp.now()-globals()['t_start'])
        globals()['t_start'] = pd.Timestamp.now()
        if len(self.rl.s)<self.rl.train_start:
            action = random.randrange(self.action_size)
        else:    
            #назначить текущую goal
            #если цели ещё нет
            #или если скорость приближения мала, но не слишком сразу после назначения
            end_id_distance_lst = np.where(self.goal_changed)[0]
            if len(end_id_distance_lst)!=0:
                end_id_distance = end_id_distance_lst[-1]
            else:
                end_id_distance = 0
            #сколько мы достигали этой цели
            #print('BB',pd.Timestamp.now()-globals()['t_start'])
            globals()['t_start'] = pd.Timestamp.now()
            try_time_fact = len(self.goal_changed) - end_id_distance 
            len_window_distance = min(try_time_fact,self.window_speed_measure)           
            speed_closing_cos_dist = (self.cos_distance[-1]+self.cos_distance[-2]) - (self.cos_distance[-len_window_distance]+self.cos_distance[-len_window_distance+1])
            #speed_closing_cos_dist - если положительное, то мы сближаемся, если отрицательное, то отдаляемся
            #print('speed_closing_cos_dist',speed_closing_cos_dist,pd.Timestamp.now())
            #print('self.current_goal',self.current_goal,pd.Timestamp.now())
            #print('CC',pd.Timestamp.now()-globals()['t_start'])
            globals()['t_start'] = pd.Timestamp.now()
            if (self.current_goal is None) or ((speed_closing_cos_dist<self.min_window_cos_speed) and (try_time_fact>self.min_try_time)):
                #обновить цель
                route = self.make_route_to_all_states(state,self.goal_state_list)
                self.route_in_graph = route
                #распарсить из графа
                if len(route)>2:
                    print('route>2',route)
                try:
                    lst = str_to_array(route[1])
                except Exception:
                    print('error in routing')
                    lst = str_to_array(route[0])
                #print('state:',state,'route:',route,'lst:',lst,pd.Timestamp.now())
                self.current_goal = lst
                goal_changed = True
                print('update local target. Speed_closing_cos_dist', speed_closing_cos_dist, 'try_time_fact', try_time_fact)
            #print('DD',pd.Timestamp.now()-globals()['t_start'])
            globals()['t_start'] = pd.Timestamp.now()
            action = self.rl.get_action(state,self.current_goal)
            #print('EE',pd.Timestamp.now()-globals()['t_start'])
            globals()['t_start'] = pd.Timestamp.now()
        if (self.mean_std_s is None) or (np.random.rand()<self.frequency_mean_std_update):
            if len(self.rl.s)>3:
                #обновить цифры для нормализации
                s_arr = np.array(self.rl.s)
                mean_values = np.mean(s_arr,axis=0)
                std_values = np.std(s_arr,axis=0)
                std_values[std_values==0] = 1
                self.mean_std_s = [mean_values,std_values]
        if np.random.rand()<0.08:
            #убрать всю очевидную хрень
            try:
            #if 1:
                self.rewrite_edges()
                print('edges rewrite success')
            except Exception as e:
                print('cannot rewrite edges', e)
        #print('FF',pd.Timestamp.now()-globals()['t_start'])
        globals()['t_start'] = pd.Timestamp.now()
        if (len(self.rl.s)>3) and (not (self.current_goal is None)):
            cos_dist = utils.cosine_similarity((state - self.mean_std_s[0])/self.mean_std_s[1],(self.current_goal - self.mean_std_s[0])/self.mean_std_s[1])
            print('GG',pd.Timestamp.now()-globals()['t_start'])
            globals()['t_start'] = pd.Timestamp.now()
        else:
            cos_dist = 1
        self.cos_distance.append(cos_dist)
        self.goal_changed.append(goal_changed)
        #print('HH',pd.Timestamp.now()-globals()['t_start'])
        globals()['t_start'] = pd.Timestamp.now()
        
        self.time_logging['acting'].append((pd.Timestamp.now()-t_start_local)/pd.Timedelta(seconds=1))
        
        return action
            
    # save sample <s,s_g,a,r,s'> to the replay memory of rl
    #у графа даже своей replay-памяти нет - а нафиг дублирование (можно сделать, если это будет важно)
    def append_sample(self, state, action, reward, next_state, done):
        if (self.current_goal is None) and (not (self.write_goal is None)):
            #self.current_goal ещё нет, а write_goal уже был
            if np.random.rand()<1/12.:
                self.write_goal = None#чтобы дальше перевыбрать
        else:
            #это либо инициализация, либо current_goal и так есть
            self.write_goal = self.current_goal

        #задать рандомную цель, чтобы что-то логировалось, и хоть как-то измерялись расстояния
        if (self.write_goal is None):
            idx = int(np.random.rand()*(len(self.rl.s)-1))
            if idx>0:
                self.write_goal=self.rl.s[idx]
            else:
                self.write_goal=state
                    
        self.rl.append_sample(state.ravel(), self.write_goal.ravel(), action, reward, next_state, done)
        self.reward.append(reward)
        
        
    def train_model(self,epochs=2,sub_batch_size=6000,verbose=0,p_default=0.1):
        t_start_local = pd.Timestamp.now()
        if len(self.rl.s) < self.rl.train_start:
            return
        #p_default - это вероятность запуска обучения. Мы не хотим каждый кадр
        ran = np.random.rand()
        if (ran > p_default) & (epochs<5) & (len(self.rl.s) > self.rl.train_start + 2):
            return
        
        max_points_part = self.max_points_part_turn   
        max_points_abs = self.max_points_abs_turn
        
        ran = np.random.rand()
        if (ran < p_default):
            #print('fit_dist_measure start',pd.Timestamp.now())
            self.fit_dist_measure(self.rl.s,self.rl.done,epochs=epochs,verbose=0)
            #print('fit_dist_measure end',pd.Timestamp.now())
        if len(self.goal_state_list)==0:
            #нет целевых состояний
            #добавить reward-ы в качестве целевых точек.
            quantile = np.percentile(np.array(self.reward),self.auto_aiming_reward_quantile*100)
            if np.std(self.reward)==0:
                reward_arr = self.reward + np.random.rand(self.reward.shape)
            else:
                reward_arr = self.reward
            idx = np.array(self.reward)>quantile
            idx_num = np.where(idx)[0]
            #это те точки, что надо добавить
            #print('add states', len(self.rl.s))
            state_list = [self.rl.s[i] for i in idx_num]
            #self.add_node_list_safety(state_list,self.max_points_abs_episode, 1,filtration_min_rel,filtration_min_abs)
            self.add_node_list_safety(state_list)
            self.goal_state_list.extend(state_list)
        if np.random.rand()<self.drop_goals_by_reward:
            #print('aims rewrite start',pd.Timestamp.now())
            #пересмотр верхнеуровневых целей. Отбросить те, где реворд мелкий.
            #реворд - последний столбец в state
            quantile = np.percentile(np.array(self.reward),self.auto_aiming_reward_quantile*100)
            goal_state_list_new = []
            for goal in self.goal_state_list:
                if goal[-1]>=quantile:
                    goal_state_list_new.append(goal)
            self.goal_state_list = goal_state_list_new
            #print('aims rewrite end',pd.Timestamp.now())
        
        #добавить новые точки в граф
        if (np.random.rand()<self.p_add_points_turn) or (self.graph.number_of_nodes()==0):
            #self.add_node_list_safety(self.rl.s,max_points_abs, max_points_part,filtration_min_rel,filtration_min_abs)
            #print('graph append start',pd.Timestamp.now())
            self.add_node_list_safety(self.rl.s)
            #print('graph append end',pd.Timestamp.now())
        #обучить rl
        #print('graph train start',pd.Timestamp.now())
        self.rl.train_model(epochs,sub_batch_size,verbose)
        self.time_logging['learning'].append((pd.Timestamp.now()-t_start_local)/pd.Timedelta(seconds=1))
        
    def update_target_model(self):
        if len(self.rl.s) < self.rl.train_start:
            return
        #добавить reward-ы в качестве целевых точек.
        quantile = np.percentile(np.array(self.reward),self.auto_aiming_reward_quantile*100)
        idx = np.array(self.reward)>quantile
        idx_num = np.where(idx)[0]
        #это те точки, что надо добавить
        state_list = [self.rl.s[i] for i in idx_num]
        #self.add_node_list_safety(state_list,self.max_points_abs_episode, 1,self.quantile_edge_min,self.min_edge)
        self.add_node_list_safety(state_list)
        self.goal_state_list.extend(state_list)
        #
        self.train_model(epochs=30,sub_batch_size=6000,verbose=0)
        self.train_model(epochs=1,sub_batch_size=6000,verbose=1)
        
    def debug_info(self, frames=200,pairs_count=200, graph_layout='xy', show_full_graph=True):
        #SSR SSAR
        s = np.array(self.rl.s)
        s_g = np.array(self.rl.s_g)
        a = np.array(self.rl.a)
        r = np.array(self.rl.r_disco)
        done=np.array(self.rl.done)
        mse, y_pred = self.rl.test_model(self.rl.model_ssr,np.hstack((s,s_g)),r, show_result=True)
        print(f'SSR model: rmse/mean:{np.sqrt(mse)/np.mean(np.abs(r))}')
        delta_r = r - y_pred
        mse, y_pred_boost = self.rl.test_model(self.rl.model_ssar,np.hstack((s,s_g,a)),delta_r, show_result=True)
        
        print(f'SSAR model: rmse/mean:{np.sqrt(mse)/np.mean(np.abs(delta_r))}')
        if len(y_pred)<=frames:
            frames = len(y_pred) - 1
        plt.plot(r[-frames:])
        plt.plot(y_pred[-frames:])
        plt.plot(y_pred[-frames:] + y_pred_boost[-frames:])
        plt.show()
        
        #DistanceMeasure
        idx_start = np.random.randint(low=0,high=len(s),size=pairs_count)
        #
        idx_len = np.random.randint(low=0,high=self.dist_meas.max_distance,size=pairs_count)
        idx_len = np.sort(idx_len)
        idx_end = idx_start + idx_len
        idx_end[idx_end>len(s)-1] = len(s)-1
        #выкинуть ситуации, где done попало в кадр
        done_sums = np.array([sum(done[idx_start[i]:idx_end[i]]) for i in range(len(idx_start))])
        idx_start = idx_start[done_sums==0]
        idx_end = idx_end[done_sums==0]
        idx_len = idx_len[done_sums==0]
        
        s1 = s[idx_start,:]
        s2 = s[idx_end,:]

        dist_predict = self.dist_meas.model.predict(np.hstack((s1,s2)))
        mse = np.mean((dist_predict - idx_len)**2)

        print('DistanceMeasure rmse', np.sqrt(mse),'rmse/mean = ', np.sqrt(mse)/np.mean(np.abs(idx_len)), 'rmse/std = ', np.sqrt(mse)/np.std(idx_len), 'shape', len(idx_len))
        plt.plot(idx_len)
        plt.plot(dist_predict)
        plt.show()
        
        #отрисовать граф
        if graph_layout=='xy':
            #self.graph.edges -> coords
            #edges = agent.graph.edges(data=True)
            #for edge in edges:
            #    node0 = edge[0]
            #    node1 = edge[1]
            #    node0 = str_to_array(node0)
            #    node1 = str_to_array(node1)
            #    len_edge = edge[2]['length']
            if show_full_graph:
                print('FULL GRAPH')
                nodes = list(self.graph.nodes)
                layout = {}
                for node in nodes:
                    node_ar = str_to_array(node)
                    layout[node] = node_ar[1:]
                nx.draw_networkx(self.graph,pos=layout,node_size=50,with_labels=False)

            layout = {}
            nodes = list(self.graph.nodes)
            edges = []
            for i in range(len(self.route_in_graph) - 1):
                edges.append((self.route_in_graph[i],self.route_in_graph[i + 1]))
            #
            path = self.graph.edge_subgraph(edges)
            #for node in nodes:
            #    path.add_node(node)
            #for edge in edges:
            #    self.graph.edge
            #    path.add_edge(edge[0], edge[1], length=)
            
                
            print('PATH',list(path.edges(data=True)))
            for node in nodes:
                node_ar = str_to_array(node)
                layout[node] = node_ar[1:]
            nx.draw_networkx(path,pos=layout,node_size=50,with_labels=True,node_color='red',edge_color='red')
            plt.show()
            for node in nodes:
                node_ar = str_to_array(node)
                layout[node] = node_ar[1:]
            nx.draw_networkx(path,pos=layout,node_size=50,with_labels=True,node_color='red',edge_color='red')
            plt.show()
        
