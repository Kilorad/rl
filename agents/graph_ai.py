import sys
import pylab
import random
import numpy as np
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

#RL вида (S,S’,A)->R. А R как рассчитывать? По идее, через близость S-после-A и S’. 
#Близость - в смысле мы нормируем все циферки и рассчитываем… Ну пусть косинусную меру.
#Базируется на двойном sarsa - то есть предсказывает и оптимизирует значение advantage
class GoalAgent:
    def __init__(self, state_size, action_size, layers_size=[100,100]):
        self.render = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # hyper parameters for the Double SARSA
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
        deque_len = 10000
        self.s = deque(maxlen=deque_len)
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
    def get_action(self, state,state_goal, verbose=0):
        if np.random.rand() <= self.epsilon or len(self.s)<self.train_start:
            return random.randrange(self.action_size)
        else:
            #Перебрать все A, для них предсказать дельта R
            r_predict_array = []
            for a in range(self.action_size):
                a_one_hot = np.zeros(self.action_size)
                a_one_hot[a]=1
                ssa_current=np.concatenate((state[0,:],state_goal[0:],a_one_hot))
                ssa_current=np.array(ssa_current,ndmin=2)
                #ssar-модель работает с дельта r
                r_predict_array.append(self.model_ssar.predict(ssa_current)[0][0])
            if verbose:
                print('r_predict_array',r_predict_array)
            return np.argmax(r_predict_array)

    # save sample <s,s_g,a,r,s'> to the replay memory
    def append_sample(self, state, state_goal, action, reward, next_state, done):
        a_one_hot = np.zeros(self.action_size)
        a_one_hot[action]=1
        sar_curr = np.concatenate((state[0,:],a_one_hot,[reward],[done]))
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
        #1/0
        #какая проблема? Все координаты равнозначны. Важны только распределения - большие дельты могут выйти лишь при сильной неравномерности по одной из координат.
        #не выйдет ли так, что реально важна лишь одна (малое подмножество) координата, а мы оптимизируем все, и той одной пренебрегаем?
        return r
    
    #по смыслу эти награды - расстояния от целевого state до текущего.
    def make_discounted_rewards(self):
        s_arr = np.array(self.s)[:,0,:]
        s_g_arr = np.array(self.s_g)[:,0,:]
        self.r = self.make_cos_distances(s_arr,s_g_arr)
        
        idx_borders = np.array(self.done)
        #borders - границы между эпизодами и между инструментальными целями
        idx_goal_changed = np.any((s_g_arr - np.roll(s_g_arr,1))!=0,axis=1) 
        #цель сменилась. Типа такого: 00001111, и мы выбираем первую единичку
        idx_borders = idx_borders | idx_goal_changed
        self.r_disco = utils.exp_smooth(self.r,self.discount_factor,self.planning_horison,idx_borders)

        
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
        
    def test_model(self,model,X,Y):
        mse = np.mean((model.predict(X)-Y)**2)
        return mse
    
    def train_model(self,epochs=1,sub_batch_size=None,verbose=0):
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
        s = np.array(self.s)[mini_batch,0,:]
        s_g = np.array(self.s_g)[mini_batch,0,:]
        a = np.array(self.a)[mini_batch,:]
        #    
        (s,s_g,a,r) = self.rebalance_data(s,s_g,a,r)
        #Это уже минибатч. Размноженный
        if np.std(r)==0:
            #Нечего тут учить
            return

        if len(self.s) < self.train_start*1.05:
            #инициализация
            verbose = True
            epochs*=2
        for i in range(10):
            self.model_ssr.fit(np.hstack((s,s_g)), r, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
            mse = self.test_model(self.model_ssr,np.hstack((s,s_g)),r)
            if epochs==1:
                break
            if np.std(r)==0:
                break
            if mse/np.std(r)<=1: #обучать до тех пор, пока не станет хорошо
                break
        r_ssr_predicted = self.model_ssr.predict(np.hstack((s,s_g)))
        #Предсказать дельту
        delta_r = r-r_ssr_predicted[:,0]
        
        for i in range(10):
            self.model_ssar.fit(np.hstack((s,s_g,a)), delta_r, batch_size=self.batch_size,
                           epochs=epochs, verbose=verbose)
            mse = self.test_model(self.model_ssar,np.hstack((s,s_g,a)),delta_r)
            if epochs==1:
                break
            if np.std(delta_r)==0:
                break
            if mse/np.std(delta_r)<=0.35: #обучать до тех пор, пока не станет хорошо
                break
                
        if verbose:
            print('delta_r',np.std(delta_r),'r',np.std(r))

            
#Bычислитель дистанции. (S,S’)->dist. Работает по принципу: ввели 2 состояния, он посчитал число тактов между ними плюс, возможно, штрафы. Сделать метод “обновить” и метод “рассчитать”.
class DistanceMeasure:
    def __init__(self, state_size, layers_size=[100,100]):
        # get size of state and action
        self.state_size = state_size
        self.learning_rate = 0.001
        self.batch_size = 3000
        self.layers_size = layers_size
        # максимальное теоретическое расстояние между 2 точками. В кадрах
        # если сделать слишком мелким, то не сможем понять, когда из одной точки в другую нельзя попасть
        self.max_distance = 500 
        self.pairs_count = 2000
        
        #моделька. Посчитать время между двумя точками.
        input_dim = 2*self.state_size
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
        self.model = model
        
    def fit(self,s,done,epochs=1,verbose=0):
        #берём последовательность s
        #берём рандомные пары и рассчитываем время между ними
        done=np.array(done)
        s = np.array(s)
        idx_start = np.random.randint(low=0,high=len(s),size=self.pairs_count)
        #
        idx_len = np.random.randint(low=0,high=self.max_distance,size=self.pairs_count)
        idx_end = idx_start + idx_len
        idx_end[idx_end>len(s)-1] = len(s)-1
        #выкинуть ситуации, где done попало в кадр
        done_sums = np.array([sum(done[idx_start[i]:idx_end[i]]) for i in range(len(idx_start))])
        idx_start = idx_start[done_sums==0]
        idx_end = idx_end[done_sums==0]
        idx_len = idx_len[done_sums==0]
        
        s1 = s[idx_start,0,:]
        s2 = s[idx_end,0,:]
        self.model.fit(np.hstack((s1,s2)),idx_len, batch_size=400,
                           epochs=epochs*10, verbose=verbose)
        
    def predict(self,s1,s2,epochs=1,verbose=0):
        #сразу адаптируем под вектора
        pred = self.model.predict(np.hstack((s1,s2)))
        return pred

#Cборщик графов. Он берёт n рандомных состояний и рассчитывает расстояние (направленное!) между каждыми двумя. Есть опция “добавить вершину” (с расчётом всех рёбер) и “удалить вершину”. А так же пересчитать рёбра. Текущий state добавляется, целевые state должны быть добавлены изначально.
class StatesGraph:
    def __init__(self,state_size):
        self.graph = nx.DiGraph()
        self.dist_meas = DistanceMeasure(state_size)
        self.goal_states = []
        self.current_goal = None
    def fit_dist_measure(self,s,done,epochs=1,verbose=0):
        self.dist_meas.fit(s,done,epochs,verbose)
    def add_node_list_safety(self,state_list,max_points_abs=500, max_points_part=0.2,filtration_min_rel=0.05,filtration_min_abs=5):
        #зарядить сюда все вершины. Единственно безопасный вариант!
        #max_points - сколько максимум вершин добавляем из выборки (абсолютное и относительное значение)
        l = len(state_list)
        idx = random.sample(range(l), min(int(l*max_points_part),max_points_abs))
        state_list_sparce = [state_list[i] for i in idx]
        self.add_node_list(state_list,filtration_min_rel=filtration_min_rel,filtration_min_abs=filtration_min_abs)
        
    def add_node_list(self,state_list,filtration_min_rel=0.05,filtration_min_abs=5):
        #Опасно! Может повесить комп!
        #filtration_min - минимальное расстояние, относительное и абсолютное
        if filtration_min_rel is None:
            filtration_min_rel = 0.05
        #ноды должны быть hashable. А значит - ну пусть string
        old_nodes = list(self.graph.nodes)
        #затем пробежаться по всем новым state и всем старым и создать пары старый-новый
        edges_coord = list(itertools.product(old_nodes, state_list)) + list(itertools.product(state_list, old_nodes)) + list(itertools.product(state_list, state_list))
        edges_coord_np = np.array(edges_coord)
        #print('edges_coord_np ',edges_coord_np.shape )#,edges_coord_np
        edges_values = self.dist_meas.predict(edges_coord_np[:,0,0,:],edges_coord_np[:,1,0,:])
        idx = (edges_values>filtration_min_abs) & (edges_values>np.percentile(edges_values,filtration_min_rel*100))
        idx = idx.ravel()
        idx = np.where(idx)[0]
        for i in idx:
            self.graph.add_edge(str(edges_coord[i][0]),str(edges_coord[i][1]),length=edges_values[i])
    def rewrite_edges(self,filtration_min_rel=0.05,filtration_min_abs=5):
        #обновить веса рёбер и дропнуть короткие рёбра
        #filtration_min - минимальное расстояние, относительное и абсолютное
        nodes = list(self.graph.nodes)
        edges_coord = itertools.product(nodes, nodes)
        edges_coord_np = np.array(edges_coord)
        #print('edges_coord_np ',edges_coord_np.shape )#,edges_coord_np
        edges_values = self.dist_meas.predict(edges_coord_np[:,0,0,:],edges_coord_np[:,1,0,:])
        
        idx = np.where(edges_values>filtration_min_abs) & (edges_values>np.percentile(edges_values,filtration_min_rel*100))
        #очистить граф
        self.graph = nx.DiGraph()
        for i in idx:
            self.graph.add_edge(str(edges_coord[i][0]),str(edges_coord[i][1]),length=edges_values[i])
    
    def add_goal_states(self,state_list,filtration_min_rel=0.05,filtration_min_abs=5):
        #добавить целевые состояния
        self.goal_states.append(state_list)
        self.add_node_list(self,state_list,filtration_min_rel=0.05,filtration_min_abs=5)
    
    def add_cur_state(self,state):
        self.cur_state = str(state)
        
    def make_route(self,s1,s2):
        s1 = str(s1)
        s2 = str(s2)
        if not (s1 in self.graph.nodes):
            self.add_node_list([s1],filtration_min_rel=0.05,filtration_min_abs=5)
        if not (s2 in self.graph.nodes):
            self.add_node_list([s2],filtration_min_rel=0.05,filtration_min_abs=5)
            
        path = nx.algorithms.shortest_path.generic.shortest_path(self.graph,s1,s2,weight='length')
        #есть ещё nx.algorithms.all_shortest_paths - оно будет нужно, если мы захотим как-то фильтровать маршруты
        return path
    
    def make_route_to_all_states(self,s1,s2_list):
        #посчитать пути ко всем целям
        s1 = str(s1)
        if not (s1 in self.graph.nodes):
            self.add_node_list([s1],filtration_min_rel=0.05,filtration_min_abs=5)
        path_list = []
        path_length_list = []
        for s2 in s2_list:
            s2 = str(s2)
            if not (s2 in self.graph.nodes):
                self.add_node_list([s2],filtration_min_rel=0.05,filtration_min_abs=5)
            path = nx.algorithms.shortest_path.generic.shortest_path(self.graph,s1,s2,weight='length')
            
            #здесь можно провести какую-нибудь хитрую проверку 
            #на предмет "а не проходит ли граф через какое-то совсем плохое состояние"
            #
            #Дальше рассчитываем длины маршрутов
            path_length = nx.algorithms.shortest_path.generic.shortest_path_length(self.graph,s1,s2,weight='length')
            path_list.append(path)
            path_length_list.append(path_length)
        argmin = np.argmin(np.array(path_length_list)) #кратчайший маршрут
        path = path_list[argmin]
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
        #быть определена, есть current goal не определена, и должна меняться не каждый ход, а раз в несколько ходов
        self.write_goal = None 
        
        #turn - это про то, как добавлять точки каждый ход. Вероятность и число
        self.p_add_points_turn=0.01
        self.max_points_abs_turn=5
        self.max_points_part_turn=0.01
        #episode - это про то, как добавлять точки каждый ход. Вероятность и число
        self.p_add_points_episode=1
        self.max_points_abs_episode=20
        self.max_points_part_episode=0.2
        #минимальная длина ребра
        self.min_edge=2
        #прунинг графа. Грохаем самые короткие рёбра
        self.quantile_edge_min = 0.02
        #окно измерения скорости сближения
        self.window_speed_measure = 8
        #если скорость сближения меньше, чем эта, то меняем цель
        #0.02 - значит, тактов за 50 идеальное совпадение получим с нуля, но я бы на это вообще не рассчитывал
        #скорее, это детектор того, что мы уже не чётко сближаемся, а в шумы воткнулись
        self.min_window_cos_speed = 0.02
        #минимальное время попыток достигать этой локальной цели
        self.min_try_time = 5
        
        super().__init__(state_size)
        self.rl = GoalAgent(state_size, action_size, layers_size)
        
        #целевые состояния. Куда наводиться.
        self.goal_state_list=goal_state_list
        
        #реворды. У нас будет альтернативный способ поиска целевых состояний.
        deque_len = 10000
        self.reward = deque(maxlen=deque_len)
        self.cos_distance = deque(maxlen=deque_len) #насколько хорошо мы приближаемся к цели
        self.goal_changed = deque(maxlen=deque_len) #сменилась ли локальная цель
        
        #среднее и std по состояниям. Нужно, чтобы рассчитывать нормированное косинусное расстояние
        self.mean_std_s = None
        #иногда надо обновлять
        self.frequency_mean_std_update = 0.02
        
        self.auto_aiming_reward_quantile = 0.8 #1 - значит, мы обрубаем эту фичу
        
        #частота, с которой мы проводим ревизию верхнеуровневых goal
        self.drop_goals_by_reward = 0.01
        
        
    def get_action(self, state, verbose=0):
        #Использовать rl
        goal_changed = False
        self.write_goal = self.current_goal
        if len(self.rl.s)<self.rl.train_start:
            #задать рандомную цель, чтобы что-то логировалось, и хоть как-то измерялись расстояния
            if (self.write_goal is None) or (np.random.rand()<0.07):
                idx = int(np.random.rand()*(len(self.rl.s)-1))
                if idx>0:
                    self.write_goal=self.rl.s[idx]
                else:
                    self.write_goal=state
            return random.randrange(self.action_size)
        else:
            if (self.mean_std_s is None) or (np.random.rand()<self.frequency_mean_std_update):
                #обновить цифры для нормализации
                s_arr = np.array(self.rl.s)[:,0,:]
                mean_values = np.mean(s_arr,axis=1)
                std_values = np.std(s_arr,axis=1)
                std_values[std_values==0] = 1
                self.mean_std_s = [mean_values,std_values]
                
            #назначить текущую goal
            #если цели ещё нет
            #или если скорость приближения мала, но не слишком сразу после назначения
            end_id_distance_lst = np.where(self.goal_changed)[0]
            if len(end_id_distance_lst)!=0:
                end_id_distance = end_id_distance_lst[-1]
            else:
                end_id_distance = 0
            #сколько мы достигали этой цели
            try_time_fact = len(self.goal_changed) - end_id_distance 
            len_window_distance = min(try_time_fact,self.window_speed_measure)
            speed_closing_cos_dist = (self.cos_distance[-1]+self.cos_distance[-2]) - (self.cos_distance[-len_window_distance]+self.cos_distance[-len_window_distance+1])
            
            #speed_closing_cos_dist - если положительное, то мы сближаемся, если отрицательное, то отдаляемся
            if (self.current_goal is None) or ((speed_closing_cos_dist<self.quantile_edge_min) and (try_time_fact>self.min_try_time)):
                route = self.make_route_to_all_states(self,state,self.goal_state_list)
                
                lst = re.findall('\d+\.\d+',route[1])
                #распарсить из графа
                lst = np.array([float(l) for l in lst])
                print('state:',state,'route:',route,'lst:',lst)
                self.current_goal = lst
                
                goal_changed = True
                
            self.rl.get_action(state,self.current_goal)
            
            cos_dist = pairwise.cosine_similarity((state - self.mean_std_s[0])/self.mean_std_s[1],(self.current_goal - self.mean_std_s[0])/self.mean_std_s[1])
            self.cos_distance.append(cos_dist)
            self.goal_changed.append(goal_changed)
            
    # save sample <s,s_g,a,r,s'> to the replay memory of rl
    #у графа даже своей replay-памяти нет - а нафиг дублирование (можно сделать, если это будет важно)
    def append_sample(self, state, action, reward, next_state, done):
          
        self.rl.append_sample(state, self.write_goal, action, reward, next_state, done)
        self.reward.append(reward)
        
        
    def train_model(self,epochs=1,sub_batch_size=6000,max_points_abs=None,
                    max_points_part=None,
                    filtration_min_rel=None,filtration_min_abs=None,verbose=0):
        if len(self.rl.s) < self.rl.train_start:
            return
        if max_points_part is None:
            max_points_part = self.max_points_part_turn
        if filtration_min_rel is None:
            filtration_min_rel = self.quantile_edge_min
        if filtration_min_abs is None:
            filtration_min_abs = self.min_edge
        if max_points_abs is None:    
            max_points_abs = self.max_points_abs_turn
            
        self.fit_dist_measure(self.rl.s,self.rl.done,epochs=epochs,verbose=0)
        if len(self.goal_state_list)==0:
            #нет целевых состояний
            #добавить reward-ы в качестве целевых точек.
            quantile = np.percentile(np.array(self.reward),self.auto_aiming_reward_quantile*100)
            idx = np.array(self.reward)>quantile
            idx_num = np.where(idx)[0]
            #это те точки, что надо добавить
            state_list = [self.rl.s[i] for i in idx_num]
            self.add_node_list_safety(state_list,self.max_points_abs_episode, 1,filtration_min_rel,filtration_min_abs)
            self.goal_state_list.extend(state_list)
        if np.random.rand()<self.drop_goals_by_reward:
            #пересмотр верхнеуровневых целей. Отбросить те, где реворд мелкий.
            #реворд - последний столбец в state
            quantile = np.percentile(np.array(self.reward),self.auto_aiming_reward_quantile*100)
            goal_state_list_new = []
            for goal in self.goal_state_list:
                if goal[-1]>=quantile:
                    goal_state_list_new.append(goal)
            self.goal_state_list = goal_state_list_new

        #добавить новые точки в граф
        if np.random.rand()<self.p_add_points_turn:
            self.add_node_list_safety(self.rl.s,max_points_abs, max_points_part,filtration_min_rel,filtration_min_abs)
        #обучить rl
        self.rl.train_model(epochs,sub_batch_size,verbose)
        
    def update_target_model(self):
        if len(self.rl.s) < self.rl.train_start:
            return
        #добавить reward-ы в качестве целевых точек.
        quantile = np.percentile(np.array(self.reward),self.auto_aiming_reward_quantile*100)
        idx = np.array(self.reward)>quantile
        idx_num = np.where(idx)[0]
        #это те точки, что надо добавить
        state_list = [self.rl.s[i] for i in idx_num]
        self.add_node_list_safety(state_list,self.max_points_abs_episode, 1,self.quantile_edge_min,self.min_edge)
        self.goal_state_list.extend(state_list)
        #
        self.train_model(epochs=30,sub_batch_size=6000,verbose=0,
                         max_points_abs=self.max_points_abs_episode,max_points_part=self.max_points_part_episode)
        self.train_model(epochs=1,sub_batch_size=6000,verbose=1,
                         max_points_abs=self.max_points_abs_episode,max_points_part=self.max_points_part_episode)