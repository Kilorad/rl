import sys
sys.path.insert(1, './env')
sys.path.insert(1, './agents')

import gym
import pylab
import random
import strategy_imitation, sarsa, ddqn, random_agent, a2c, model_based,graph_ai
import aa_gun,jet_table_simple
import numpy as np
from collections import deque
import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd



#Проверь на зенитке, на cartpole и на mountain car
EPISODES=1400

print('_____',pd.Timestamp.now())
#здесь весь код от инициализации модели до выдачи scores.
# In case of CartPole-v1, maximum length of episode is 500

env = jet_table_simple.jet_table_simple_env()
#env = gym.make('Seaquest-ramNoFrameskip-v0')
#env=CartPoleEnv9()
# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

#agent = sarsa.SarsaAgent(state_size, action_size)
agent = graph_ai.GraphAI(state_size, action_size)
agent.rl.train_start=340
#agent.train_start=7000
#agent.epsilon_decay=0.9999
agent.render=True

scores, episodes = [], []
reward_lst = []
s_list=[]
a_list=[]
agent.epsilon = 1

for e in range(EPISODES):
    done = False
    score = 0
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    

    while not done:
        if (e in range(0,18)) or (e in range(20,22)) or (e in range(30,32)) or (e in range(40,42)) or (e in range(50,52)) or (e in range(100,103)) or (e in range(200,202)) or (e in range(300,306)) or (e in range(400,406)) or (e in range(500,506)) or (e in range(600,604)):
            if agent.render:
                env.render()

        # get action for the current state and go one step in environment)
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        # if an action make the episode end, then gives penalty of -100


        # save the sample <s, a, r, s'> to the replay memory
        agent.append_sample(state, action, reward, next_state, done)
        #if next_state[0,11]!=reward:
        #    print('state[13]!=reward',state[0,11],reward)
        #
        s_list.append(state)
        a_list.append(action)
        reward_lst.append(reward)
        #

        # every time step do the training
        agent.train_model()
        score += reward
        state = next_state

        if done:
            # every episode update the target model to be same with model
            agent.update_target_model()

            # every episode, plot the play time
            scores.append(score)
            episodes.append(e)
            pylab.plot(episodes, scores, 'b')
            #pylab.savefig("./save_graph/aa_gun_dqn.png")
            try:
                print("episode:", e, "  score:", score,np.mean(scores), "  memory length:",
                      len(agent.rl.s), "  epsilon:", agent.epsilon)
            except Exception:
                print("episode:", e, "  score:", score,np.mean(scores), "  memory length:",
                      len(agent.rl.s), "  epsilon:", agent.epsilon)
