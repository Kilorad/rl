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
sys.path.append('./common/')
import utils


class randomAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.action_size = action_size

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        return random.randrange(self.action_size)