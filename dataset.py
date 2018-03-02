import gym
import math
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

# stores previous states (which consists of observation + action)
# and next states (which consists of next observation)
ckass Dataset:
    def __init__(self):
        self.prev_states = None
        self.next_states = None
        self.n_data = None
        self.cur_idx = 0


    def set_data(self, prev_states, next_states):
        assert prev_states.shape[0] == next_states.shape[0]
        self.n_data = prev_states.shape[0]
        self.prev_states = prev_states
        self.next_states = next_states
        self.cur_idx %= self.n_data

    def add_data(self, prev_states, next_states):
        assert prev_states.shape[0] == next_states.shape[0]

        if self.prev_states is not None:
            self.prev_states = np.concatenate([self.prev_states, prev_states], axis=0)
            self.next_states = np.concatenate([self.next_states, next_sta
                                               tes], axis=0)
        else:
            self.prev_states = prev_states
            self.next_states = next_states
        self.n_data = self.prev_states.shape[0]


    def get_num_data(self):
        if self.n_data is None:
            return 0
        return self.n_data

    # uniformly randomly sample a batch of data (with replacement)
    def sample(self, batch_size):
        indices = np.floor(self.n_data * np.random.uniform(0.0, 1.0, size=batch_size)).astype(np.intp)
        return self.prev_states[indices, :], self.next_states[indices, :]

    def get_next_batch(self, batch_size, is_shuffled=False):
        assert batch_size <= self.n_data, "Batch size %d is larger than n_data %d"%(batch_size,self.n_data)
        start_idx = self.cur_idx
        end_idx = self.cur_idx + batch_size
        if end_idx > self.n_data:
            remainder = batch_size - (self.n_data - start_idx)
            indices = list(range(start_idx, self.n_data)) + list(range(0, remainder))
            self.cur_idx = remainder
        else:
            indices = list(range(start_idx, end_idx))
            self.cur_idx = end_idx

        assert(len(indices) == end_idx - start_idx)

        return self.prev_states[indices, :], self.next_states[indices, :]

