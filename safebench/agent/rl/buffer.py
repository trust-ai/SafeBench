'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-06-19 11:45:14
@LastEditTime: 2020-07-13 20:41:11
@Description: 
'''
import numpy as np


class ReplayBuffer():
    ''' This buffer stores both safe and risk data separetrly. '''
    def __init__(self, memory_capacity, buffer_dim, risk_aware):
        self.risk_aware = risk_aware
        self.buffer_dim = buffer_dim
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((self.memory_capacity, self.buffer_dim))
        self.risk_memory = np.zeros((self.memory_capacity, self.buffer_dim))
        self.risk_memory_counter = 0
        self.memory_counter = 0
        # since we initiate an empty buffer, we should remember the length
        self.risk_memory_len = 0
        self.memory_len = 0

    def push(self, data, risk):
        # only when the risk_aware is True we separate the buffer
        if risk and self.risk_aware:
            index = self.risk_memory_counter % self.memory_capacity
            self.risk_memory[index, :] = data
            self.risk_memory_counter += 1
            self.risk_memory_len = min(self.risk_memory_len+1, self.memory_capacity)
        else:
            index = self.memory_counter % self.memory_capacity
            self.memory[index, :] = data
            self.memory_counter += 1
            self.memory_len = min(self.memory_len+1, self.memory_capacity)

    def sample(self, batch_size):
        sample_index = np.random.randint(0, self.risk_memory_len+self.memory_len, size=batch_size)
        self.total_memory = np.concatenate([self.memory[0:self.memory_len], self.risk_memory[0:self.risk_memory_len]], axis=0)
        batch_memory = self.total_memory[sample_index, :]
        return batch_memory
