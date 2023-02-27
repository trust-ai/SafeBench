'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-06-09 23:39:36
@LastEditTime: 2020-07-16 15:02:39
@Description: 
'''

import numpy as np
import torch
import time

from .optimizer import Optimizer


class RandomOptimizer(Optimizer):
    def __init__(self, sol_dim, popsize, upper_bound=None, lower_bound=None, max_iters=10, num_elites=100, epsilon=0.001, alpha=0.25):
        super().__init__()
        self.sol_dim = sol_dim
        self.popsize = popsize
        self.ub, self.lb = torch.FloatTensor(upper_bound), torch.FloatTensor(lower_bound)
        self.solution = None
        self.cost_function = None

    def setup(self, cost_function):
        #print("lb, ub", self.lb, self.ub)
        self.cost_function = cost_function
        self.sampler = torch.distributions.uniform.Uniform(self.lb, self.ub)
        self.size = [self.popsize, self.sol_dim]

    def reset(self):
        pass

    def obtain_solution(self, *args, **kwargs):        
        solutions = self.sampler.sample(self.size).cpu().numpy()[:,:,0]
        #solutions = np.random.uniform(self.lb, self.ub, [self.popsize, self.sol_dim])
        costs = self.cost_function(solutions)
        return solutions[np.argmin(costs)], None
