'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-04-16 21:29:40
LastEditTime: 2020-09-14 22:27:54
@Description: 
'''

import numpy as np
from .optimizers import RandomOptimizer, CEMOptimizer
import copy
import sys


class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, mpc_args):
        self.type = mpc_args['type']
        self.horizon = mpc_args["horizon"]
        self.gamma = mpc_args["gamma"]
        self.action_low = np.array(mpc_args["action_low"]) # array (dim,)
        self.action_high = np.array(mpc_args["action_high"]) # array (dim,)
        self.action_dim = mpc_args["action_dim"]
        self.popsize = mpc_args["popsize"]
        self.particle = mpc_args["particle"]
        self.early_stop = mpc_args["early_stop"]

        self.risk_cost = mpc_args["risk_cost"]
        self.boundary_cost = mpc_args["boundary_cost"]

        self.init_mean = np.array([mpc_args["init_mean"]] * self.horizon)
        self.init_var = np.array([mpc_args["init_var"]] * self.horizon)

        self.action_low = np.tile(self.action_low, [self.action_dim])
        self.action_high = np.tile(self.action_high, [self.action_dim])

        self.optimizer = MPC.optimizers[self.type](
            sol_dim=self.horizon*self.action_dim,
            popsize=self.popsize,
            upper_bound=np.array(mpc_args["action_high"]),
            lower_bound=np.array(mpc_args["action_low"]),
            max_iters=mpc_args["max_iters"],
            num_elites=mpc_args["num_elites"],
            epsilon=mpc_args["epsilon"],
            alpha=mpc_args["alpha"]
        )

        self.optimizer.setup(self.cost_function)
        self.reset()

    def reset(self):
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])

    def act(self, model, state, ego_goal):
        self.model = model
        self.state = state
        self.ego_goal = np.array(ego_goal)

        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        action = soln[:self.action_dim]
        return action

    def preprocess(self, batch_size=None):
        state = np.repeat(self.state[None], self.popsize*self.particle, axis=0)
        return state

    def cost_function(self, actions):
        # the observation need to be processed since we use a common model
        state = self.preprocess()

        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))
        costs = np.zeros(self.popsize*self.particle)

        # for recording the risk flag
        self.risk_mask = np.zeros((self.popsize*self.particle))

        # for calculating the goal distance incremental
        ego_pos = state[:, 0:2]
        self.previous_goal_distance = np.sum((self.ego_goal[None] - ego_pos)**2, axis=1)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size, timestep, action dim)
            # the output of the prediction model is [state_next - state]
            state_next = self.model.predict(state, action) + state
            cost = self.intersection_cost(state_next)  # compute cost
            costs += cost*self.gamma**t
            state = copy.deepcopy(state_next)
        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def intersection_cost(self, state):
        ''' distance to the goal '''
        ego_pos = state[:, 0:2]
        ego_goal_distance = np.sum((self.ego_goal[None] - ego_pos)**2, axis=1)
        delta_distance = ego_goal_distance - self.previous_goal_distance # positive difference means being far away from goal
        final_cost = delta_distance    
        self.previous_goal_distance = ego_goal_distance # update previous distance

        ''' road constraint, MAKE SURE THE PENALTY IS LARGER THAN LONG TERM GOAL REWARD '''
        road_blocks = [
            [-5, -12, np.pi/2, 15, 0.5], # check left
            [4, 0, np.pi/2, 40, 0.5],    # check right
            [-16, -4.5, np.pi, 22, 0.5], # check up
            [-16, 5, np.pi, 22, 0.5],    # check down
            [-5, 13, np.pi/2, 16, 0.5],  # check left
        ]
        # These three regions are not drivable
        # OR
        left_top_region = (ego_pos[:, 0] < road_blocks[0][0]) * (ego_pos[:, 1] < road_blocks[2][1])
        left_bottom_region = (ego_pos[:, 0] < road_blocks[0][0]) * (ego_pos[:, 1] > road_blocks[3][1])
        right_region = ego_pos[:, 0] > road_blocks[1][0]
        # AND
        region_validation = left_top_region + left_bottom_region + right_region
        final_cost[region_validation] += self.boundary_cost

        ''' when two vehicles are too close, MAKE SURE THE PENALTY IS LARGER THAN LONG TERM GOAL REWARD '''
        other_pos = state[:, 4:6]
        ego_other_distance = np.sum((other_pos - ego_pos)**2, axis=1)
        # even using early stop, we still give the penalty once 
        final_cost[ego_other_distance < 3.0] += self.risk_cost 
        if self.early_stop:
            # for the crashed data, set all following cost to 0
            final_cost = final_cost*(1.0-self.risk_mask)
            # set the flag to true once collision happens
            self.risk_mask[ego_other_distance < 3.0] = 1.0

        return final_cost
