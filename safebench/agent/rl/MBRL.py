'''
@Author: 
@Email: 
@Date:   2020-03-24 01:01:42
@LastEditTime: 2020-07-16 17:05:31
@Description:
'''

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import sys

import torch
import torch.nn as nn
from loguru import logger
from mpc.mpc import MPC

sys.path.append('../')
from utils import CUDA, CPU, kaiming_init
np.set_printoptions(precision=5)


class MLP(nn.Module):
    def __init__(self, n_input=7, n_output=6, n_h=2, size_h=128):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc_list = nn.ModuleList()
        for i in range(n_h - 1):
            self.fc_list.append(nn.Linear(size_h, size_h))
        self.fc_out = nn.Linear(size_h, n_output)

    def forward(self, x):
        out = x.view(-1, self.n_input)
        out = self.fc_in(out)
        out = self.relu(out)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.relu(out)
        out = self.fc_out(out)
        out = self.tanh(out)
        return out


class NN(object):
    def __init__(self, args):
        self.state_dim = args["state_dim"]
        self.action_dim = args["action_dim"]
        self.input_dim = self.state_dim+self.action_dim

        self.n_epochs = args["n_epochs"]
        self.lr = args["lr"]
        self.batch_size = args["batch_size"]
        
        self.validation_flag = args["validation_flag"]
        self.validate_freq = args["validation_freq"]
        self.validation_ratio = args["validation_ratio"]

        self.model = CUDA(MLP(self.input_dim, self.state_dim, args["hidden_dim"], args["hidden_size"]))
        self.model.apply(kaiming_init)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.data = None
        self.label = None
        self.mu = CUDA(torch.tensor(0.0))
        self.sigma = CUDA(torch.tensor(args['scale']))
        self.label_mu = CUDA(torch.tensor(0.0))
        self.label_sigma = CUDA(torch.tensor(args['scale']))
        self.eps = 1e-30

    def data_process(self, data):
        s = data[0][None]
        a = data[1][None]
        label = data[2][None] # here label means the next state
        data = np.concatenate((s, a), axis=1)

        # add new data point to data buffer
        if self.data is None:
            self.data = CUDA(torch.Tensor(data))
            self.label = CUDA(torch.Tensor(label))
        else:
            self.data = torch.cat((self.data, CUDA(torch.tensor(data).float())), dim=0)
            self.label = torch.cat((self.label, CUDA(torch.tensor(label).float())), dim=0)

    def split_train_validation(self):
        num_data = len(self.data)

        # normalization, note that we should not overrite the original data and label
        self.train_data = (self.data-self.mu) / self.sigma
        self.train_label = (self.label-self.label_mu) / self.label_sigma

        # use validation
        if self.validation_flag:
            indices = list(range(num_data))
            split = int(np.floor(self.validation_ratio * num_data))
            np.random.shuffle(indices)
            train_idx, test_idx = indices[split:], indices[:split]

            train_set = [[self.train_data[idx], self.train_label[idx]] for idx in train_idx]
            test_set = [[self.train_data[idx], self.train_label[idx]] for idx in test_idx]

            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        else:
            train_set = [[self.train_data[idx], self.train_label[idx]] for idx in range(num_data)]
            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = None
        return train_loader, test_loader

    def fit(self):
        train_loader, test_loader = self.split_train_validation()
        self.model.apply(kaiming_init)
        best_test_loss = np.inf

        for epoch in range(self.n_epochs):
            loss_this_epoch = []
            for datas, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_this_epoch.append(loss.item())

            if self.validation_flag and (epoch+1) % self.validate_freq == 0:
                with torch.no_grad():
                    loss_test = self.validate_model(test_loader)
                if best_test_loss >= loss_test:
                    best_test_loss = loss_test
                    best_model = self.model.state_dict()
                #logger.info(f"Epoch [{epoch}/{self.n_epochs}], loss train: {np.mean(loss_this_epoch):.4f}, loss test  {loss_test:.4f}")

        # return the best model if we use validation
        if self.validation_flag:
            self.model.load_state_dict(best_model)

        return np.mean(loss_this_epoch)

    def validate_model(self, testloader):
        loss_list = []
        for datas, labels in testloader:
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss_list.append(loss.item())
        return np.mean(loss_list)

    def predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        inputs = (inputs-self.mu) / (self.sigma + self.eps)
        with torch.no_grad():
            ds = self.model(inputs)
            ds = ds * (self.label_sigma + self.eps) + self.label_mu
            ds = ds.cpu().detach().numpy()
        return ds

    def test(self, s, a, x_g):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        inputs = (inputs-self.mu) / (self.sigma + self.eps)
        with torch.no_grad():
            ds = self.model(inputs)
            ds_unnormal = ds * (self.label_sigma + self.eps) + self.label_mu
            ds_unnormal = ds_unnormal.cpu().detach().numpy()
            ds = ds.cpu().detach().numpy()
        
        x_g = CUDA(torch.Tensor(x_g))
        x_g = (x_g-self.label_mu) / (self.label_sigma + self.eps)
        x_g = x_g.cpu().numpy()
        mse_error = np.sum((ds-x_g)**2)

        return ds_unnormal, mse_error


class MBRL():
    name = 'MBRL'

    def __init__(self, args):
        self.pretrain_buffer_size = args['pretrain_buffer_size']
        self.exploration_noise = args['exploration_noise']
        self.action_dim = args['action_dim']
        self.model_path = args['model_path']
        self.model_id = args['model_id']
        self.ego_goal = np.array([2, -30])

        self.mpc_controller = MPC(args['mpc'])
        self.mpc_controller.reset()
        self.model = NN(args)

    def select_action(self, state, deterministic):
        action = self.mpc_controller.act(model=self.model, state=state, ego_goal=self.ego_goal)
        # add some noise for exploration
        if not deterministic:
            action += np.random.normal(0, self.exploration_noise, size=self.action_dim)
        return action

    def store_transition(self, data):
        # [_state, _action, _state_next - _state, ego_goal]
        self.ego_goal = data[3]
        self.model.data_process(data=data)

    def train(self):
        # when data has been collected enough, train model
        if self.model.data.shape[0] < self.pretrain_buffer_size:
            print('Skip training, buffer size: {}'.format(self.model.data.shape[0]))
            return
        self.model.fit()

    def save_model(self):
        states = {'model': self.model.model.state_dict()}
        filepath = os.path.join(self.model_path, 'model.mbrl.'+str(self.model_id)+'.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.model_path, 'model.mbrl.'+str(self.model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.model.load_state_dict(checkpoint['model'])
        else:
            raise Exception('No MBRL model found!')

    def test(self, s, a, x_g):
        ds_unnormal, mse_error = self.model.test(s[None], a[None], x_g[None])
        #print('predict: {}'.format(ds_unnormal[0]))
        #print('truth  : {}'.format(x_g))
        print('mse: {}'.format(mse_error))
