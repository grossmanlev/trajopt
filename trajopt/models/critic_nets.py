import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trajopt.utils import Tuple

STATE_DIM = 14


class Critic(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, gamma=1.0, num_iters=10000, batch_size=128,
                 input_dim=14, inner_layer=128):
        super(Critic, self).__init__()

        self.gamma = gamma
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.input_dim = input_dim

        self.tanh = torch.tanh
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 1)
        # )

        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.BatchNorm1d(128),
        #     nn.Tanh(),
        #     nn.Linear(128, 128),
        #     nn.BatchNorm1d(128),
        #     nn.Tanh(),
        #     nn.Linear(128, 1)
        # )

        self.linear1 = nn.Linear(input_dim, inner_layer)

        self.linear2 = nn.Linear(inner_layer, inner_layer)
        # self.linear2_5 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(inner_layer, 1)

    def forward(self, x):
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        # x = self.relu(self.linear2_5(x))

        # critic: evaluates being in the state s_t
        value = self.linear3(x)
        return value

        # return self.model(x)

    def compress_state(self, state, dim=14):
        """ Put a reacher_env observation into a compressed format """
        if dim == 14:
            return state[:14]
        elif dim == 17:
            return np.concatenate((state[:14], state[-3:]))
        elif dim == 3:
            return state[-6:-3]
        elif dim == 6:
            return state[-6:]
            # return np.concatenate(
            #     (state["qp"],
            #      state["qv"],
            #      state["target_pos"]))

    def compress_states(self, tuples, dim=14):
        """ Put states into array representation, so everything is an np.array """
        for i in range(len(tuples)):
            state = self.compress_state(
                tuples[i].state, dim=dim)
            next_state = self.compress_state(
                tuples[i].next_state, dim=dim)

            tuples[i] = Tuple(
                state,
                tuples[i].action,
                tuples[i].reward,
                next_state)

        return tuples

    def compress_agent(self, agent, dim=14):
        """ Put agent solution trajectory into Tuple format """
        tuples = []
        for i in range(len(agent.sol_obs) - 1):
            state = self.compress_state(
                agent.sol_obs[i], dim=dim)
            next_state = self.compress_state(
                agent.sol_obs[i+1], dim=dim)

            tuples.append(Tuple(
                state,
                agent.sol_act[i],
                agent.sol_reward[i],
                next_state))

        return tuples
