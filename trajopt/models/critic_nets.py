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
    def __init__(self, gamma=1.0, num_iters=10000, batch_size=128):
        super(Critic, self).__init__()

        self.gamma = gamma
        self.num_iters = num_iters
        self.batch_size = batch_size

        self.tanh = torch.tanh
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(STATE_DIM, 128)  # TODO: 7DOF qpos and qvel

        self.linear2 = nn.Linear(128, 128)
        # self.linear2_5 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        # x = self.relu(self.linear2_5(x))

        # critic: evaluates being in the state s_t
        value = self.linear3(x)

        return value

    def compress_state(self, state):
        """ Put a reacher_env state into a compressed format """
        return np.concatenate((state["qp"], state["qv"]))

    def compress_states(self, tuples):
        """ Put states into array representation, so everything is an np.array """
        for i in range(len(tuples)):
            state = self.compress_state(tuples[i].state)
            np.concatenate(
                (tuples[i].state["qp"], tuples[i].state["qv"]))
            next_state = np.concatenate(
                (tuples[i].next_state["qp"], tuples[i].next_state["qv"]))
            tuples[i] = Tuple(
                state, tuples[i].action, tuples[i].reward, next_state)
        return tuples
