from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
from trajopt.utils import ReplayBuffer
from tqdm import tqdm
import time as timer
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =======================================
ENV_NAME = 'reacher_7dof'
SEED = 12345
N_ITER = 5
H_total = 100
STATE_DIM = 14
# =======================================


class Critic(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, gamma=1.0, num_iters=10000, batch_size=32):
        super(Critic, self).__init__()

        self.gamma = gamma
        self.num_iters = num_iters
        self.batch_size = batch_size

        self.linear1 = nn.Linear(STATE_DIM, 128)  # TODO: 7DOF qpos and qvel

        # self.linear2 = nn.Linear(128, 32)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))

        # critic: evaluates being in the state s_t
        value = self.linear2(x)

        return value


def compress_states(tuples):
    """ Put states into array representation, so everything is an np.array """
    for i in range(len(tuples)):
        state = np.concatenate(
            (tuples[i].state["qp"], tuples[i].state["qv"]))
        next_state = np.concatenate(
            (tuples[i].next_state["qp"], tuples[i].next_state["qv"]))
        tuples[i] = ReplayBuffer.Tuple(
            state, tuples[i].action, tuples[i].reward, next_state)
    return tuples


if __name__ == '__main__':
    s0 = torch.zeros(14)
    sf = torch.tensor([10.99164932,  0.06841799, -1.50792112, -1.56400837, -1.52414601,
        0.01832143, -1.52851301, 0.24921592,  0.4719793 ,  0.01687387,  0.38145194, -0.02975449,
       -0.23017395,  0.00742663])
    critic = Critic()
    critic.load_state_dict(torch.load('reacher_7dof_critic.pt'))
    critic.eval()

    out = critic(s0)
    print(out)
