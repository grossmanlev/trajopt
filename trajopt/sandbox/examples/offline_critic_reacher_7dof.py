from trajopt.utils import ReplayBuffer, Tuple
from tqdm import tqdm
import time as timer
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from trajopt.models.critic_nets import Critic

import argparse
import os
from datetime import datetime

# =======================================
DATA_DIR = 'data'
now = datetime.now()
DATE_STRING = now.strftime("%m:%d:%Y-%H:%M:%S")
os.mkdir(DATA_DIR + '/' + DATE_STRING)
ENV_NAME = 'reacher_7dof'
PICKLE_FILE = DATA_DIR + '/' + DATE_STRING + '/' + ENV_NAME + '_mppi.pickle'
MODEL_PATH = DATA_DIR + '/' + DATE_STRING + '/' + ENV_NAME + '_critic.pt'
REPLAY_PICKLE_FILE = DATA_DIR + '/' + DATE_STRING + '/' + ENV_NAME + '_replay.pickle'
SEED = 12345
N_ITER = 5
H_total = 100
STATE_DIM = 14
# =======================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_buffer', default=None,
                        help='path to ReplayBuffer (.pickle) file')
    parser.add_argument('--critic', type=bool, default=None,
                        help='path to critic model (.pt) file')
    parser.add_argument('--num_iters', type=int, default=10000,
                        help='number of iterations for critic training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for critic net')
    args = parser.parse_args()

    replay_buffer = pickle.load(open(args.replay_buffer, 'rb'))

    critic = Critic(num_iters=args.num_iters, batch_size=args.batch_size)
    loaded_critic = None
    if args.critic is not None:
        loaded_critic = Critic(num_iters=args.num_iters, batch_size=args.batch_size)
        loaded_critic.load_state_dict(torch.load(args.critic))
        loaded_critic.eval()
        loaded_critic.float()
    critic.float()

    optimizer = optim.Adam(critic.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    ts = timer.time()
    for t in tqdm(range(critic.num_iters)):

        # Critic step
        minibatch = replay_buffer.get_minibatch(size=critic.batch_size)

        # unpack minibatch
        state_batch = torch.stack(tuple(torch.tensor(d.state, dtype=torch.float32) for d in minibatch))
        action_batch = torch.stack(tuple(torch.tensor(d.action, dtype=torch.float32) for d in minibatch))
        reward_batch = torch.tensor([d.reward for d in minibatch], dtype=torch.float32)
        # reward_batch = torch.cat(tuple(torch.tensor(d.reward) for d in minibatch))
        next_state_batch = torch.stack(tuple(torch.tensor(d.next_state, dtype=torch.float32) for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # get output for the next state
        output_batch = critic(next_state_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(V)
        y_batch = []
        for j in range(len(minibatch)):
            if abs(reward_batch[j]) < 2.0 or True:
                y_batch.append(reward_batch[j] + 0.0 * output_batch[j])
            else:
                y_batch.append(reward_batch[j] + critic.gamma * output_batch[j])
        y_batch = torch.stack(tuple(y_batch))
        # y_batch = torch.stack(
        #     tuple(reward_batch[i] if abs(reward_batch[i]) < 1.0
        #           else reward_batch[i] + critic.gamma * output_batch[i]
        #           for i in range(len(minibatch))))

        # extract V
        # v_batch = torch.sum(critic(state_batch), dim=1)  # ??
        v_batch = critic(state_batch)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(v_batch, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        if t % 1000 == 0:
            print('Loss: {}'.format(loss))
            s0 = torch.zeros(STATE_DIM)
            sf = torch.tensor([10.99164932,  0.06841799, -1.50792112, -1.56400837, -1.52414601,
                               0.01832143, -1.52851301, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            print('{}'.format(critic(s0)))
            print('{}'.format(critic(sf)))

        # if t % (critic.num_iters // 4) == 0:
        #     print("==============>>>>>>>>>>> saving progress ")
        #     torch.save(critic.state_dict(), MODEL_PATH)

    print("==============>>>>>>>>>>> saving progress ")
    torch.save(critic.state_dict(), MODEL_PATH)
