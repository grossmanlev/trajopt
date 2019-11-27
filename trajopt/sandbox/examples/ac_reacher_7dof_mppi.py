from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
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
    parser.add_argument('--critic', default=None,
                        help='path to critic model (.pt) file')
    parser.add_argument('--train', default=False,
                        help='train the critic net?')
    parser.add_argument('--save_buffer', default=False,
                        help='save the replay buffer?')
    args = parser.parse_args()

    e = get_environment(ENV_NAME)
    e.reset_model(seed=SEED)
    mean = np.zeros(e.action_dim)
    sigma = 1.0*np.ones(e.action_dim)
    filter_coefs = [sigma, 0.25, 0.8, 0.0]

    agent = MPPI(e, H=16, paths_per_cpu=40, num_cpu=1,
                 kappa=25.0, gamma=1.0, mean=mean, filter_coefs=filter_coefs,
                 default_act='mean', seed=SEED)

    replay_buffer = ReplayBuffer(max_size=1000000)

    critic = Critic(num_iters=3000)
    loaded_critic = None
    if args.critic is not None:
        loaded_critic = Critic()
        loaded_critic.load_state_dict(torch.load(args.critic))
        loaded_critic.eval()
        loaded_critic.float()
    critic.float()

    optimizer = optim.Adam(critic.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    ts = timer.time()
    for t in tqdm(range(H_total)):

        # Actor step
        tuples = agent.train_step(critic=loaded_critic, niter=N_ITER)

        # Add new transitions into replay buffer
        tuples = critic.compress_states(tuples)
        replay_buffer.concatenate(tuples)

        if args.train:
            # Critic step
            for i in range(critic.num_iters):
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

                # import pdb; pdb.set_trace()
                # returns a new Tensor, detached from the current graph, the result will never require gradient
                y_batch = y_batch.detach()

                # calculate loss
                loss = criterion(v_batch, y_batch)

                # do backward pass
                loss.backward()
                optimizer.step()

                if i % 1000 == 0:
                    print('Loss: {}'.format(loss))
                    s0 = torch.zeros(STATE_DIM)
                    sf = torch.tensor([10.99164932,  0.06841799, -1.50792112, -1.56400837, -1.52414601,
                                       0.01832143, -1.52851301, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    print('{}'.format(critic(s0)))
                    print('{}'.format(critic(sf)))

        if t % 25 == 0 and t > 0:
            print("==============>>>>>>>>>>> saving progress ")
            pickle.dump(agent, open(PICKLE_FILE, 'wb'))
            torch.save(critic.state_dict(), MODEL_PATH)

    # Save the replay buffer
    if args.save_buffer:
        print("==============>>>>>>>>>>> saving replay buffer")
        pickle.dump(replay_buffer, open(REPLAY_PICKLE_FILE, 'wb'))


    pickle.dump(agent, open(PICKLE_FILE, 'wb'))
    # agent = pickle.load(open(PICKLE_FILE, 'rb'))
    print("Trajectory reward = %f" % np.sum(agent.sol_reward))
    print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))

    # wait for user prompt before visualizing optimized trajectories
    _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
    for _ in range(100):
        agent.animate_result()
