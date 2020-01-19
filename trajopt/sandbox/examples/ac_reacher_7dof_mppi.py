from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
from trajopt.utils import ReplayBuffer, Tuple
from trajopt.models.critic_nets import Critic
from tqdm import tqdm
import time as timer
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


import argparse
import os
from datetime import datetime

# =======================================
DATA_DIR = 'data'
AC_DIR = 'ac_tests'

now = datetime.now()
DATE_STRING = now.strftime("%m:%d:%Y-%H:%M:%S")
os.mkdir(DATA_DIR + '/' + DATE_STRING)
AC_SAVE_DIR = AC_DIR + '/' + DATE_STRING
os.mkdir(AC_SAVE_DIR)

ENV_NAME = 'reacher_7dof'
PICKLE_FILE = DATA_DIR + '/' + DATE_STRING + '/' + ENV_NAME + '_mppi.pickle'
MODEL_PATH = DATA_DIR + '/' + DATE_STRING + '/' + ENV_NAME + '_critic.pt'
REPLAY_PICKLE_FILE = DATA_DIR + '/' + DATE_STRING + '/' + ENV_NAME + '_replay.pickle'
SEED = 12345
N_ITER = 5
H_total = 100
STATE_DIM = 14
H = 16
# =======================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic', default=None,
                        help='path to critic model (.pt) file')
    parser.add_argument('--train', action='store_true',
                        help='train the critic net?')
    parser.add_argument('--save_buffer', action='store_true',
                        help='save the replay buffer?')
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--eta', default=0.9, type=float)
    parser.add_argument('--goals', action='store_true')
    args = parser.parse_args()

    # Check to add goal position to state space
    if args.goals:
        STATE_DIM += 3

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    e = get_environment(ENV_NAME, sparse_reward=True)
    # e.sparse_reward = True
    e.reset_model(seed=SEED)
    mean = np.zeros(e.action_dim)
    sigma = 1.0*np.ones(e.action_dim)
    filter_coefs = [sigma, 0.25, 0.8, 0.0]

    good_agent = pickle.load(open('116_agent.pickle', 'rb'))
    sol_actions = np.array(good_agent.sol_act)  # should be (100, 7)
    init_seq = sol_actions[:H]

    replay_buffer = ReplayBuffer(max_size=1000000)

    critic = Critic(num_iters=2000, input_dim=STATE_DIM)
    loaded_critic = None
    if args.critic is not None:
        loaded_critic = Critic(num_iters=2000, input_dim=STATE_DIM)
        loaded_critic.load_state_dict(torch.load(args.critic))
        loaded_critic.eval()
        loaded_critic.float()
    critic.float()

    # Set up target critic network
    if args.target:
        target_critic = Critic(num_iters=2000, input_dim=STATE_DIM)
        target_critic.load_state_dict(critic.state_dict())
        target_critic.eval()
        target_critic.float()

    optimizer = optim.Adam(critic.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    eta = args.eta
    limit = int((H_total + H) / 0.9)

    env_seeds = [0, 1, 2, 3, 4]

    for x in range(100):
        limit = int(limit * 0.9)
        print('*'*36)
        print('Round: {}, Limit: {}'.format(x, limit))
        print()
        if limit >= H:
            e.reset_model(seed=SEED)
            agent = MPPI(e, H=H, paths_per_cpu=40, num_cpu=1,
                         kappa=25.0, gamma=1.0, mean=mean,
                         filter_coefs=filter_coefs,
                         default_act='mean', seed=SEED,
                         init_seq=init_seq)
        else:
            e.reset_model(seed=SEED)
            agent = MPPI(e, H=H, paths_per_cpu=40, num_cpu=1,
                         kappa=25.0, gamma=1.0, mean=mean,
                         filter_coefs=filter_coefs,
                         default_act='mean', seed=SEED,
                         init_seq=None)

        ts = timer.time()
        for t in tqdm(range(H_total)):

            # Actor step
            # if x == 0:
            #     tuples = agent.train_step(critic=None, niter=N_ITER)
            # else:
            #     tuples = agent.train_step(critic=critic, niter=N_ITER)
            if limit >= t + H:
                tuples = agent.train_step(critic=critic, niter=N_ITER,
                                          act_sequence=sol_actions[t:t+H])
            else:
                tuples = agent.train_step(critic=critic, niter=N_ITER,
                                          act_sequence=None)
            # tuples = agent.train_step(critic=critic, niter=N_ITER)
            # Add new transitions into replay buffer
            tuples = critic.compress_states(tuples, include_goal=True)
            replay_buffer.concatenate(tuples)

            import pdb; pdb.set_trace()

        print("Trajectory reward = %f" % np.sum(agent.sol_reward))
        writer.add_scalar('Trajectory Return', np.sum(agent.sol_reward), x)

        pickle.dump(agent, open(AC_SAVE_DIR + '/ac_test_agent_{}.pickle'.format(x), 'wb'))
        torch.save(critic.state_dict(), AC_SAVE_DIR + '/ac_test_critic_{}.pt'.format(x))

        if args.train:
            # Critic step
            for i in tqdm(range(critic.num_iters)):
                minibatch = replay_buffer.get_minibatch(size=critic.batch_size)

                # unpack minibatch
                state_batch = torch.stack(tuple(torch.tensor(d.state, dtype=torch.float32) for d in minibatch))
                action_batch = torch.stack(tuple(torch.tensor(d.action, dtype=torch.float32) for d in minibatch))
                reward_batch = torch.tensor([d.reward for d in minibatch], dtype=torch.float32)
                # reward_batch = torch.cat(tuple(torch.tensor(d.reward) for d in minibatch))
                next_state_batch = torch.stack(tuple(torch.tensor(d.next_state, dtype=torch.float32) for d in minibatch))

                # get output for the next state
                if args.target:
                    output_batch = target_critic(next_state_batch)
                else:
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

                if args.target and i % 250 == 0:
                    # Update target critic network
                    target_critic.load_state_dict(critic.state_dict())
                    target_critic.eval()
                    target_critic.float()

    # Save the replay buffer
    if args.save_buffer:
        print("==============>>>>>>>>>>> saving replay buffer")
        pickle.dump(replay_buffer, open(REPLAY_PICKLE_FILE, 'wb'))

    pickle.dump(agent, open(PICKLE_FILE, 'wb'))
    # agent = pickle.load(open(PICKLE_FILE, 'rb'))
    print("Trajectory reward = %f" % np.sum(agent.sol_reward))
    print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))

    # wait for user prompt before visualizing optimized trajectories
    # _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
    # for _ in range(100):
    #     agent.animate_result()
