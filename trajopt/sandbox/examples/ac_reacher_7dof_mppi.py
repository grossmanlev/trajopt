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
# STATE_DIM = 7
H = 16
# =======================================


def custom_reward_fn(sol_reward):
    """ 1 if any of the rewards is +100 (reached goal), 0 otherwise."""
    return int(100 in [int(elt) for elt in sol_reward])


def test_goals(critic, seeds=None, goals=None, dim=14):
    print('=' * 20)

    if seeds is not None:
        iters = len(seeds)
    elif goals is not None:
        iters = len(goals)
    else:
        return

    for i in range(iters):
        e = get_environment(ENV_NAME, sparse_reward=True)
        if seeds is not None:
            e.reset_model(seed=seeds[i])
        else:
            e.reset_model(seed=None, goal=goals[i])
        goal = e.get_env_state()['target_pos']
        mean = np.zeros(e.action_dim)
        sigma = 1.0*np.ones(e.action_dim)
        filter_coefs = [sigma, 0.25, 0.8, 0.0]

        agent_test = MPPI(e, H=H, paths_per_cpu=40, num_cpu=1,
                          kappa=25.0, gamma=1.0, mean=mean,
                          filter_coefs=filter_coefs,
                          default_act='mean', seed=SEED,
                          init_seq=None)

        for t in tqdm(range(H_total)):
            agent_test.train_step(critic=critic, niter=N_ITER, dim=dim, goal=goal)

        print("Trajectory reward = %f" % np.sum(agent_test.sol_reward))
        print("Custom reward = %f" % custom_reward_fn(agent_test.sol_reward))

    print('=' * 20)


def test_critic(critic, dim=14):
    e = get_environment(ENV_NAME, reward_type='sparse')
    set_goal = (0.0, 0.0, 0.0)
    e.reset_model(seed=None, goal=set_goal)
    mean = np.zeros(e.action_dim)
    sigma = 1.0*np.ones(e.action_dim)
    filter_coefs = [sigma, 0.25, 0.8, 0.0]

    test_agent = MPPI(e, H=H, paths_per_cpu=40, num_cpu=1,
                      kappa=25.0, gamma=1.0, mean=mean,
                      filter_coefs=filter_coefs,
                      default_act='mean', seed=SEED,
                      reward_type='sparse')
    for t in tqdm(range(H_total)):
        test_agent.train_step(critic=critic, niter=N_ITER,
                              goal=set_goal, dim=dim)
    return test_agent


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--critic', default=None,
        help='path to critic model (.pt) file'
    )
    parser.add_argument(
        '--train', action='store_true',
        help='train the critic net?'
    )
    parser.add_argument(
        '--save_buffer', action='store_true',
        help='save the replay buffer?'
    )
    parser.add_argument(
        '--target', default=True, type=bool,
        help="use target network? (default: True)"
    )
    parser.add_argument(
        '--epochs', default=100, type=int, help="actor-critic epochs")
    parser.add_argument('--eta', default=0.9, type=float, help="eta param")
    parser.add_argument('--goals', action='store_true')
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument(
        '--iters', default=2000, type=int, help="critic training iterations")
    parser.add_argument(
        '--save', action='store_true', help="save critic network and agent")
    parser.add_argument(
        '--POLO', action='store_true', help="run POLO algorithm")


if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()

    # Check to add goal position to state space
    if args.goals:
        STATE_DIM += 3

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    # Load reference trajectory
    reference_agent = pickle.load(open('dense_agent.pickle', 'rb'))
    reference_pos = []
    for i in range(len(reference_agent.sol_state)):
        reference_agent.env.set_env_state(reference_agent.sol_state[i])
        reference_pos.append(
            reference_agent.env.data.site_xpos[reference_agent.env.hand_sid])
    reference_pos = np.array(reference_pos)
    reward_type = 'cooling'
    reference = reference_pos

    e = get_environment(ENV_NAME, reward_type='sparse')  # need sparse for sol_reward
    mean = np.zeros(e.action_dim)
    sigma = 1.0*np.ones(e.action_dim)
    filter_coefs = [sigma, 0.25, 0.8, 0.0]

    replay_buffer = ReplayBuffer(max_size=10000)

    critic = Critic(
        num_iters=args.iters,
        input_dim=STATE_DIM,
        inner_layer=128,
        batch_size=128,
        gamma=0.9
    )
    if args.critic is not None:
        critic.load_state_dict(torch.load(args.critic))
    critic.eval()
    critic.float()

    # Set up target critic network
    if args.target:
        target_critic = Critic(
            num_iters=args.iters,
            input_dim=STATE_DIM,
            inner_layer=128
        )
        target_critic.load_state_dict(critic.state_dict())
        target_critic.eval()
        target_critic.float()

    # set up optimizer
    optimizer = optim.Adam(critic.parameters(), lr=args.lr)
    milestones = list(range(0, args.iters * 100, int(args.iters * 100 / 4)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)
    criterion = nn.MSELoss()

    eta = args.eta
    set_goal = (0.0, 0.0, 0.0)

    writer_x = 0
    rewards = []

    # set alpha parameter to determine level of tracking vs. learned controller
    alpha = 1.0
    if args.POLO:
        alpha = 0.0

    for x in range(args.epochs):
        e.reset_model(seed=None, goal=set_goal, alpha=alpha)
        print('Goal: {}'.format(e.get_env_state()['target_pos']))
        goal = e.get_env_state()['target_pos']
        print('*'*36)
        print('Round: {}, Alpha: {}'.format(x, alpha))
        print()

        agent = MPPI(
            e, H=H, paths_per_cpu=40, num_cpu=1,
            kappa=25.0, gamma=1.0, mean=mean,
            filter_coefs=filter_coefs,
            default_act='mean', seed=SEED,
            reward_type=reward_type, reference=reference
        )

        samples = []
        ts = timer.time()
        for t in tqdm(range(H_total)):
            tuples = agent.train_step(
                critic=critic, niter=N_ITER, goal=goal, dim=STATE_DIM)

        samples += critic.compress_agent(agent, dim=STATE_DIM)  # add solution traj
        samples += critic.compress_agent(agent, dim=STATE_DIM)  # add solution traj
        replay_buffer.concatenate(samples)  # add to replay buffer

        # test_agent = test_critic(critic, dim=STATE_DIM)  # test critic using just sparse reward
        tmp_reward = np.sum(agent.sol_reward)
        rewards.append(tmp_reward)
        print("Trajectory reward = %f" % tmp_reward)
        writer.add_scalar('Trajectory Return', tmp_reward, writer_x)
        writer_x += 1

        if args.save:
            pickle.dump(agent, open(AC_SAVE_DIR + '/ac_test_agent_{}_{}.pickle'.format(x, s), 'wb'))
            torch.save(critic.state_dict(), AC_SAVE_DIR + '/ac_test_critic_{}_{}.pt'.format(x, s))

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
                    if reward_batch[j] > 10.0:
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
                scheduler.step()

                if i % 500 == 0:
                    print('Loss: {}'.format(loss))

                if args.target and i % int(args.iters / 8) == 0:
                    # Update target critic network
                    target_critic.load_state_dict(critic.state_dict())
                    target_critic.eval()
                    target_critic.float()

            # test_goals(critic, seeds=None, goals=[(0, 0, 0), set_goal], dim=STATE_DIM)
        alpha *= eta

    print(rewards)
    if args.save:
        np.save(AC_SAVE_DIR + '/sparse_rewards.npy', np.array(rewards))

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
