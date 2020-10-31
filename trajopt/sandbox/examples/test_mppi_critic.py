import argparse
from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
from trajopt.models.critic_nets import Critic
from tqdm import tqdm
import time as timer
import numpy as np
import pickle

import torch


# =======================================
ENV_NAME = 'reacher_7dof'
PICKLE_FILE = ENV_NAME + '_mppi.pickle'
SEED = 12345
N_ITER = 5
H_total = 100
STATE_DIM = 14
H = 16
# =======================================


def custom_reward_fn(sol_obs, radius=0.025):
    """ 1 if the robot was within radius meters of the goal"""
    dsts = [np.linalg.norm(obs[-6:-3]-obs[-3:]) for obs in sol_obs]
    return int(min(dsts) < radius)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic', default=None,
                        help='path to critic model (.pt) file')
    parser.add_argument('--iters', default=10, type=int,
                        help='number of random initializations to test')
    parser.add_argument('--goals', action='store_true')
    args = parser.parse_args()

    if args.goals:
        STATE_DIM += 1

    e = get_environment(ENV_NAME, reward_type='sparse')
    goal = np.zeros(3)
    # e.reset_model(seed=SEED, goal=goal)
    seed = 11

    mean = np.zeros(e.action_dim)
    sigma = 1.0*np.ones(e.action_dim)
    filter_coefs = [sigma, 0.25, 0.8, 0.0]

    critic = None
    if args.critic is not None:
        critic = Critic(input_dim=STATE_DIM)
        critic.load_state_dict(torch.load(args.critic))
        critic.eval()
        critic.float()

    joint_limits = np.array([[-2.2854, 1.714602],
                             [-0.5236, 1.3963],
                             [-1.5, 1.7],
                             [-2.3213, 0],
                             [-1.5, 1.5],
                             [-1.094, 0],
                             [-1.5, 1.5]])
    joint_limits /= 2.0

    np_seed = 0
    rewards = []
    custom_rewards = []
    agents = []

    for _ in tqdm(range(args.iters), disable=True):
        # e = get_environment(ENV_NAME, sparse_reward=False)
        e.reset_model(seed=seed)
        print('Goal: {}'.format(e.get_env_state()['target_pos']))
        goal = e.get_env_state()['target_pos']

        # np.random.seed(seed=np_seed)
        # random_qpos = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])
        # e.set_state(random_qpos, e.init_qvel)

        agent = MPPI(e, H=H, paths_per_cpu=40, num_cpu=1,
                     kappa=25.0, gamma=1.0, mean=mean,
                     filter_coefs=filter_coefs,
                     default_act='mean', seed=SEED,
                     init_seq=None)

        ts = timer.time()

        for t in tqdm(range(H_total), disable=False):

            # Actor step
            tuples = agent.train_step(
                critic=critic, niter=N_ITER, goal=goal, dim=STATE_DIM)

        reward = np.sum(agent.sol_reward)
        custom_reward = custom_reward_fn(agent.sol_obs)
        rewards.append(reward)
        custom_rewards.append(custom_reward)

        # pickle.dump(agent, open('sparse_reward_agent.pickle', 'wb'))

        print("Trajectory reward = %f" % reward)
        print("Custom reward = %f" % custom_reward)

        _ = input("Press enter to display optimized trajectory (will be played 100 times) : ")
        for _ in range(10):
            agent.animate_result()
        agents.append(agent)

        np_seed += 1

    print("Avg. Reward: {} ({})".format(
        np.mean(rewards), np.std(rewards)))
    print("Avg. Custom Reward: {} ({})".format(
        np.mean(custom_rewards), np.std(custom_rewards)))

    for agent in agents:
        agent.animate_result()

    # agent = pickle.load(open('116_agent.pickle', 'rb'))

    # sol_actions = np.array(agent.sol_act)  # should be (100, 7)
    # init_seq = sol_actions[:H]

    # agent = MPPI(e, H=H, paths_per_cpu=40, num_cpu=1,
    #              kappa=25.0, gamma=1.0, mean=mean,
    #              filter_coefs=filter_coefs,
    #              default_act='mean', seed=SEED,
    #              init_seq=None)

    # ts = timer.time()
    # for t in tqdm(range(H_total)):

    #     # Actor step
    #     # print(sol_actions[t:t+H][0])
    #     if t < 0:
    #         agent.train_step(critic=None, niter=N_ITER, act_sequence=sol_actions[t:t+H])
    #     else:
    #         agent.train_step(critic=None, niter=N_ITER)

    # print("Time for trajectory optimization = %f seconds" % (timer.time()-ts))
    # # pickle.dump(agent, open('good_agent.pickle', 'wb'))
    # print("Trajectory reward = %f" % np.sum(agent.sol_reward))

    # # wait for user prompt before visualizing optimized trajectories
    # _ = input("Press enter to display optimized trajectory (will be played 100 times) : ")
    # for _ in range(100):
    #     agent.animate_result()
