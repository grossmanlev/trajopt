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
H_total = 100 + 16
STATE_DIM = 14
H = 16
# =======================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic', default=None,
                        help='path to critic model (.pt) file')
    parser.add_argument('--iters', default=10, type=int,
                        help='number of random initializations to test')
    args = parser.parse_args()

    e = get_environment(ENV_NAME, sparse_reward=False)
    e.reset_model(seed=SEED)

    mean = np.zeros(e.action_dim)
    sigma = 1.0*np.ones(e.action_dim)
    filter_coefs = [sigma, 0.25, 0.8, 0.0]

    critic = None
    if args.critic is not None:
        critic = Critic()
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
    joint_limits /= 1.0

    rewards = []

    env_seeds = [0, 1, 2, 3, 4]
    for seed in env_seeds:
        # random_qpos = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])
        # e.set_state(random_qpos, e.init_qvel)

        e.reset_model(seed=seed)

        agent = MPPI(e, H=H, paths_per_cpu=40, num_cpu=1,
                     kappa=25.0, gamma=1.0, mean=mean,
                     filter_coefs=filter_coefs,
                     default_act='mean', seed=SEED,
                     init_seq=None)

        ts = timer.time()
        for t in tqdm(range(H_total)):

            # Actor step
            agent.train_step(critic=critic, niter=N_ITER)

        rewards.append(np.sum(agent.sol_reward))
        print("Trajectory reward = %f" % np.sum(agent.sol_reward))
        pickle.dump(agent, open('agent_116_seed_{}.pickle'.format(seed), 'wb'))

        # _ = input("Press enter to display optimized trajectory (will be played 100 times) : ")
        # print(e.data.site_xpos[e.target_sid])
        # for _ in range(10):
        #     agent.animate_result()

    print("Avg. Reward: {} ({})".format(np.mean(rewards), np.std(rewards)))

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
