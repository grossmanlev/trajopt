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
# =======================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('critic', default=None,
                        help='path to critic model (.pt) file')
    args = parser.parse_args()

    e = get_environment(ENV_NAME)
    e.reset_model(seed=SEED)

    mean = np.zeros(e.action_dim)
    sigma = 1.0*np.ones(e.action_dim)
    filter_coefs = [sigma, 0.25, 0.8, 0.0]

    critic = Critic()
    critic.load_state_dict(torch.load(args.critic))
    critic.eval()
    critic.float()

    # agent = MPPI(e, H=16, paths_per_cpu=40, num_cpu=1,
    #              kappa=25.0, gamma=1.0, mean=mean,
    #              filter_coefs=filter_coefs,
    #              default_act='mean', seed=SEED)

    # ts = timer.time()
    # for t in tqdm(range(H_total)):

    #     # Actor step
    #     agent.train_step(critic=critic, niter=N_ITER)

    # print("Trajectory reward = %f" % np.sum(agent.sol_reward))
    # print("Time for trajectory optimization = %f seconds" % (timer.time()-ts))

    # # wait for user prompt before visualizing optimized trajectories
    # _ = input("Press enter to display optimized trajectory (will be played 100 times) : ")
    # for _ in range(100):
    #     agent.animate_result()

    # new_q = np.zeros(7)
    # # new_q = np.array([10.99164932,  0.06841799, -1.50792112, -1.56400837, -1.52414601,
    # #                                    0.01832143, -1.52851301])
    # new_q[0] = 1

    # e.set_state(new_q, e.init_qvel)
    # print(e._get_obs())

    for _ in range(10):
        e.render()
        print(e._get_obs())
        a = np.zeros(7)
        e.step(a)
    e.close()
