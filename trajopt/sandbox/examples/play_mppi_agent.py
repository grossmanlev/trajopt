import numpy as np
import pickle
import argparse
from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment

# =======================================
ENV_NAME = 'reacher_7dof'
PICKLE_FILE = ENV_NAME + '_mppi.pickle'
SEED = 12345
# =======================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', default=None,
                        help='path to agent model (.pickle) file')
    args = parser.parse_args()

    e = get_environment(ENV_NAME, sparse_reward=True)
    e.reset_model(seed=SEED)

    agent = pickle.load(open(args.agent, 'rb'))

    print("Trajectory reward = %f" % np.sum(agent.sol_reward))

    total_reward = 0.0
    for k in range(len(agent.sol_act)):
        e.set_env_state(agent.sol_state[k])
        obs, reward, _, _ = e.step(agent.sol_act[k])
        total_reward += reward
    print('Sparse reward: {}'.format(total_reward))

    # wait for user prompt before visualizing optimized trajectories
    _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
    for _ in range(100):
        agent.animate_result()
