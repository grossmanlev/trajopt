from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
from tqdm import tqdm
import time as timer
import numpy as np
import pickle

# =======================================
ENV_NAME = 'reacher_7dof'
PICKLE_FILE = ENV_NAME + '_mppi.pickle'
SEED = 2
N_ITER = 5
H_total = 100
# =======================================

# Load reference trajectory
reference_agent = pickle.load(open('dense_agent.pickle', 'rb'))
reference_pos = []
for i in range(len(reference_agent.sol_state)):
    reference_agent.env.set_env_state(reference_agent.sol_state[i])
    reference_pos.append(
        reference_agent.env.data.site_xpos[reference_agent.env.hand_sid])
reference_pos = np.array(reference_pos)

goal = (0, 0, 0)
reward_type = 'tracking'
reference = reference_pos

e = get_environment(ENV_NAME, reward_type=reward_type, reference=reference)
e.reset_model(seed=SEED, goal=goal)
mean = np.zeros(e.action_dim)
sigma = 1.0*np.ones(e.action_dim)
filter_coefs = [sigma, 0.25, 0.8, 0.0]

agent = MPPI(e, H=16, paths_per_cpu=40, num_cpu=1,
             kappa=25.0, gamma=1.0, mean=mean, filter_coefs=filter_coefs,
             default_act='mean', seed=SEED, reward_type=reward_type,
             reference=reference)

ts = timer.time()
for t in tqdm(range(H_total)):
    agent.train_step(niter=N_ITER, goal=goal)
    # if t % 25 == 0 and t > 0:
    #     print("==============>>>>>>>>>>> saving progress ")
    #     pickle.dump(agent, open(PICKLE_FILE, 'wb'))

# pickle.dump(agent, open('dense_agent.pickle', 'wb'))
# agent = pickle.load(open(PICKLE_FILE, 'rb'))
print("Trajectory reward = %f" % np.sum(agent.sol_reward))
print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))

# wait for user prompt before visualizing optimized trajectories
_ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
for _ in range(100):
    agent.animate_result()
