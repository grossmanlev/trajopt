from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
from tqdm import tqdm
import time as timer
import numpy as np
import pickle
from trajopt.utils import ReplayBuffer, Tuple
from trajopt.models.critic_nets import Critic

# =======================================
ENV_NAME = 'reacher_7dof'
PICKLE_FILE = ENV_NAME + '_mppi.pickle'
SEED = 2
N_ITER = 5
H_total = 100
STATE_DIM = 14
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

replay_buffer = ReplayBuffer(max_size=10000)

# e = get_environment(ENV_NAME, reward_type=reward_type, reference=reference)
e = get_environment(ENV_NAME, reward_type='sparse')
e.reset_model(seed=SEED, goal=goal)
mean = np.zeros(e.action_dim)
sigma = 1.0*np.ones(e.action_dim)
filter_coefs = [sigma, 0.25, 0.8, 0.0]

agent = MPPI(e, H=16, paths_per_cpu=40, num_cpu=1,
             kappa=25.0, gamma=1.0, mean=mean, filter_coefs=filter_coefs,
             default_act='mean', seed=SEED, reward_type=reward_type,
             reference=reference)
critic = Critic(input_dim=STATE_DIM, inner_layer=128, batch_size=128, gamma=0.9)

ts = timer.time()
samples = []
for t in tqdm(range(H_total)):
    tuples = agent.train_step(niter=N_ITER, goal=goal)
    indices = np.random.choice(list(range(len(tuples))), 10, replace=False)
    tuples = [tuples[i] for i in indices]  # get sample of tuples
    tuples = critic.compress_states(tuples, dim=STATE_DIM)  # format tuples
    samples += tuples
samples += critic.compress_agent(agent, dim=STATE_DIM)  # add solution traj
replay_buffer.concatenate(samples)  # add to replay buffer


# TODO: Write method to take solution trajcetory and convert to valid tuples

# pickle.dump(agent, open('dense_agent.pickle', 'wb'))
# agent = pickle.load(open(PICKLE_FILE, 'rb'))
print("Trajectory reward = %f" % np.sum(agent.sol_reward))
print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))

# wait for user prompt before visualizing optimized trajectories
_ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
for _ in range(100):
    agent.animate_result()
