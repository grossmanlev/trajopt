from trajopt.envs.reacher import Reacher2DOFEnv
from trajopt.envs.reacher_env import Reacher7DOFEnv
from trajopt.envs.continual_reacher_env import ContinualReacher7DOFEnv
from trajopt.envs.hopper import HopperEnv


def get_environment(env_name, sparse_reward=False):
    if env_name == 'reacher_7dof':
        return Reacher7DOFEnv(sparse_reward=sparse_reward)
    elif env_name == 'reacher_2dof':
        return Reacher2DOFEnv(sparse_reward=sparse_reward)
    elif env_name == 'continual_reacher_7dof':
        return ContinualReacher7DOFEnv()
    elif env_name == 'hopper':
        return HopperEnv()
    else:
        print("Unknown Environment Name")
        return None
