from trajopt.envs.reacher_env import Reacher7DOFEnv
from trajopt.envs.continual_reacher_env import ContinualReacher7DOFEnv
from trajopt.envs.hopper import HopperEnv


def get_environment(env_name, reward_type='dense', reference=None):
    if env_name == 'reacher_7dof':
        return Reacher7DOFEnv(reward_type=reward_type, reference=reference)
    elif env_name == 'continual_reacher_7dof':
        return ContinualReacher7DOFEnv()
    elif env_name == 'hopper':
        return HopperEnv()
    else:
        print("Unknown Environment Name")
        return None
