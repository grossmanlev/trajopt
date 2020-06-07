from trajopt.envs.reacher_env import Reacher7DOFEnv
from trajopt.envs.continual_reacher_env import ContinualReacher7DOFEnv
from trajopt.envs.hopper import HopperEnv
from trajopt.envs.humanoid_env import HumanoidEnv
import gym


def get_environment(env_name, reward_type='dense', reference=None, renders=False):
    if env_name == 'reacher_7dof':
        return Reacher7DOFEnv(reward_type=reward_type, reference=reference)
    elif env_name == 'continual_reacher_7dof':
        return ContinualReacher7DOFEnv()
    elif env_name == 'hopper':
        return HopperEnv()
    elif env_name == 'HumanoidDeepMimicBackflipBulletEnv-v1':
        return HumanoidEnv(renders=renders)
    else:
        print("Unknown Environment Name")
        return None
