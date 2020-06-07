import numpy as np
import gym
from gym import utils
from trajopt.envs import mujoco_env
import os
import pybullet_envs
from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepBulletEnv, HumanoidDeepMimicBackflipBulletEnv


class HumanoidDeepMimicSpinkickEnv(HumanoidDeepBulletEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        # start the bullet physics server
        HumanoidDeepBulletEnv.__init__(self, renders, arg_file="run_humanoid3d_spinkick_args.txt")


class HumanoidEnv(utils.EzPickle):
    def __init__(self, renders=False):
        self.env_name = 'HumanoidDeepMimicBackflipBulletEnv-v1'
        self.env = HumanoidDeepMimicSpinkickEnv(renders=renders)
        self.env._time_step = 1./120.
        self.env.state = self.env.reset()
        self.real_step = True
        self.env_timestep = 0
        self.alpha = 1.0
        self.reached_goal = False
        self.done = False

        utils.EzPickle.__init__(self)

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.state_to_id = dict()

    def calculateGoalReward(self, done):
        ''' DeepMimic "Strike" goal reward structure '''
        p_target = [-0.2, 1.8, 0.2]
        humanoid = self.env._internal_env._humanoid
        p_link = humanoid._pybullet_client.getLinkState(
            humanoid._sim_model, 5)[0]
        dist = np.linalg.norm(np.array(p_target) - np.array(p_link))
        if self.reached_goal or dist < 0.2:
            return 1.0
        return np.exp(-4 * dist)

    def feetTrackingReward(self):
        humanoid = self.env._internal_env._humanoid
        p_right_sim = humanoid._pybullet_client.getLinkState(
            humanoid._sim_model, 5)[0]
        p_left_sim = humanoid._pybullet_client.getLinkState(
            humanoid._sim_model, 11)[0]
        p_right_kin = humanoid._pybullet_client.getLinkState(
            humanoid._kin_model, 5)[0]
        p_left_kin = humanoid._pybullet_client.getLinkState(
            humanoid._kin_model, 11)[0]

        dist = (np.linalg.norm(np.array(p_right_kin) - np.array(p_right_sim))
                + np.linalg.norm(np.array(p_left_kin) - np.array(p_left_sim)))
        return np.exp(-4 * dist)

    def step(self, a):
        self.env_timestep += 1
        state, reward, done, info = self.env.step(a)
        reward = 1.0 * reward + 0.8 * self.calculateGoalReward(done) + 0.3 * self.feetTrackingReward()
        if done:
            # reward -= 0.5
            self.done = True
        reward += 0.1 * np.exp(-4*np.linalg.norm(a))

        humanoid = self.env._internal_env._humanoid
        p_neck = humanoid._pybullet_client.getLinkState(
            humanoid._sim_model, 2)[0]
        if np.linalg.norm(p_neck) > 2.0 or p_neck[1] < 1.0:
            reward -= 10

        return state, reward, done, info

    def _get_obs(self):
        return self.env.state

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
        self.env.state = self.env.reset()

        # HACK to start at beginning of motion
        self.env._internal_env.t = 0
        self.env._internal_env._humanoid.setSimTime(0)
        self.env._internal_env._humanoid.resetPose()

        return self.env.state

    def reset_model(self, goal=None, alpha=None):
        return self.reset()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        if self.real_step:  # if the main env in MPPI, then save states
            self.env._p.saveBullet('/home/robocup/levs_files/trajopt/bullet/{}.bullet'.format(self.env_timestep))
        return self._get_obs()

    def set_env_state(self, state_id):
        # TODO
        self.env._p.restoreState(fileName='/home/robocup/levs_files/trajopt/bullet/{}.bullet'.format(state_id))
        self.env.state = self.env._internal_env.record_state(-1)

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())
