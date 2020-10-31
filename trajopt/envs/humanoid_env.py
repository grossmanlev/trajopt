import numpy as np
import gym
from gym import utils
from trajopt.envs import mujoco_env
import os
import pybullet_envs
from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepBulletEnv, HumanoidDeepMimicBackflipBulletEnv


class HumanoidDeepMimicStandEnv(HumanoidDeepBulletEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        # start the bullet physics server
        HumanoidDeepBulletEnv.__init__(self, renders, arg_file="run_humanoid3d_stand_args.txt")


class HumanoidDeepMimicWalkEnv(HumanoidDeepBulletEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        # start the bullet physics server
        HumanoidDeepBulletEnv.__init__(self, renders, arg_file="run_humanoid3d_walk_args.txt")


class HumanoidDeepMimicJumpEnv(HumanoidDeepBulletEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        # start the bullet physics server
        HumanoidDeepBulletEnv.__init__(self, renders, arg_file="run_humanoid3d_jump_args.txt")


class HumanoidDeepMimicSpinkickEnv(HumanoidDeepBulletEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        # start the bullet physics server
        HumanoidDeepBulletEnv.__init__(self, renders, arg_file="run_humanoid3d_spinkick_args.txt")


class HumanoidEnv(utils.EzPickle):
    def __init__(self, renders=False):
        self.env_name = 'HumanoidDeepMimicBackflipBulletEnv-v1'
        self.env = HumanoidDeepMimicSpinkickEnv(renders=renders)
        self.env._time_step = 1./180.
        self.env.state = self.env.reset()
        self.real_step = True
        self.env_timestep = 0
        self.alpha = 1.0
        self.reached_goal = False
        self.done = False

        utils.EzPickle.__init__(self)

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.humanoid = self.env._internal_env._humanoid
        self.pb = self.env._p
        self.p_target = [-0.2, 1.8, 0.2]
        self.target_radius = 0.2
        self.target_color = [56./255., 208./255., 87./255., 0.6]

        self.ballVisID = self.pb.createVisualShape(
            shapeType=self.pb.GEOM_SPHERE, radius=self.target_radius,
            rgbaColor=self.target_color)
        self.ballID = self.pb.createMultiBody(
            baseVisualShapeIndex=self.ballVisID, basePosition=self.p_target)

    def calculateGoalReward(self, done):
        ''' DeepMimic "Strike" goal reward structure '''
        humanoid = self.env._internal_env._humanoid
        p_link = humanoid._pybullet_client.getLinkState(
            humanoid._sim_model, 5)[0]
        dist = np.linalg.norm(np.array(self.p_target) - np.array(p_link))
        if self.reached_goal or dist < self.target_radius:
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

    def COMTrackingReward(self, exp=True):
        humanoid = self.env._internal_env._humanoid
        p_chest_sim = humanoid._pybullet_client.getLinkState(
            humanoid._sim_model, 1)[0]
        p_chest_kin = humanoid._pybullet_client.getLinkState(
            humanoid._kin_model, 1)[0]
        dist = np.linalg.norm(np.array(p_chest_kin) - np.array(p_chest_sim))
        # if dist > 1.0:
        #     return -10
        if not exp:
            return -dist
        return np.exp(-4 * dist)

    def AllTrackingReward(self, exp=True):
        joints = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
        humanoid = self.env._internal_env._humanoid
        dist = 0.
        for j in joints:
            p_j_sim = humanoid._pybullet_client.getLinkState(
                humanoid._sim_model, j)[0]
            p_j_kin = humanoid._pybullet_client.getLinkState(
                humanoid._kin_model, j)[0]
            dist += np.linalg.norm(np.array(p_j_kin) - np.array(p_j_sim))
        if not exp:
            return -dist
        return np.exp(-4 * dist)

    def step(self, a):
        self.env_timestep += 1
        # explicitly clipping actions
        np.clip(a, self.env.action_space.low, self.env.action_space.high)
        state, reward, done, info = self.env.step(a)
        reward = (
            0.0 * reward +
            7.0 * self.calculateGoalReward(done) +
            5.0 * self.feetTrackingReward() +
            10.0 * self.COMTrackingReward(exp=True) +
            100.0 * self.AllTrackingReward(exp=True) +
            0.06 * -np.linalg.norm(a) +
            0.0 * np.exp(-np.linalg.norm(a)))
        if done:
            # reward -= 0.5
            self.done = True

        humanoid = self.env._internal_env._humanoid
        p_neck_kin = humanoid._pybullet_client.getLinkState(
            humanoid._kin_model, 2)[0]
        p_neck_sim = humanoid._pybullet_client.getLinkState(
            humanoid._sim_model, 2)[0]
        # if np.linalg.norm(p_neck) > 2.0 or p_neck[1] < 1.0:  # neck/head soft constraint
        #     reward -= 10
        if p_neck_sim[1] < 1.0:
            reward -= 10
        # if np.linalg.norm(np.array(p_neck_kin) - np.array(p_neck_sim)) > 1.0:
        #     reward -= 10

        return state, reward, done, info

    def _get_obs(self, goal_dist=False):
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
