import numpy as np
from gym import utils
# from trajopt.envs import mujoco_env
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer
import os


class Reacher2DOFEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, sparse_reward=False):

        # trajopt specific attributes
        self.env_name = 'reacher_2dof'
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0
        self.sparse_reward = sparse_reward

        # placeholder
        # self.hand_sid = -2
        # self.target_sid = -1

        utils.EzPickle.__init__(self)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/reacher.xml', 2)
        self.observation_dim = 11
        self.action_dim = 2

        # self.hand_sid = self.model.site_name2id("finger")
        # self.target_sid = self.model.site_name2id("target")

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = 10*reward_dist + 0.25*reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def step(self, a):
        # overloading to preserve backwards compatibility
        return self._step(a)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def target_reset(self):
        self.goal = np.array([0.1, 0.1])
        if self.seeding:
            while True:
                self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self.goal) < 0.2:
                    break
        return self.goal

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        # qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos = self.init_qpos
        qpos[-2:] = self.target_reset()
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        # print(self.get_body_com("target"))
        # print(self.get_body_com("fingertip"))
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    # def target_reset(self):
    #     target_pos = np.array([0.1, 0.1, 0.1])
    #     if self.seeding is True:
    #         target_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
    #         target_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
    #         target_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
    #     self.model.site_pos[self.target_sid] = target_pos
    #     self.sim.forward()

    # def reset_model(self, seed=None):
    #     if seed is not None:
    #         self.seeding = True
    #         self.seed(seed)
    #     self.robot_reset()
    #     self.target_reset()
    #     return self._get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.goal
        # target_pos = self.get_body_com("target")[:2]
        # print(target_pos)
        # print(self.get_body_com("target"))
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos, timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        qp[-2:] = target_pos
        qv[-2:] = 0
        self.set_state(qp, qv)
        # self.model.site_pos[self.target_sid] = target_pos
        self.env_timestep = state['timestep']
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 1.2
