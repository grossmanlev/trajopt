import numpy as np
from gym import utils
from trajopt.envs import mujoco_env
from mujoco_py import MjViewer
import os


class Reacher7DOFEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='dense', reference=None):

        # trajopt specific attributes
        self.env_name = 'reacher_7dof'
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0
        self.alpha = 1.0

        self.reward_type = reward_type  # 'dense', 'sparse', 'tracking'
        self.reference = reference  # reference trajectory (np.array)
        # ensure if tracking reward, then we have a reference
        # assert ((reward_type is not 'tracking') or
        #         (reward_type is 'tracking' and reference is not None))

        # placeholder
        self.hand_sid = -2
        self.target_sid = -1

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/sawyer.xml', 2)
        utils.EzPickle.__init__(self)
        self.observation_dim = 26
        self.action_dim = 7

        self.hand_sid = self.model.site_name2id("finger")
        self.target_sid = self.model.site_name2id("target")

        hand_pos = self.data.site_xpos[self.hand_sid]
        target_pos = self.data.site_xpos[self.target_sid]
        self.goal_dist = np.array([np.linalg.norm(hand_pos-target_pos)])
        # self.init_qpos[0] = -1.0
        # self.init_qpos[1] = 1.0

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        # keep track of env timestep (needed for continual envs)
        self.env_timestep += 1
        hand_pos = self.data.site_xpos[self.hand_sid]
        target_pos = self.data.site_xpos[self.target_sid]
        dist = np.linalg.norm(hand_pos-target_pos)
        self.goal_dist = np.array([dist])
        reward = - 10.0 * dist - 0.25 * np.linalg.norm(self.data.qvel)

        if self.reward_type == 'sparse':
            if dist < 0.025:
                reward = 100.0
            elif dist > 0.8:
                reward = -100.0
            else:
                reward = -10.0
        elif self.reward_type == 'tracking':
            ref_pos = self.reference[self.env_timestep]  # reference position
            dist = np.linalg.norm(hand_pos - ref_pos)
            reward = -10.0 * dist - 0.25 * np.linalg.norm(self.data.qvel)
            # reward = -10.0 * dist
        elif self.reward_type == 'cooling':
            sparse_reward = 0.0
            if dist < 0.025:
                sparse_reward = 100.0
            elif dist > 0.8:
                sparse_reward = -100.0
            else:
                sparse_reward = -10.0

            track_reward = 0.0
            ref_pos = self.reference[self.env_timestep]  # reference position
            dist = np.linalg.norm(hand_pos - ref_pos)
            track_reward = -10.0 * dist - 0.25 * np.linalg.norm(self.data.qvel)

            reward = (self.alpha * track_reward) + ((1.0 - self.alpha) * sparse_reward)
        else:
            reward = - 10.0 * dist - 0.25 * np.linalg.norm(self.data.qvel)

        ob = self._get_obs()
        rtn_dct = self.get_env_infos()
        return ob, reward, False, rtn_dct

    def step(self, a):
        # overloading to preserve backwards compatibility
        return self._step(a)

    def _get_obs(self, goal_dist=False):
        if goal_dist:
            return np.concatenate([
                self.data.qpos.flat,
                self.data.qvel.flat,
                self.data.site_xpos[self.hand_sid],
                self.data.site_xpos[self.target_sid],
                self.goal_dist,
            ])
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            self.data.site_xpos[self.hand_sid],
            self.data.site_xpos[self.target_sid],
        ])

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self, goal=None):
        # target_pos = np.array([0.1, 0.1, 0.1])
        target_pos = np.zeros(3)
        if self.seeding is True:
            target_pos[0] = self.np_random.uniform(low=-0.1, high=0.1)
            target_pos[1] = self.np_random.uniform(low=-0.1, high=0.1)
            target_pos[2] = self.np_random.uniform(low=-0.1, high=0.1)
        if goal is not None:
            target_pos = np.array(goal)
        # print(target_pos)
        self.model.site_pos[self.target_sid] = target_pos
        self.data.site_xpos[self.target_sid] = target_pos
        self.sim.forward()

    def reset_model(self, seed=None, goal=None, alpha=None):
        if seed is not None:
            self.seeding = True
            self._seed(seed)
        if alpha is not None:
            self.alpha = alpha
        self.robot_reset()
        self.target_reset(goal)
        self.env_timestep = 0  # reset timestep
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.site_pos[self.target_sid].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos, timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        self.set_state(qp, qv)
        self.model.site_pos[self.target_sid] = target_pos
        self.data.site_xpos[self.target_sid] = target_pos
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
