import numpy as np

from trajopt.envs import mujoco_env
from dm_control.composer.environment import Environment


class EnvironmentWrapper:
    def __init__(self, env: Environment):
        self.env = env
        self.sim = self.env.physics
        self.data = self.sim.data
        self.model = self.sim.model

        self.env_timestep = 0
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))

        self.observation_dim = 26
        self.action_dim = 7

    def step(self, a):
        timstep = self.env.step(a)
        return self._get_obs(), timestep.reward, False, timestep

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
        ])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def reset_model(self):
        self.robot_reset()
        self.env_timestep = 0  # reset timestep
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _reset(self):
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        state = self.sim.get_state()
        for i in range(self.model.nq):
            state.qpos[i] = qpos[i]
        for i in range(self.model.nv):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        self.set_state(qp, qv)
        self.env_timestep = state['timestep']
        self.sim.forward()
