"""
This implements a shooting trajectory optimization algorithm.
The closest known algorithm is perhaps MPPI and hence we stick to that terminology.
Uses a filtered action sequence to generate smooth motions.
"""

import numpy as np
from trajopt.algos.trajopt_base import Trajectory
from trajopt.utils import gather_paths_parallel
from trajopt.utils import ReplayBuffer, Tuple

import torch


class MPPI(Trajectory):
    def __init__(self, env, H, paths_per_cpu,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 default_act='repeat',
                 seed=123,
                 init_seq=None,
                 ):
        self.env, self.seed = env, seed
        self.n, self.m = env.observation_dim, env.action_dim
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu

        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]
        self.default_act = default_act

        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []

        # self.env.reset_model()
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env._get_obs())
        self.act_sequence = np.ones((self.H, self.m)) * self.mean

        if init_seq is not None and init_seq.shape == self.act_sequence.shape:
            self.act_sequence = init_seq

    def update(self, paths, use_critic=False):
        num_traj = len(paths)
        act = np.array([paths[i]["actions"] for i in range(num_traj)])
        R = self.score_trajectory(paths, use_critic=use_critic)
        S = np.exp(self.kappa*(R-np.max(R)))

        # blend the action sequence
        weighted_seq = S*act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)
        self.act_sequence = act_sequence

    def advance_time(self, act_sequence=None):
        act_sequence = self.act_sequence if act_sequence is None else act_sequence
        # accept first action and step
        self.sol_act.append(act_sequence[0].copy())
        state_now = self.sol_state[-1].copy()
        self.env.set_env_state(state_now)
        _, r, _, _ = self.env.step(act_sequence[0])
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env._get_obs())
        self.sol_reward.append(r)

        # get updated action sequence
        self.act_sequence[:-1] = act_sequence[1:]
        if self.default_act == 'repeat':
            self.act_sequence[-1] = self.act_sequence[-2]
        else:
            self.act_sequence[-1] = self.mean.copy()

    def score_trajectory(self, paths, use_critic=False):
        scores = np.zeros(len(paths))
        for i in range(len(paths)):
            scores[i] = 0.0
            if use_critic:
                for t in range(paths[i]["critic_rewards"].shape[0]):
                    scores[i] += (self.gamma**t)*paths[i]["critic_rewards"][t]
            else:
                for t in range(paths[i]["rewards"].shape[0]):
                    scores[i] += (self.gamma**t)*paths[i]["rewards"][t]
        return scores

    def do_rollouts(self, seed, goal=None):
        paths = gather_paths_parallel(self.env.env_name,
                                      self.sol_state[-1],
                                      self.act_sequence,
                                      self.filter_coefs,
                                      seed,
                                      goal,
                                      self.paths_per_cpu,
                                      self.num_cpu,
                                      )
        return paths

    def train_step(self, critic=None, niter=1, act_sequence=None, goal=None):
        # states = []
        # actions = []
        # rewards = []
        replay_tuples = []
        t = len(self.sol_state) - 1
        for _ in range(niter):
            paths = self.do_rollouts(self.seed+t, goal=goal)

            for i, path in enumerate(paths):
                critic_rewards = []

                for j in range(len(path["states"])):
                    if j < len(path["states"]) - 1:
                        replay_tuples.append(
                            Tuple(path["states"][j],
                                  path["actions"][j],
                                  path["rewards"][j],
                                  path["states"][j + 1]))
                    # Compute state values based on Critic
                    if critic is not None:
                        # HACK to get value of next state, s'
                        #import pdb; pdb.set_trace()
                        # print('Goal: {}'.format(path["next_observations"][j][-3:]))
                        critic_state = np.concatenate((path["next_observations"][j][:14], path["next_observations"][j][-3:]))
                        critic_state = torch.tensor(critic_state, dtype=torch.float32)
                        critic_state = critic_state.unsqueeze(0)
                        critic_reward = critic(critic_state).detach().numpy()
                        critic_rewards.append(critic_reward)
                        # if j < len(path["states"]) - 1:
                        #     replay_tuples.append(
                        #         Tuple(path["states"][j],
                        #               path["actions"][j],
                        #               critic_reward,
                        #               path["states"][j + 1]))
                    # else:
                    #     if j < len(path["states"]) - 1:
                    #         replay_tuples.append(
                    #             Tuple(path["states"][j],
                    #                   path["actions"][j],
                    #                   path["rewards"][j],
                    #                   path["states"][j + 1]))
                paths[i]["critic_rewards"] = np.array(critic_rewards)

            self.update(paths, use_critic=(critic is not None))
        self.advance_time(act_sequence=act_sequence)
        return replay_tuples
        # return dict(states=states, actions=actions, rewards=rewards)
