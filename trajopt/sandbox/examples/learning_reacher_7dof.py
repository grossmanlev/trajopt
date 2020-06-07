from spinup import ppo_pytorch as ppo
from spinup import sac_pytorch as sac
from trajopt.envs.utils import get_environment
import torch
import gym
import argparse

ENV_NAME = 'reacher_7dof'
H_total = 100


def main(alg):
    env_fn = lambda : get_environment(ENV_NAME, reward_type='sparse')

    ac_kwargs = dict(hidden_sizes=[128, 128], activation=torch.nn.ReLU)

    logger_kwargs = dict(output_dir='./corl/{}'.format(alg), exp_name=alg+'_reacher_sparse')

    if alg == 'PPO':
        # ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=4000, epochs=250,
        #     logger_kwargs=logger_kwargs)
        ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=10000, epochs=250,
            max_ep_len=H_total, logger_kwargs=logger_kwargs)
    elif alg == 'SAC':
        sac(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=10000, epochs=250,
            max_ep_len=H_total, logger_kwargs=logger_kwargs)
    else:
        print('Invalid Algorithm. Exiting...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', default='PPO',
                        help='algorithm (PPO or SAC)')
    args = parser.parse_args()
    main(alg=args.alg)
