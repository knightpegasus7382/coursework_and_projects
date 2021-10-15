#!/usr/bin/env python

import click
import numpy as np
import gym


def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)
    
def include_bias(inp):
	return np.append(inp, 1)	 

def log_pi(ob, theta):
    s_ = include_bias(ob)
    mean = s_.dot(np.transpose(theta))
    cov = [[1, 0], [0, 1]]
    pi = np.squeeze(np.random.multivariate_normal(mean, cov, 1))
    logpi = np.log(pi)
    return logpi
    
def grad_log_pi(pol, ob):
	s_ = include_bias(ob)
    s_d = np.expand_dims(np.array(s_), axis = 0)
    pol_d = np.expand_dims(np.array(pol), axis = 0)
    gradpi = np.transpose(pol_d).dot(s_d)
    return grad_pi

@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'chakra':
        from rlpa2 import chakra
        env = gym.make('chakra')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)

    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))

    while True:
        ob = env.reset()
        done = False
        rewards = []
        while not done:
            action = get_action(theta, ob, rng=rng)
            next_ob, rew, done, _ = env.step(action)
            ob = next_ob
            env.render()
            rewards.append(rew)

        print("Episode reward: %.2f" % np.sum(rewards))

if __name__ == "__main__":
    main()
