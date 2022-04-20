"""
RL HW5 Q5

Author: Sam Donald
Date: 16/11/2021

This script uses the REINFORCE policy gradient algorithm with function approximation using fourier basis outlined here:
https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf

This is applied to the OpenAI CartPole-v0 environment (and can also be applied to CartPole-v1)

On average the environment is solved (195 point average over 100 consecutive runs) within ~105 runs with the
current hyper parameters. The hyper parameters are exceptionally sensitive!

Of the 10 sample seeds randomly chosen, 8 are solved within ~110 episodes, 1 in ~300, and 1 in ~3000.

TODO - expand to actor critic, tidy code!
"""

import gym
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product


class FourierBasis:
    """Basis used for function approximation """

    def __init__(self, state_space, action_space, order, value_clip, max_non_zero=4):

        self.action_space = action_space
        # FIXME Deep copy to not modify Q-learning env state space - quick work around (scared it would break env)
        self.state_space = copy.deepcopy(state_space)
        self.state_dim = state_space.shape[0]
        self.action_dim = self.action_space.n

        self.value_clip = value_clip

        self.action_space = action_space
        self.order = order
        self.max_non_zero = max_non_zero
        self.coefficients = self.generate_coefficients()

        self.bound_state_space()

    def generate_learning_rates(self, alpha):
        """ Generates the learning rates for each individual coefficient (as per paper linked)"""

        lrs = np.linalg.norm(self.coefficients, axis=1, ord=2)

        # " (avoiding division by zero by setting α0 = α1 where c0 = 0) - Scaling Gradient Descent Parameters "
        lrs[0] = lrs[1] if lrs[0] == 0 else lrs[0]

        lrs = alpha / lrs
        return lrs

    def generate_coefficients(self):
        """ Generates the coefficients used in function approximation"""

        coefficients = np.array(np.zeros(self.state_dim))  # Bias (all zeros)

        for combination_len in range(1, self.max_non_zero + 1):  # Upper bound for non zero combinations
            for index in combinations(range(self.state_dim), combination_len):
                for c in product(range(1, self.order + 1), repeat=combination_len):
                    coefficient = np.zeros(self.state_dim)  # Baseline
                    coefficient[list(index)] = list(c)  # Update elements
                    coefficients = np.vstack((coefficients, coefficient))  # Combine

        return coefficients

    def bound_state_space(self):
        """ Modifies the upper and lower bounds for inf approximations, allows for scaling"""
        if self.value_clip is not None:
            self.state_space.low[1] = -self.value_clip
            self.state_space.low[3] = -self.value_clip
            self.state_space.high[1] = self.value_clip
            self.state_space.high[3] = self.value_clip

    def get_features(self, state):
        """ Scale the state space values (based on modified bounds if provided) and return features"""

        state = np.clip(state, self.state_space.low, self.state_space.high)  # Clipping values within new bounds
        state = (state - self.state_space.low) / (self.state_space.high - self.state_space.low)  # Normalizing

        return np.cos(np.dot(np.pi * self.coefficients, state))


class PolicyGrad:
    def __init__(self, state_space, action_space,
                 basis='fourier', alpha=0.001, gamma=0.99,
                 fourier_order=3, value_clip=None, clip=None):

        self.state_space = state_space
        self.action_space = action_space
        self.action_dim = self.action_space.n
        self.basis = basis

        self.alpha = alpha
        self.gamma = gamma
        self.error_clip = clip

        if basis == 'fourier':
            self.basis = FourierBasis(self.state_space, self.action_space, fourier_order, value_clip=value_clip)
            self.learning_rate = self.basis.generate_learning_rates(self.alpha)

        self.num_basis = len(self.basis.coefficients)
        self.theta = {a: np.zeros(self.num_basis) for a in range(self.action_dim)}

    def softmax(self, state):
        """
        Returning the softmax of each action (un-normalized)
        :param state:
        :return: probability vector [P(action_0), P(action_1)]
        """

        # TODO - vectorize this!
        action_0 = np.dot(self.get_features(state), self.theta[0])
        action_1 = np.dot(self.get_features(state), self.theta[1])

        # Normalizing by maximum value
        probs = np.zeros(2)
        probs[0] = np.exp(action_0 - max(action_1, action_0))
        probs[1] = np.exp(action_1 - max(action_1, action_0))

        return probs

    def policy(self, state) -> np.array:
        """Policy: probability distribution of actions in given state"""

        probs = self.softmax(state)
        return probs / np.sum(probs)

    def get_features(self, state):
        return self.basis.get_features(state)

    def cum_rewards(self, t, rewards):
        """Cumulative reward function"""

        total = 0
        cum_array = []
        for tau in range(t, len(rewards)):
            total += self.gamma ** (tau - t) * rewards[tau]
            cum_array.append(total)
        return total, cum_array

    def update_action_probabilities(self,
                                    state_trajectory: list,
                                    action_trajectory: list,
                                    reward_trajectory: list,
                                    probs_trajectory: list) -> np.array:

        for t in range(len(reward_trajectory)):

            state = state_trajectory[t]
            action = action_trajectory[t]
            cum_reward, cum_reward_array = self.cum_rewards(t, reward_trajectory)

            # Determine action probabilities with policy
            action_probs = probs_trajectory[t]
            baseline_reward = cum_reward # * self.gamma ** t

            # TODO - Vectorized this mess
            if action == 1:
                self.theta[1] += self.alpha * baseline_reward * (1 - self.get_features(state) * action_probs[1])
                self.theta[0] += self.alpha * baseline_reward * (- self.get_features(state) * action_probs[0])

            elif action == 0:
                self.theta[0] += self.alpha * baseline_reward * (1 - self.get_features(state) * action_probs[0])
                self.theta[1] += self.alpha * baseline_reward * (- self.get_features(state) * action_probs[1])


def moving_average(interval, window_size):
    """ Helper function for averaging over different windows"""

    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    averaged = np.convolve(interval, window, 'same')
    return np.append(np.repeat(np.nan, window_size), averaged[int(window_size / 2):-int(window_size / 2)])


def training_instance(gym_env, episodes, alpha_initial, alpha_end, order, gamma, alpha_ratio=1,
                      max_steps=500, solved_threshold=195,
                      clip_val=None, value_clip=None, track_metrics=False, seed=0):
    """
    Run a singular instance of training (eg multiple episodes until environment is solved, or max episodes reached)
    """

    env = gym.make(gym_env)
    env.seed(seed)
    episode_rewards = []
    run_info = {'alpha': [], 'epsilon': [], 'learning_rates': []}

    agent = PolicyGrad(env.observation_space, env.action_space,
                       alpha=alpha_initial,
                       fourier_order=order,
                       gamma=gamma,
                       clip=clip_val,
                       value_clip=value_clip)

    start = time.perf_counter()
    for episode in range(episodes):

        alpha_mod = max(alpha_end, alpha_initial - alpha_initial * (episode / (alpha_ratio * episodes)))
        agent.learning_rate = agent.basis.generate_learning_rates(alpha_mod)

        if track_metrics:
            run_info['alpha'].append(alpha_mod)
            run_info['learning_rates'].append(agent.learning_rate)

        net_reward = 0
        state = env.reset()

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        for step in range(max_steps):


            action_probs = agent.policy(state)
            action = np.random.choice(2, p=np.squeeze(action_probs))

            state, reward, done, info = env.step(action)

            net_reward += reward

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)

            if done:
                break

        agent.update_action_probabilities(state_trajectory, action_trajectory,
                                          reward_trajectory, probs_trajectory)

        episode_rewards.append(net_reward)
        print("Episode: {0}, Avg Reward:{1:.2f}".format(episode, np.average(episode_rewards[-10:])))

        if np.average(episode_rewards[-100:]) >= solved_threshold and episode >= 100:
            end = time.perf_counter()
            print(f"Downloaded the tutorial in {end - start:0.4f} seconds")
            print("solved")
            break

    env.close()

    return episode_rewards


def run():
    """ Run a singular training instance to completion (until environment is solved or max episodes reached)"""

    gym_env = "CartPole-v0"

    alpha = 0.0005
    gamma = 0.99
    seed = 10
    fourier_order = 6
    np.random.seed(seed)

    quick_solves = [0, 1, 2, 4, 5, 6, 8, 9]  # seeds able to be solved <200 episodes
    slow_solves = [3, 7]  # seeds solved in ~300 and ~3000 episodes
    all_solves = slow_solves + quick_solves

    demonstration_seeds = [0]

    for seed_num in demonstration_seeds:
        np.random.seed(seed_num)

        episode_rewards = training_instance(gym_env=gym_env,
                                            episodes=140,
                                            alpha_initial=alpha,
                                            alpha_end=alpha,
                                            order=fourier_order,
                                            gamma=gamma,
                                            value_clip=1.111,
                                            solved_threshold=195,
                                            track_metrics=True,
                                            seed=seed_num)

        plt.plot(episode_rewards)

    #plt.suptitle("{}, {} runs\nPolicy gradient with Fourier basis function approximation".format(gym_env, 10))
    plt.title("REINFORCE with Fourier basis, α = {}, γ = {}, Fourier order = {}".format(alpha, gamma, fourier_order))
    plt.ylabel("Net reward")
    plt.xlabel("Episode")
    plt.grid()
    plt.show()


def main():
    run()


if __name__ == '__main__':
    main()
