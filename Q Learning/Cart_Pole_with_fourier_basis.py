"""
RL HW4 Q5 - Sam Donald

Author: Sam Donald
Date: 26/10/2021

This script uses Q-learning with function approximation using fourier basis as outlined here:
https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf

This is applied to the OpenAI CartPole-v0 environment (and can also be applied to CartPole-v1)

On average the environment is solved (195 point average over 100 consecutive runs) within ~130 runs with the
current hyper parameters, or ~1.7s

Epsilon and alpha are decayed throughout the runs. Additionally the state space values for cart velocity and angular
velocity normalized using bounds manually identified within training runs - as opposed to the min/max values listed
by OpenAI which are infinite. These additions significantly reduce the time to solve the environment,
 along with improving its stability.

"""

import gym
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations, product
from matplotlib import animation
from PIL import Image
import PIL.ImageDraw as ImageDraw


class FourierBasis:
    """Basis used for function approximation """

    def __init__(self, state_space, action_space, order, value_clip, max_non_zero=3):

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


class QLearning:
    def __init__(self, state_space, action_space, basis='fourier', alpha=0.001,
                 gamma=0.99, epsilon=0.05, fourier_order=3, value_clip=None, clip=None):

        self.state_space = state_space
        self.action_space = action_space
        self.action_dim = self.action_space.n
        self.basis = basis

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.error_clip = clip

        if basis == 'fourier':
            self.basis = FourierBasis(self.state_space, self.action_space, fourier_order, value_clip=value_clip)
            self.learning_rate = self.basis.generate_learning_rates(self.alpha)

        self.num_basis = len(self.basis.coefficients)
        self.theta = {a: np.zeros(self.num_basis) for a in range(self.action_dim)}

    def update(self, state_cur, action, reward, state_next):
        """Q-learning learning step"""

        phi_cur = self.get_features(state_cur)
        phi_next = self.get_features(state_next)

        q_cur = self.get_q(phi_cur, action)
        q_next = self.get_q(phi_next, self.get_action(phi_next))

        td_error = self.clip_td_error(reward + self.gamma * q_next - q_cur)

        self.theta[action] += self.learning_rate * td_error * phi_cur

    def get_features(self, state):
        return self.basis.get_features(state)

    def get_action(self, phi, epsilon_disable=False):
        """Epsilon greedy"""

        if np.random.rand() < self.epsilon and not epsilon_disable:
            return self.action_space.sample()
        else:
            q = [self.get_q(phi, action) for action in range(self.action_dim)]
            return q.index(max(q))

    def get_q(self, phi, action):
        return np.dot(self.theta[action], phi)

    def act(self, state):
        features = self.get_features(state)
        return self.get_action(features)

    def clip_td_error(self, td_error):
        """ Bounding the td error as it is known to be unstable during Q learning"""
        # Unused currently - values need to be tuned

        if self.error_clip is not None:  # TODO - this is symmetric, could be improved + janky code
            if td_error > self.error_clip:
                td_error = self.error_clip
            if td_error < -self.error_clip:
                td_error = -self.error_clip

        return td_error


def training_instance(gym_env, episodes, epsilon_initial, epsilon_ratio, alpha_initial, alpha_end, alpha_ratio, order,
                      gamma, max_steps=500, solved_threshold=195, clip_val=None, value_clip=None, track_metrics=False,
                      render_gif=False):
    """
    Run a singular instance of training (eg multiple episodes until environment is solved, or max episodes reached)

    :param gym_env: OpenAI gym env name
    :param episodes: maximum number of episodes to run
    :param epsilon_initial: initial value of epsilon
    :param epsilon_ratio: ratio within run for which epsilon will decay to zero (eg 0.5, epsilon = 0 half way to max ep)
    :param alpha_initial: initial value of alpha
    :param alpha_end: final value of alpha
    :param alpha_ratio: ratio within run for which alpha will decay to alpha_end (linearly)
    :param order: fourier basis order for value approximation
    :param clip_val: value used to clip TD error
    :param gamma: discount
    :param max_steps: maximum steps per episode
    :param solved_threshold: threshold for which the environment is solved (based on average over 100 most recent runs)
    :param value_clip: value used to clip infinite state space values (improves ability to normalize state space values)
    :param track_metrics: flag to gather metrics (epsilon, alpha, learning rates)
    :param render_gif: flag to determine if gif of training is rendered (slow)

    :return: episode_rewards: list of reward at each given episode
    :return: state_stack: matrix of state space values (used to determine value_clip threshold)
    :return: episode: episode number for which the environment was solved (or max episode if not)
    """

    env = gym.make(gym_env)
    _disable_view_window()

    episode_rewards = []
    run_info = {'alpha': [], 'epsilon': [], 'learning_rates': []}

    agent = QLearning(env.observation_space, env.action_space,
                      alpha=alpha_initial,
                      fourier_order=order,
                      gamma=gamma,
                      epsilon=epsilon_initial,
                      clip=clip_val,
                      value_clip=value_clip)

    state_stack = np.array(np.zeros(4))

    episode = 0
    avg_reward = 0
    frames = []

    for episode in range(episodes):

        # Update epsilon, alpha and associated learning rates for each episode
        agent.epsilon = max(epsilon_initial - epsilon_initial * (episode / (epsilon_ratio * episodes)), 0)
        alpha_mod = max(alpha_end, alpha_initial - alpha_initial * (episode / (alpha_ratio * episodes)))
        agent.learning_rate = agent.basis.generate_learning_rates(alpha_mod)

        if track_metrics:
            run_info['alpha'].append(alpha_mod)
            run_info['epsilon'].append(agent.epsilon)
            run_info['learning_rates'].append(agent.learning_rate)

        net_reward = 0
        state = env.reset()

        if render_gif:
            if episode > 100:
                avg_reward = np.nanmax(moving_average(episode_rewards, 100))
            else:
                avg_reward = np.nanmax(np.average(episode_rewards))

        for step in range(max_steps):

            action = agent.act(state)
            state_next, reward, done, info = env.step(action)

            state_stack = np.vstack((state_stack, state_next))

            net_reward += reward

            agent.update(state, action, reward, state_next)

            state = state_next

            if render_gif:
                frames.append(_add_label(env.render(mode="rgb_array"),
                                         episode_num=episode,
                                         reward=net_reward,
                                         avg_reward=int(avg_reward)))

            if done:
                episode_rewards.append(net_reward)

                if episode >= 100:
                    average_100 = np.nanmax(moving_average(episode_rewards, 100))
                    if average_100 >= solved_threshold:
                        env.close()
                        run_info['Done'] = True  # Tidy (quick addition)
                        return episode_rewards, state_stack, episode, run_info, frames

                break

    env.close()
    run_info['Done'] = False  # Tidy (quick addition)

    return episode_rewards, state_stack, episode, run_info, frames


def moving_average(interval, window_size):
    """ Helper function for averaging over different windows"""

    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    averaged = np.convolve(interval, window, 'same')
    return np.append(np.repeat(np.nan, window_size), averaged[int(window_size / 2):-int(window_size / 2)])


def plot_rewards(episode_rewards, environment="CartPole-v0", multi=False, run_info=None):
    """
    Helper plotter function

    :param episode_rewards: list of episode rewards (or list of lists containing episode rewards if multi)
    :param environment: environment name
    :param multi: flag determining if data for multiple runs are provided
    :param run_info: additional info (alpha rates, epsilon) to be used in plotting (only for single atm)
    """

    if multi:
        fig, ax = plt.subplots()

        ax.set_ylabel("Net reward")
        ax.set_xlabel("Episode")

        ax.set_title("Q-learning with value approximation using 3rd order Fourier basis", fontsize=12)
        fig.suptitle("{} ({} runs)".format(environment, len(episode_rewards)), fontsize=14)

        plot_successful = False
        plot_failed = False

        for run_num, run in enumerate(episode_rewards):

            if plot_successful is False and run_info[run_num] is True:
                ax.plot(moving_average(run, 1), label='Individual runs reward', color="0.5")
                plot_successful = True

            elif run_info[run_num]:
                ax.plot(moving_average(run, 1), color="0.5")

            elif run_info[run_num] is False and plot_failed is False:
                ax.plot(moving_average(run, 1), label='Individual failed runs reward', color="0.2")
                plot_failed = True

            else:
                ax.plot(moving_average(run, 1), color="0.2")

        df = pd.DataFrame(episode_rewards).fillna(200)
        df = df.drop(np.where(run_info is False)[0])  # Dropping runs that failed to solve (as messes with avg)
        ax.plot(df.mean(axis=0), label='Average run reward', color='red')
        plt.legend()

    else:

        fig, (ax1, ax2) = plt.subplots(2)

        ax1.set_ylabel("Net Reward")
        ax2.set_xlabel("Episode")

        ax1.set_title("Q-learning with value approximation using 3rd order Fourier basis", fontsize=12)

        fig.suptitle("{} (Individual run)".format(environment), fontsize=14)

        ax1.plot(moving_average(episode_rewards, 1), label='Sample run')
        ax1.grid(True)

        ax3 = ax2.twinx()

        p3, = ax3.plot(run_info['alpha'], color='red', label='alpha')
        p2, = ax2.plot(run_info['epsilon'], color='black', label='epsilon')

        ax2.set_ylabel("epsilon")
        ax3.set_ylabel("alpha")
        ax2.yaxis.label.set_color(p2.get_color())
        ax3.yaxis.label.set_color(p3.get_color())
        ax2.tick_params(axis='y', colors=p2.get_color())
        ax3.tick_params(axis='y', colors=p3.get_color())

        ax2.grid(True)

    plt.show()


def _plot_states(state_array):
    """ Plotting function to identify limits of state values (used to determine threshold)"""

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(state_array[:, 0])
    axs[0, 1].plot(state_array[:, 1])
    axs[1, 0].plot(state_array[:, 2])
    axs[1, 1].plot(state_array[:, 3])


def _save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    """ Post process frame array and save as a gif!"""
    plt.figure(figsize=(frames[0].size[1] / 50.0, frames[0].size[0] / 50.0), dpi=75)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, interval=40, repeat=False, blit=False)
    anim.save(path + filename, writer='imagemagick', fps=30)


def _disable_view_window():
    """ Such that frames can be saved, yet don't need to waste time displaying them"""

    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


def _add_label(frame, episode_num, reward=0, avg_reward=0):
    """ Add information to display (only viewable on output gif)"""
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0] / 20, im.size[1] / 18),
                'Episode: {0}\nEp Reward: {1:.0f}\nAvg Reward: {2:.2f}'.format(episode_num, reward, avg_reward),
                fill=text_color)

    return im


def _plot_time(time_array):
    """ Template code for histogram generation"""

    bins = [x * 0.1 for x in range(15, 25)]

    plt.hist(time_array, bins=bins, facecolor='blue', alpha=0.5, ec='black')

    # add a 'best fit' line
    plt.xlabel('Time to solve [s]')
    plt.ylabel('Number of runs')

    average_time = np.average(time_array)
    plt.title("Runtime to solve CartPole-v0 ({0} runs, avg={1:.2f}s)".format(len(time_array), average_time))
    plt.subplots_adjust(left=0.15)
    print("Average time to solve: {0:.2f}\n".format(average_time))
    plt.show()


def run_multi(number_of_runs, time_runs=False):
    """ Complete multiple runs to completion (solving the environment or max episode) and average across them"""

    print("Starting {} runs".format(number_of_runs))
    gym_env = "CartPole-v0"

    episode_array = []
    episode_rewards_array = []
    completed_array = []
    time_array = []

    for sample in range(number_of_runs):
        start = time.time()
        episode_rewards, state_stack, episode_to_complete, run_info, frames = training_instance(gym_env=gym_env,
                                                                                                episodes=500,
                                                                                                epsilon_initial=0.25,
                                                                                                epsilon_ratio=0.1,
                                                                                                alpha_initial=0.05,
                                                                                                alpha_end=0.01,
                                                                                                alpha_ratio=1,
                                                                                                order=3,
                                                                                                gamma=0.9,
                                                                                                value_clip=5,
                                                                                                solved_threshold=195,
                                                                                                track_metrics=True)
        end = time.time()
        if run_info["Done"]:
            print("Run {} completed in {} episodes".format(sample, episode_to_complete))
        else:
            print("Run {} failed to solve after {} episodes".format(sample, episode_to_complete))

        episode_rewards_array.append(episode_rewards)
        episode_array.append(episode_to_complete)
        completed_array.append(run_info["Done"])
        time_array.append(end - start)

    print("Average episodes: {}".format(np.average(episode_array)))

    if time_runs:
        _plot_time(time_array)
    plot_rewards(episode_rewards_array, multi=True, run_info=completed_array)

    # plot_rewards(episode_rewards_array[0], multi=False)


def run_individual():
    """ Run a singular training instance to completion (until environment is solved or max episodes reached)"""

    print("Starting individual run")

    gym_env = "CartPole-v0"
    episode_rewards, state_stack, episode_to_complete, run_info, frames = training_instance(gym_env=gym_env,
                                                                                            episodes=500,
                                                                                            epsilon_initial=0.25,
                                                                                            epsilon_ratio=0.1,
                                                                                            alpha_initial=0.05,
                                                                                            alpha_end=0.01,
                                                                                            alpha_ratio=1,
                                                                                            order=3,
                                                                                            gamma=0.9,
                                                                                            value_clip=5,
                                                                                            solved_threshold=195,
                                                                                            track_metrics=True)
    # _save_frames_as_gif(frames)

    if run_info["Done"]:
        print("Run completed in {} episodes\n".format(episode_to_complete))
    else:
        print("Run failed to solve after {} episodes\n".format(episode_to_complete))

    plot_rewards(episode_rewards, run_info=run_info)


def main():
    run_individual()
    run_multi(number_of_runs=100, time_runs=True)


if __name__ == '__main__':
    main()
