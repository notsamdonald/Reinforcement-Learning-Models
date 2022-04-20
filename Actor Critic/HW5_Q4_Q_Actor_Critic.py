"""
RL HW5 Q4

Author: Sam Donald
Date: 16/11/2021

This script uses the Q-Actor Critic policy gradient algorithm to attempt to solve the Cliff Walking environment.

The algorithm is able to solve the environment, the the resultant policy taking on average 15.7 steps to reach the
terminal node. This policy however is suboptimal, with the agent favouring a safe path away from the cliff edge.

TODO - Tidy up!
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # Spooky - turn this off when debugging!


class QLearning:
    def __init__(self, state_space, action_space, basis='fourier', alpha=0.001,
                 gamma=0.99, epsilon=0, fourier_order=3, value_clip=None):
        self.state_space = state_space
        self.action_space = action_space
        self.action_dim = self.action_space.n
        self.basis = basis

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.num_basis = 48
        self.theta = np.zeros((48, 4))

    def get_features(self, state):
        return self.theta[state]

    def get_q(self, phi, action):
        return np.dot(self.theta[action], phi)

    def update(self, state_cur, action, reward, state_next, action_next):
        """Q-learning learning step"""

        q_cur = self.theta[state_cur][action]
        q_next = self.theta[state_next][action_next]

        td_error = reward + self.gamma * q_next - q_cur

        self.theta[state_cur][action] += self.alpha * td_error


class PolicyGrad:
    def __init__(self, state_space, action_space, basis='direct', alpha=0.001,
                 gamma=0.99):

        self.state_space = state_space
        self.action_space = action_space
        self.action_dim = self.action_space.n
        self.basis = basis

        self.alpha = alpha
        self.gamma = gamma

        if basis == 'direct':
            self.theta = np.zeros([self.state_space.n, self.action_space.n])

    def softmax(self, state):
        """ Softmax function, offsetting by max value to deal with large exponentials"""

        return np.exp(self.theta[state] - np.max(self.theta[state]))

    def cumulative_rewards(self, t, rewards):
        """Calculate cumulative reward for given trajectory"""

        total = 0
        cumulative_array = []
        for tau in range(t, len(rewards)):
            total += self.gamma ** (tau - t) * rewards[tau]
            cumulative_array.append(total)
        return total

    def policy(self, state):
        """Probability distribution of actions in given state"""

        probs = self.softmax(state)
        return probs / np.sum(probs)

    def encode_vector(self, index, dim):
        """One hot encode vector"""
        vector_encoded = np.zeros(dim)
        vector_encoded[index] = 1

        return vector_encoded

    def update(self, state_array, action_array, reward_array, critic):
        """Update the policy for the current trajectory"""

        for t in range(len(state_array)):

            action = action_array[t]
            state = state_array[t]
            action_probs = self.policy(state)

            baseline_reward = critic.theta[state, action]

            one_hot = self.encode_vector(action, 4)
            self.theta[state] += self.alpha * baseline_reward * (one_hot - action_probs) * self.gamma ** t


def training_instance(gym_env, episodes, alpha,
                      gamma, max_steps=500, value_clip=None,
                      seed=None, generate_heatmap=False):
    """
    Run a singular instance of training (eg multiple episodes until environment is solved, or max episodes reached)
    """

    env = gym.make(gym_env)

    # Create the actor and critic
    agent = PolicyGrad(env.observation_space, env.action_space,
                       alpha=0.01,
                       gamma=gamma,
                       basis='direct')

    critic = QLearning(env.observation_space, env.action_space,
                       alpha=0.05,
                       gamma=gamma)

    episode_rewards = []

    for episode in range(episodes):

        net_reward = 0
        state = env.reset()

        vistied = [0] * 48
        state_array, action_array, reward_array = np.zeros((3, max_steps), dtype=np.int16)

        action_probs = agent.policy(state)
        action = np.random.choice(agent.action_dim, p=np.squeeze(action_probs))

        for step in range(max_steps):

            state_next, reward, done, info = env.step(action)

            action_probs_next = agent.policy(state_next)
            action_next = np.random.choice(agent.action_dim, p=np.squeeze(action_probs_next))

            critic.update(state, action, reward, state_next, action_next)

            state_array[step] = state
            action_array[step] = action
            reward_array[step] = reward

            vistied[state] += 1

            net_reward += reward
            state = state_next
            action = action_next

            if done:
                vistied[state_next] += 1
                break

        agent.update(state_array, action_array, reward_array, critic)

        episode_rewards.append(net_reward)
        if episode % 10 == 0:
            print("Episode: {0}, Avg Reward:{1:.2f}".format(episode, np.average(episode_rewards[-10:])))

    if generate_heatmap:

        visited = [0] * 48
        for sample_run in range(1000):
            state = env.reset()
            for step in range(200):
                action_probs = agent.policy(state)
                action = np.random.choice(agent.action_dim, p=np.squeeze(action_probs))
                state_next, reward, done, info = env.step(action)
                visited[state] += 1
                state = state_next
                if done:
                    visited[state] += 1
                    break
        heatmap2d(visited)

    env.close()

    return episode_rewards


def heatmap2d(arr):
    arr = np.flip(np.array(arr).reshape(4, 12), axis=0)

    plt.imshow(arr, cmap='YlOrRd_r')
    plt.colorbar()
    for (j, i), label in np.ndenumerate(arr):
        plt.text(i, j, label, ha='center', va='center')
        plt.text(i, j, label, ha='center', va='center')

    plt.axis('off')
    plt.title("Q-Actor Critic Cliff Walking heatmap for final policy (1000 sample trajectories)\n")

    plt.show()


def moving_average(data):
    """Moving average helper function"""

    avg_data = []
    for index, data_point in enumerate(data):
        if index == 0:
            avg_data.append(data[0])
        elif index < 100:
            avg_data.append(np.average(data[0:index + 1]))
        else:
            avg_data.append(np.average(data[index - 100:index + 1]))
    return avg_data


def display_grid(state_array):
    """
    Formats the array into a 4x12 grid (used for visual interpretation)
    :param state_array: any list with 48 elements (eg current policy, count that each state is visited)
    """

    row_count = 4
    col_count = 12
    for i in range(row_count):

        row_to_print = state_array[(row_count - i - 1) * col_count:(row_count - i - 1) * col_count + col_count]
        to_print = ""
        for element in row_to_print:
            try:
                to_print += "{:>10.2f}".format(element)
            except:
                to_print += "{:>10}".format(element)  # Work around for printing non numeric values

        print(to_print)


def generate_plots(data, gym_env, alpha, gamma, fourier_order):
    """ Simple plot helper function"""

    for run in data:
        plt.plot(run, color="0.5")

    for run in data:
        plt.plot(moving_average(run), color="r")
        plt.scatter(len(run) - 1, moving_average(run)[-1], s=50, color='red', zorder=2)

    plt.suptitle("Q-Actor Critic Cliff Walking")
    plt.title("α actor = 0.01, α critic = 0.05 γ = {}".format(gamma))
    plt.ylabel("Net reward")
    plt.xlabel("Episode")
    plt.axhline(y=-13, color='g', linestyle='--')

    plt.grid()
    plt.show()


def run_individual():
    """ Run training instances to completion (until environment is solved or max episodes reached)"""

    gym_env = 'gym_cliffwalking:cliffwalking-v0'

    # Hyperparameter
    alpha = 0.01  # learning rate
    gamma = 0.99
    max_episodes = 2000
    run_nums = 1
    runs = []

    print("Starting {} run{}".format(run_nums, "s" if run_nums > 1 else ""))

    for run in range(run_nums):
        np.random.seed(0)

        episode_rewards = training_instance(gym_env=gym_env,
                                            episodes=max_episodes,
                                            alpha=alpha,
                                            gamma=gamma,
                                            seed=0,
                                            max_steps=200,
                                            generate_heatmap=False)

        runs.append(episode_rewards)

    generate_plots(runs, gym_env, alpha, gamma, 0)


def main():
    run_individual()


if __name__ == '__main__':
    main()
