import gym
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from tqdm import tqdm

def rand_argmax(b, **kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


def epsilon_greedy_policy(state, q_table, epsilon=0.01):
    if np.random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        action = rand_argmax(q_table[:, state])  # Choose the action with largest Q-value (state value)
    return action


def create_q_table(rows=5, cols=5):
    """ generate Q Table"""

    # initialize the q_table with all zeros for each state and action
    q_table = np.zeros((4, cols * rows))

    return q_table


def generate_policy(q_table):
    """
    Generates a greedy policy based on the current q_table (used for visualization)
    :param q_table: current instance of q_table
    :return: policy table (list comprising of directional arrows)
    """

    policy_table = [""] * 25
    symbol_dict = {3: "↑", 1: "↓", 0: "←", 2: "→"}
    # action_decode = {3: 'up', 0: 'right', 1: 'down', 2: 'left'}

    # Choose the action with largest Q-value (state value)
    for state_index, state in enumerate(policy_table):
        policy_table[state_index] = symbol_dict[int(rand_argmax(q_table[:, state_index]))]

    return policy_table


def display_grid(state_array):
    """
    Formats the array into a 4x12 grid (used for visual interpretation)
    :param state_array: any list with 48 elements (eg current policy, count that each state is visited)
    """

    output = np.array([symbol_dict[action] for action in np.argmax(state_array, axis=0)]).reshape(5, 5)
    print(output)



def get_max_state_value(state, q_table):
    """ Determine the maximum value state based on the current state and Q table"""

    state_action = q_table[:, int(state)]
    maximum_state_value = np.amax(state_action)  # return the state value with for the highest action
    return maximum_state_value


def Q_learning(env, alpha, gamma, episodes, epsilon_func, epsilon_override=None):
    """
    Complete a run of Q_learning
    :param name: OpenAI environment name
    :param alpha: learning rate
    :param gamma: decay rate
    :param episodes: episodes to run
    :param epsilon_func: function defining how epsilon is decayed
    :param epsilon_override: override for epsilon (if a constant value is to be used)
    :return: reward_cum_array (list of cumulative rewards for each episode)
    :return: policy - greedy policy based on final Q table (populated with arrows showing direction)
    """

    env.reset()
    q_table = create_q_table()
    reward_cum_array = []
    delta_array = []
    for episode in tqdm(range(episodes)):

        reward_cum = 0
        state = env.reset()

        for _ in range(100):

            epsilon = epsilon_override if epsilon_override is not None else epsilon_func(episode)

            action = epsilon_greedy_policy(state, q_table=q_table, epsilon=epsilon)

            # Take action, observe rewards
            new_state, reward, done, info = env.step(action)

            next_state_value = get_max_state_value(new_state, q_table=q_table)

            target = reward + gamma * next_state_value

            reward_cum += reward

            # Update Q table:
            updated_q_value = (1-alpha) * q_table[action, state] + alpha * target
            old_q_value = q_table[action, state]
            q_table[action, state] = updated_q_value

            state = new_state

            delta_array.append(abs(updated_q_value-old_q_value))
            if max(delta_array[-2000:]) < 1e-4 and len(delta_array) > 2000:
                print("Converged after {} updates, {} episode".format(len(delta_array), episode))
                env.close()

                #plt.plot(delta_array)
                #plt.show()

                return reward_cum_array, q_table  # generate_policy(q_table)

            # End conditions
            if done:
                # Final state reached
                break

        reward_cum_array.append(reward_cum)

    env.close()
    print("Failed to converge")
    return reward_cum_array, q_table #generate_policy(q_table)


def generate_plots(run_data, params, multi=False):
    fig, ax = plt.subplots()

    ax.set_ylabel("Net reward")
    ax.set_xlabel("Episode")

    if multi:
        fig.suptitle("cliffwalking-v0 ({} runs)".format(len(run_data)), fontsize=14)
    else:
        fig.suptitle("cliffwalking-v0 (Individual run)", fontsize=14)

    ax.set_title("Q-learning: α={}, γ={}, ε={} ".format(params['alpha'], params['gamma'], params['epsilon']),
                 fontsize=12)

    for run_num, run_ in enumerate(run_data):
        if run_num == 0:
            ax.plot(run_, color='0.5', label='Individual runs')
        else:
            ax.plot(run_, color='0.5')

    # Plot averages
    if multi:
        df = pd.DataFrame(run_data)
        ax.plot(df.mean(axis=0), label='Average run reward', color='red')
        plt.legend()

    plt.grid()
    plt.show()


def iteration_decay(episode):
    """Simple function to decay epsilon based on episode number"""

    if episode == 0:
        return 1
    else:

        episode_end = 2000
        start_rate = 0.1
        return max(((-start_rate/episode_end)*episode) + start_rate, 0)

        #return 1 / ((1 + episode) ** 2)


def run(run_count=1):

    map_layout = ['SFHFF', 'FHFFF', 'FFFHF', 'FFFFH', 'HFFFG']
    slippy = True

    if slippy:
        # env = gym.make('Stochastic-8x8-FrozenLake-v0')
        env = gym.make('FrozenLake8x8-v1', desc=map_layout, is_slippery=True)

    else:
        env = gym.make('FrozenLake8x8-v1', desc=map_layout, is_slippery=False)

    env.reset()
    alpha = 0.5
    gamma = 0.6

    reward_ts_list = []
    policy_table = []

    for _run in range(run_count):
        reward_ts, policy_table = Q_learning(env=env,
                                             alpha=alpha,
                                             gamma=gamma,
                                             episodes=5000,
                                             epsilon_func=iteration_decay,
                                             epsilon_override=None)

    print("Success rate over final 1000 episodes : {:.2f}%".format(100*np.average(reward_ts[-1000:])))
    print("Sample policy table:")
    display_grid(policy_table)

    plt.plot(running_mean(reward_ts, 250))
    plt.show()

symbol_dict = {3: "↑", 1: "↓", 0: "←", 2: "→"}


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def main():
    run(run_count=1)


if __name__ == "__main__":
    main()
