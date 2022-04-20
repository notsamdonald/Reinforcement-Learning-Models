import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import gym

def value_iteration(P, state_count, gamma, tol):
    """
    Performing value iteration based on the available actions and rewards denoted in P
    :param P: State transition dictionary, containing prob, s_next, r, done for every action in a given state
    :param state_count: number of states
    :param gamma: discount factor
    :param tol: tolerance to identify when iteration has converged
    :return: V: converged value function V(s)
    """

    V, pi = np.zeros(state_count), np.zeros(state_count, dtype=int)

    # Value iteration
    steps = 0
    while True:
        steps += 1
        delta = 0
        for state in P:
            actions = P[state]
            v_prior = V[state]
            action_values = []
            for action in actions:
                action_rewards = 0
                for result in actions[action]:
                    prob, s_next, r, done = result
                    action_rewards += prob * (r + gamma * V[s_next])
                action_values.append(action_rewards)

            v_new = max(action_values)
            V[state] = v_new
            delta = max(delta, abs(v_prior - v_new))

        if delta < tol:
            break

    # Extracting policy from converged value function
    # TODO - this could be tidied such that the above is put into a function
    for state in range(state_count):
        actions = P[state]
        action_values = []
        for action in actions:
            action_rewards = 0
            for result in actions[action]:
                prob, s_next, r, done = result
                action_rewards += prob * (r + gamma * V[s_next])
            action_values.append(action_rewards)
        action_max = np.argmax(action_values)
        pi[state] = action_max

    return V, pi, steps


def policy_iteration(P, state_count, gamma, tol):
    def policy_evaluation():

        # Policy evaluation:

        while True:
            delta = 0

            for state in range(state_count):
                V_prior = V[state]

                V_new = 0
                for action in range(action_count):
                    p_a = pi[state][action]  # probability that action is taken based on policy
                    for result in P[state][action]:
                        prob, s_next, r, done = result
                        V_new += p_a * prob * (r + gamma * V[s_next])

                V[state] = V_new
                delta = max(delta, abs(V_prior - V_new))

            if delta < tol:
                return V, pi

    def policy_update():
        # Extracting policy from converged value function
        # TODO - this could be tidied such that the above is put into a function
        policy_stable = True
        for state in range(state_count):
            actions = P[state]
            action_values = []
            old_action = pi[state]
            for action in actions:
                action_rewards = 0
                for result in actions[action]:
                    prob, s_next, r, done = result
                    action_rewards += prob * (r + gamma * V[s_next])
                action_values.append(action_rewards)

            new_action = one_hot(np.argmax(action_values), 4)
            if np.array_equal(old_action, new_action) == False:
                policy_stable = False
            pi[state] = new_action

        return policy_stable

    V, pi_star = np.zeros(state_count), np.zeros(state_count, dtype=int)
    pi_random = np.array([[0.25, 0.25, 0.25, 0.25]] * 25)
    action_count = 4
    pi = pi_random
    steps = 0

    while True:
        steps += 1
        V, pi = policy_evaluation()
        policy_stable = policy_update()
        if policy_stable:
            for state in range(state_count):
                pi_star[state] = np.argmax(pi[state])
            return V, pi_star, steps


def one_hot(id, length):
    one_hot_vector = []
    for i in range(length):
        if i == id:
            one_hot_vector.append(1)
        else:
            one_hot_vector.append(0)
    return np.array(one_hot_vector)


def pretty_print(V, pi, steps, name):
    print("----------------------------------------------------")
    print("\n{} iteration converged in {} steps\n".format(name, steps))

    print('Value Function:')
    print(V.reshape(5, 5))

    print("\nOptimal policy:")
    print("0:Left, 1:Down, 2:Right, 3:Up")

    print(pi.reshape(5, 5))
    print("----------------------------------------------------")


def main():

    map_layout = ['SFHFF', 'FHFFF', 'FFFHF', 'FFFFH', 'HFFFG']
    slippy = True

    if slippy:
        #env = gym.make('Stochastic-8x8-FrozenLake-v0')
        env = gym.make('FrozenLake8x8-v1', desc=map_layout, is_slippery=True)

    else:
        env = FrozenLakeEnv(desc=map_layout, is_slippery=False)

    env.reset()

    state_count = env.observation_space.n

    V_V, pi_V, steps_V = value_iteration(P=env.P, state_count=state_count, gamma=0.9, tol=1e-3)
    V_P, pi_P, steps_P = policy_iteration(P=env.P, state_count=state_count, gamma=0.9, tol=1e-3)

    # Displaying results
    pretty_print(V_V, pi_V, steps_V, "Value")
    pretty_print(V_P, pi_P, steps_P, "Policy")

if __name__ == "__main__":
    main()
print("")
