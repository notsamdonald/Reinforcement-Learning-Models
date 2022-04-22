import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    # From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.l1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.action_dim)
        self.act = nn.ReLU()
    def forward(self, x):

        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)

        return x






def optimize_model():
    # Taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch.to(device)).gather(1, action_batch.reshape(128,1).to(device))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.to(device)).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch.reshape(128)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def epsilon_greedy_policy(state, policy_net, epsilon=0.01):
    if np.random.random() < epsilon:
        action = random.randint(0, 2) # TODO - this should instead sample from the action space
    else:
        action = torch.argmax(policy_net(torch.tensor(state).to(device))).item() # FIXME
    return action

def rand_argmax(b, **kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)

def iteration_decay(episode):
    """Simple function to decay epsilon based on episode number"""

    if episode == 0:
        return 1
    else:

        episode_end = 100
        start_rate = 0.9
        min_eps = 0.05
        return max(((-start_rate/episode_end)*episode) + start_rate, min_eps)

        #return 1 / ((1 + episode) ** 2)

env = gym.make('Acrobot-v1')

batch_size = 128
gamma = 0.99
epsilon = 0.1  # TODO - make this decay
target_update = 5  # TODO - what is this?
TARGET_UPDATE = 5
epsilon_override = None # Used to define a constant ep value
epsilon_func = iteration_decay


n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

policy_net = DQN(n_states, 20, n_actions).to(device)
target_net = DQN(n_states, 20, n_actions).to(device)

# Not sure what this is doing (think it is setting the weights to be the same for both nets)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)

steps = 0
episode_durations = []

num_episodes = 200
for i_episode in range(num_episodes):
    # Initialize the environment and state

    state = env.reset()

    for t in count():
        # Select and perform an action

        epsilon = epsilon_override if epsilon_override is not None else epsilon_func(i_episode)

        action = epsilon_greedy_policy(state, policy_net, epsilon=epsilon)
        #env.render()
        new_state, reward, done, info = env.step(action)

        reward = torch.tensor([reward], device=device)

        # Observe new state
        if done:
            episode_durations.append(t + 1)
            break

        # Store the transition in memory
        memory.push(torch.tensor(state), torch.tensor(action), torch.tensor(new_state), torch.tensor(reward))

        # Move to the next state
        state = new_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        if done:
            episode_durations.append(t + 1)
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print("Episode:{}, Steps:{} Epsilon:{}".format(i_episode, t+1, epsilon))


plt.plot(episode_durations)
plt.show()

print('Complete')

#env.render()
#env.close()
#plt.ioff()
#plt.show()
