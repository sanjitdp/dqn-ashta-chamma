import gymnasium as gym
from env import AshtaChammaEnv
from dqn import DQN
from replay_memory import ReplayMemory, Transition
from policies.random_policy import random_policy

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

opponent_policy = random_policy
env = AshtaChammaEnv(opponent_policy)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor
# EPS_START is the initial value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 500
TAU = 0.005
LR = 5e-5

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, _  = env.reset()
n_observations = len(np.concatenate((state[env.PLAYER_ONE], state[env.PLAYER_TWO], state[env.ROLL]), axis=None))

policy_net = DQN(n_observations, n_actions).to(device) # These network constructions may need to be altered a tiny bit depending on how we build our network.
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10) # Tentatively set to 1


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            output = policy_net(state)
            # get rid of illegal moves...
            penalty = torch.zeros(n_actions, device=device)
            for i in range(n_actions):
                if env.is_illegal_move(i):
                    penalty[i] -= np.inf
            move = (output + penalty).max(1)[1].view(1, 1) # t.max(1) will return the largest column value of each row. Second column on max result is index of where max element was found, so we pick action with the larger expected reward.
            if move.item() == -1:
                move = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    else:
        try:
            move = torch.tensor([random.choice([i for i in range(n_actions) if not env.is_illegal_move(i)])], device=device, dtype=torch.long)
        except:
            move = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
    return move

episode_durations = []


#def plot_durations(show_result=False):
#    plt.figure(1)
#    durations_t = torch.tensor(episode_durations, dtype=torch.float)
#    if show_result:
#        plt.title('Result')
#    else:
#        plt.clf()
#        plt.title('Training...')
#    plt.xlabel('Episode')
#    plt.ylabel('Duration')
#    plt.plot(durations_t.numpy())
#    # Take 100 episode averages and plot them too
#    if len(durations_t) >= 100:
#        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#        means = torch.cat((torch.zeros(99), means))
#        plt.plot(means.numpy())
#
#    plt.pause(0.001)  # pause a bit so that plots are updated
#    if is_ipython:
#        if not show_result:
#            display.display(plt.gcf())
#            display.clear_output(wait=True)
#        else:
#            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return # If memory has less elements than the batch, something is wrong and can't call next line
    transitions = memory.sample(BATCH_SIZE) # Randomly selects the elements of history for the batch, decouples dependenceis that otherwise would exist if sequential data were taken in this case
    # Transpose the batch (https://stackoverflow.com/a/19343/3343043).  Converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions)) # Don't fully understand this, but it seems to be required

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool) # Compute a mask of non-final states (a final state is one after which simulation ends)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # Concatenate the batch elements
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch) # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken for each batch state according to policy_net

    # Compute V(s_{t+1}) for all next states. Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch # the expected Q values

    criterion = nn.SmoothL1Loss() # Huber be like
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) # unsqueeze(1) to get tensor size correct

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # in-place gradient clipping
    # Maybe add on a clipping for output, this is sometimes done, could be sus though
    optimizer.step()

num_episodes = 10000

reward_count = np.zeros(num_episodes)
for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state_np = np.concatenate((state[env.PLAYER_ONE], state[env.PLAYER_TWO], state[env.ROLL]), axis=None)
    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated = env.step(action.item())
        reward_count[i_episode] += reward
        reward = torch.tensor([reward], device=device)
        done = terminated

        if terminated:
            next_state = None
        else:
            observation_np = np.concatenate((observation[env.PLAYER_ONE], observation[env.PLAYER_TWO], observation[env.ROLL]), axis=None)
            next_state = torch.tensor(observation_np, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break

def plot_windowed_average():
    avg_rewards = []
    for idx in range(5, len(reward_count)):
        avg_rewards.append(sum(reward_count[idx-5:idx]) / 5)
    plt.plot(range(5, len(reward_count)), avg_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Five-episode average reward (legal moves)")
    plt.show()

def plot_win_rate():
    win_count = [(reward + 1) / 2 for reward in reward_count]

    win_rates = []
    running_sum = 0
    for idx, win in enumerate(win_count):
        running_sum += win
        win_rates.append(running_sum / (idx + 1))

    plt.plot(range(len(reward_count)), win_rates)
    plt.xlabel("Episodes")
    plt.ylabel("Win rate")
    plt.title("Episodes vs. win rate (legal moves)")
    plt.axhline(y = 0.5, color = 'black', linestyle = 'dashed')

    print("Final win rate:", win_rates[-1])

    plt.show()

print('Complete')
#plot_windowed_average()
plot_win_rate()
plt.ioff()
plt.show()
