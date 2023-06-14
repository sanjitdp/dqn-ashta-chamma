# standard imports
from env import AshtaChammaEnv
from dqn import DQN
import math
from random import random, choice
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import os

# import helper objects
from replay_memory import ReplayMemory, Transition

# import opponent policies
from policies.random_policy import random_policy
from policies.safe_policy import safe_policy
from policies.offense_policy import offense_policy
from policies.smart_policy import smart_policy
from policies.fast_policy import fast_policy
from policies.slow_policy import slow_policy
from policies.model_policy import model_policy

# import pytorch modules
import torch
import torch.nn as nn
import torch.optim as optim

# import tqdm for loading bars
from tqdm import tqdm

# set opponent policy
opponent_policy = offense_policy

# initialize game environment
env = AshtaChammaEnv(opponent_policy)

# use metal performance shaders for acceleration if possible (apple silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# hyperparameters for the training loop:
# BATCH_SIZE is the number of transitions sampled from the replay buffer,
# GAMMA is the discount factor,
# EPS_START is the initial value of epsilon,
# EPS_END is the final value of epsilon,
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay,
# TAU is the update rate of the target network,
# LR is the learning rate of the optimizer,
# NUM_EPISODES is the number of episodes for training,
# MEMORY_SIZE is the size of the replay memory
BATCH_SIZE = 16
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 300
TAU = 0.005
LR = 5e-5
NUM_EPISODES = 500
MEMORY_SIZE = 64
LOAD_FROM_CHECKPOINT = False
LOAD_OPTIMIZER = True
SAVE_CHECKPOINT = True
LOAD_PATH = "./models/model-v6"
SAVE_PATH = "./models/model-v7"

# get number of actions from gym action space
n_actions = env.action_space.n

# reset environment
state, _ = env.reset()

# get the number of state observations
n_observations = len(env.to_array(state))

# set policy network and target network
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

# set optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# load from checkpoint if necessary
if LOAD_FROM_CHECKPOINT:
    # get training checkpoint
    checkpoint = torch.load(LOAD_PATH)

    policy_net.load_state_dict(checkpoint["model_state_dict"])

    if LOAD_OPTIMIZER:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# copy weights for target network
target_net.load_state_dict(policy_net.state_dict())

# initialize memory
memory = ReplayMemory(MEMORY_SIZE)

# count number of steps
steps_done = 0

# keep track of episode durations
episode_durations = []


def select_action(state, env, always_capture=False, always_safe=False):
    """
    maps state to action, taking into account randomness (epsilon) and
    """

    # keep track of steps done (in global scope)
    global steps_done

    # compute threshold for making random move
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )

    # increment step counter
    steps_done += 1

    # with probability 1 - eps_threshold,
    # make a reasoned move using the DQN policy network
    if random() > eps_threshold:
        # don't keep track of gradients
        with torch.no_grad():
            # compute output action
            output = policy_net(state)

            # compute penalties
            penalty = torch.zeros(n_actions, device=device)
            for i in range(n_actions):
                # penalize an illegal move with -infinity reward
                if env.is_illegal_move(i):
                    penalty[i] -= np.inf

                # if always_capture, penalize moves that don't lead to capture
                elif always_capture and not env.is_capture_move(i):
                    penalty[i] -= 7

                # # if always_safe, penalize moves that don't land on a safe square
                # elif always_safe and not env.moving_to_safe(i):
                #     penalty[i] -= 5

            # return move with the maximum expected reward
            move = (output + penalty).max(1)[1].view(1, 1)

            # if no best move was found (i.e., all moves are illegal), just make a random move
            if move.item() == -1:
                move = torch.tensor(
                    [env.action_space.sample()], device=device, dtype=torch.long
                )

    # otherwise, make a random move
    else:
        # compute all capturing moves
        capture_moves = [
            i
            for i in range(n_actions)
            if not env.is_illegal_move(i) and env.is_capture_move(i)
        ]

        # compute all safe moves
        safe_moves = [
            i
            for i in range(n_actions)
            if not env.is_illegal_move(i) and env.moving_to_safe(i)
        ]

        # compute all legal moves
        legal_moves = [i for i in range(n_actions) if not env.is_illegal_move(i)]

        # if always_capture, try to make a capturing move
        if always_capture and capture_moves:
            move = torch.tensor(
                [choice(capture_moves)],
                device=device,
                dtype=torch.long,
            )

        # if always_safe, try to make a safe move
        elif always_safe and safe_moves:
            move = torch.tensor(
                [choice(safe_moves)],
                device=device,
                dtype=torch.long,
            )

        # try to make any legal move
        elif legal_moves:
            move = torch.tensor(
                [choice(legal_moves)],
                device=device,
                dtype=torch.long,
            )

        # otherwise, it doesn't matter what move we make - the environment will handle it
        else:
            move = torch.tensor(
                [0],
                device=device,
                dtype=torch.long,
            )

    # return computed best move
    return move


def optimize():
    """
    computes an optimization step
    """

    # if memory has fewer elements than the batch size, then we can't optimize
    if len(memory) < BATCH_SIZE:
        return

    # randomly selects elements from history for batch (decouples dependencies due to exchangeability)
    transitions = memory.sample(BATCH_SIZE)

    # transpose the batch, convert batch-array of Transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))

    # compute a mask of non-final states (a final state is one after which simulation ends)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    # concatenate batch elements
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # tensorize state, action, and reward
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # compute Q(s_t, a) - the model computes Q(s_t), then we select
    # actions which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # expected values of actions for non_final_next_states are computed based on the "older" target_net
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # compute V(s_{t+1}) for all next states
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # select loss criterion
    criterion = nn.SmoothL1Loss()  # Huber loss

    # compute loss function
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # reset gradients
    optimizer.zero_grad()

    # compute backward pass
    loss.backward()

    # in-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    # compute a step from the optimizer
    optimizer.step()


def plot_windowed_average(reward_count, window_size):
    """
    plots windowed average win rate
    """

    # keep track of average win rate in the last window_size episodes
    avg_rewards = []
    for idx in range(window_size, len(reward_count)):
        avg_rewards.append(sum(reward_count[idx - window_size : idx]) / window_size)

    # plot windowed win rate
    plt.plot(range(window_size, len(reward_count)), avg_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Five-episode average reward (legal moves)")

    # show plot
    plt.show()


def plot_win_rate(reward_count):
    """
    plots the running win rate
    """

    # associate reward to win count
    win_count = [(reward + 1) / 2 if reward else 0 for reward in reward_count]

    # keep track of running win rates
    win_rates = []
    running_sum = 0
    for idx, win in enumerate(win_count):
        running_sum += win
        win_rates.append(running_sum / (idx + 1))

    # plot running win rates
    plt.plot(range(len(reward_count)), win_rates)
    plt.xlabel("Episodes")
    plt.ylabel("Win rate")
    plt.title("Episodes vs. win rate (legal moves)")
    plt.axhline(y=0.5, color="black", linestyle="dashed")

    # output final win rate
    print("Final win rate:", win_rates[-1])

    # show plot
    plt.show()


def validate():
    with torch.no_grad():
        rewards = []
        for _ in range(100):
            # initialize environment
            env = AshtaChammaEnv(opponent_policy=opponent_policy)
            observation, _ = env.reset()

            # simulate a game, selecting an action at each step
            done = False
            while not done:
                # env.render()
                # print()

                # convert observation to ndarray
                observation_array = env.to_array(observation)

                # tensorize the next state
                observation = torch.tensor(
                    observation_array, dtype=torch.float32, device=device
                ).unsqueeze(0)

                # compute new observation, reward, and whether the game is over
                move = select_action(
                    observation, env, always_capture=True, always_safe=True
                ).item()
                observation, reward, done = env.step(move)

            rewards.append(reward)

        return sum(rewards) / len(rewards)


def train():
    # keep track of rewards
    reward_count = np.zeros(NUM_EPISODES)

    # keep track of validation win rate
    # validation_rates = []

    # training loop
    for i_episode in tqdm(range(NUM_EPISODES)):
        # compute average validation rate
        # if i_episode % 10 == 0:
        #     validation_rates.append(validate())

        # initialize the environment and get its state
        state, _ = env.reset()

        # compute state array
        state_array = env.to_array(state)

        # tensorize state
        state = torch.tensor(state_array, dtype=torch.float32, device=device).unsqueeze(
            0
        )

        # training loop: equivalent to while True (with a counter t)
        for t in count():
            # compute action for current state
            action = select_action(state, env, always_capture=True, always_safe=True)

            # get observation and reward for making the computed action
            observation, reward, terminated = env.step(action.item())
            reward_count[i_episode] += reward
            reward = torch.tensor([reward], device=device)

            # figure out whether the game is over
            done = terminated

            # if game is over, there is no next state
            if terminated:
                next_state = None

            # otherwise, compute the next state based on the observation
            else:
                # convert observation to ndarray
                observation_array = env.to_array(observation)

                # tensorize the next state
                next_state = torch.tensor(
                    observation_array, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # store the transition in memory
            memory.push(state, action, next_state, reward)

            # move to the next state
            state = next_state

            # perform one optimizer step (on the policy network)
            optimize()

            # soft update the target network's weights:
            # θ' ← τ θ + (1 − τ) θ'
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            # break out of loop if the game is over
            if done:
                episode_durations.append(t + 1)
                break

    if SAVE_CHECKPOINT:
        # save model
        torch.save(
            {
                "model_state_dict": policy_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            SAVE_PATH,
        )

    # mark the end of training
    print("Done!")

    # create output plots
    plt.figure()
    plot_win_rate(reward_count)
    # plt.figure()
    # plt.plot(validation_rates)
    plt.show()


if __name__ == "__main__":
    train()
