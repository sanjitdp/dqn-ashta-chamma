import torch
from dqn import DQN
import numpy as np


def model_policy(env):
    LOAD_PATH = "./models/model-v17"
    ALWAYS_CAPTURE = True

    n_actions = env.action_space.n
    n_observations = len(env.to_array(env.observation))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy_net = DQN(n_observations, n_actions).to(device)
    checkpoint = torch.load(LOAD_PATH)

    policy_net.load_state_dict(checkpoint["model_state_dict"])

    # don't keep track of gradients
    with torch.no_grad():
        # convert observation to ndarray
        observation_array = env.to_array(env.observation)

        # tensorize the next state
        observation = torch.tensor(
            observation_array, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # compute output action
        output = policy_net(observation)

        # compute penalties
        penalty = torch.zeros(n_actions, device=device)
        for i in range(n_actions):
            # penalize an illegal move with -infinity reward
            if env.is_illegal_move(i):
                penalty[i] -= np.inf

            # if always_capture, penalize moves that don't lead to capture
            elif ALWAYS_CAPTURE and not env.is_capture_move(i):
                penalty[i] -= 5

        # return move with the maximum expected reward
        move = (output + penalty).max(1)[1].view(1, 1)

        # if no best move was found (i.e., all moves are illegal), just make a random move
        if move.item() == -1:
            move = torch.tensor(
                [[env.action_space.sample()]], device=device, dtype=torch.long
            )

        return move.item()
