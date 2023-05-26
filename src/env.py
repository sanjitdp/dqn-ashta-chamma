import numpy as np
import gymnasium as gym
from gymnasium import spaces

# set relevant constants
BOARD_SIZE = 5
NUM_PIECES = 4
PLAYER_ONE = "O"
PLAYER_TWO = "X"


class AshtaChammaEnv(gym.Env):
    def __init__(self):
        # observation space consists of two BOARD_SIZE x BOARD_SIZE arrays, each between 0 and NUM_PIECES
        # (one corresponding to PLAYER_ONE, one corresponding to PLAYER_TWO)
        self.observation_space = spaces.Dict(
            {
                PLAYER_ONE: spaces.Box(
                    low=0, high=NUM_PIECES, shape=(BOARD_SIZE, BOARD_SIZE)
                ),
                PLAYER_TWO: spaces.Box(
                    low=0, high=NUM_PIECES, shape=(BOARD_SIZE, BOARD_SIZE)
                ),
            }
        )

        # can choose which piece to move at each turn
        self.action_space = spaces.Discrete(NUM_PIECES)

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
