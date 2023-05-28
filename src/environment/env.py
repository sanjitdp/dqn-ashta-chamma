import numpy as np
import gymnasium as gym
from gymnasium import spaces


# set relevant constants
NUM_PIECES = 4
PLAYER_ONE = "O"
PLAYER_TWO = "X"

# define move paths for each player in the 5x5 array
MOVE_PATH = {
    PLAYER_ONE: [
        (4, 2),
        (4, 3),
        (4, 4),
        (3, 4),
        (2, 4),
        (1, 4),
        (0, 4),
        (0, 3),
        (0, 2),
        (0, 1),
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (3, 1),
        (2, 1),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (3, 2),
        (2, 2),
    ],
    PLAYER_TWO: [
        (0, 2),
        (0, 1),
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
        (3, 4),
        (2, 4),
        (1, 4),
        (0, 4),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (3, 2),
        (3, 1),
        (2, 1),
        (1, 1),
        (1, 2),
        (2, 2),
    ]
}

# array containing safe squares
SAFE_SQUARES = [(0, 2), (1, 1), (1, 3), (2, 0), (2, 2), (2, 4), (3, 1), (3, 3), (4, 2)]

# distances of possible moves
MOVES = [1, 2, 3, 4, 8]


class AshtaChammaEnv(gym.Env):
    def __init__(self):
        # observation space consists of two BOARD_SIZE x BOARD_SIZE arrays, each between 0 and NUM_PIECES
        # (one corresponding to PLAYER_ONE, one corresponding to PLAYER_TWO)
        self.observation_space = spaces.Dict(
            {
                PLAYER_ONE: spaces.Box(low=0, high=NUM_PIECES, shape=(5, 5)),
                PLAYER_TWO: spaces.Box(low=0, high=NUM_PIECES, shape=(5, 5)),
            }
        )

        # can choose which piece to move at each turn
        self.action_space = spaces.Tuple(
            (spaces.Discrete(NUM_PIECES), spaces.Discrete(len(MOVES)))
        )

    def reset(self):
        """
        resets board to its default state after each episode
        """

        # clean board to default
        self.observation = {PLAYER_ONE: np.zeros((5, 5), dtype=int), PLAYER_TWO: np.zeros((5, 5), dtype=int)}
        self.observation[PLAYER_ONE][4, 2] = NUM_PIECES
        self.observation[PLAYER_TWO][0, 2] = NUM_PIECES

        # reset player
        self.player = PLAYER_ONE

        return self.observation, self.player

    def step(self, action: np.ndarray):
        """
        handles one step given an action
        inputs: action - tuple (piece no. in order, move id)
            --> number of moves for move id:
                    0: 1,
                    1: 2,
                    2: 3,
                    3: 4,
                    4: 8
        outputs: observation - dict of ndarrays representing board,
                 reward - +1 if P1 wins, -1 if P2 wins, 0 otherwise,
                 done - boolean representing the win condition

        notes:
         - does nothing if player tries to move onto own piece
        """

        # ensure that action is valid
        assert self.action_space.contains(action)

        # get position array
        pos_array = MOVE_PATH[self.player]

        # unpack piece number and number of spaces to move
        piece_number, number_moves = action

        # figure out how many spaces to move
        number_moves = MOVES[number_moves]

        # get piece position indices
        piece_pos_indices = self.__get_piece_pos_indices()

        # get current position index
        curr_pos_idx = piece_pos_indices[piece_number]

        # if move is out of bounds, don't change the board
        if curr_pos_idx + number_moves >= 25:
            self.__switch_player()
            return self.observation, 0, False

        # get current position and position to move
        curr_pos = pos_array[curr_pos_idx]
        pos_to_move = pos_array[curr_pos_idx + number_moves]

        if pos_to_move not in SAFE_SQUARES:
            # check if we're moving onto one of our own pieces
            if self.observation[self.player][pos_to_move] > 0:
                return (
                    self.observation,
                    0,
                    False,
                )

            # capture the opponent's piece, if relevant
            elif self.observation[self.__other_player()][pos_to_move] > 0:
                self.observation[self.__other_player()][pos_to_move] -= 1
                self.observation[self.__other_player()][
                    MOVE_PATH[self.__other_player()][0]
                ] += 1

        # move piece to correct spot
        self.observation[self.player][curr_pos] -= 1
        self.observation[self.player][pos_to_move] += 1

        # check for win condition (otherwise, return default values)
        if self.observation[self.player][2, 2] == NUM_PIECES:
            return (
                self.observation,
                self.__get_reward(self.player),
                True,
            )
        else:
            self.__switch_player()
            return self.observation, 0, False

    def __other_player(self):
        # get other player symbol
        if self.player == PLAYER_ONE:
            return PLAYER_TWO
        else:
            return PLAYER_ONE

    def __get_piece_pos_indices(self):
        piece_indices = []
        pos_array = MOVE_PATH[self.player]

        # get piece positions of current player in order of how far along they are
        # (from nearest to farthest)
        for pos_idx, _ in enumerate(pos_array):
            pieces_here = self.observation[self.player][pos_array[pos_idx]]
            while pieces_here > 0:
                piece_indices.append(pos_idx)
                pieces_here -= 1

        return piece_indices

    @staticmethod
    def __get_reward(player):
        # map player symbol to their reward, if they win
        if player == PLAYER_ONE:
            return 1
        elif player == PLAYER_TWO:
            return -1

    def render(self):
        """
        renders the board onto the console
        """
        observation = self.observation

        # print top line
        print("-" * 31)
        state = np.vstack((observation[PLAYER_ONE], observation[PLAYER_TWO]))

        for row1, row2 in zip(state, state[int(len(state)/2):]):
            # print bar before each square
            print("| ", end="")

            # print player one piece count
            for square_1 in row1:
                if square_1 != 0:
                    print(PLAYER_ONE, square_1, "| ", end="")
                else:
                    print("    | ", end="")

            # separate player one and player two piece count
            print()

            # print bar after each square
            print("| ", end="")

            # print player two piece count
            for square_2 in row2:
                if square_2 != 0:
                    print(PLAYER_TWO, square_2, "| ", end="")
                else:
                    print("    | ", end="")
            
            # print bottom line after each row
            print()
            print("-------------------------------")
    
    def __switch_player(self):
        self.player = self.__other_player()
