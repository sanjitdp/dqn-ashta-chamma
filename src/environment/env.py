import numpy as np
import gymnasium as gym
from util.shells import Shells
from gymnasium import spaces


class AshtaChammaEnv(gym.Env):
    # set relevant constants
    NUM_PIECES = 4
    PLAYER_ONE = "O"
    PLAYER_TWO = "X"
    ROLL = "ROLL"

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

    # distances of possible self.MOVES
    MOVES = [1, 2, 3, 4, 8]

    def __init__(self, opponent_policy):
        # observation space consists of two BOARD_SIZE x BOARD_SIZE arrays, each between 0 and self.NUM_PIECES
        # (one corresponding to self.PLAYER_ONE, one corresponding to self.PLAYER_TWO)
        self.observation_space = spaces.Dict(
            {
                self.PLAYER_ONE: spaces.Box(low=0, high=self.NUM_PIECES, shape=(5, 5)),
                self.PLAYER_TWO: spaces.Box(low=0, high=self.NUM_PIECES, shape=(5, 5)),
                self.ROLL: spaces.Discrete(len(self.MOVES))
            }
        )

        # keep track of opponent's policy (which must always output a legal move)
        self.opponent_policy = opponent_policy

        # can choose which piece to move at each turn
        self.action_space = spaces.Discrete(self.NUM_PIECES)

    def reset(self):
        """
        resets board to its default state after each episode
        """

        # create shells object
        self.__shells = Shells()
        self.__shells.roll()

        # clean board to default
        self.observation = {self.PLAYER_ONE: np.zeros((5, 5), dtype=int), self.PLAYER_TWO: np.zeros((5, 5), dtype=int)}
        self.observation[self.PLAYER_ONE][4, 2] = self.NUM_PIECES
        self.observation[self.PLAYER_TWO][0, 2] = self.NUM_PIECES
        self.observation[self.ROLL] = self.__shells.state

        # reset player
        self.player = self.PLAYER_ONE

        return self.observation, self.player

    def step(self, action: np.ndarray, player_move=True):
        """
        handles one step given an action
        inputs: action - piece no. in order (0-self.NUM_PIECES)
        outputs: observation - dict of ndarrays representing board,
                 reward - +1 if P1 wins, -1 if P2 wins, 0 otherwise,
                 done - boolean representing the win condition

        notes:
         - does nothing if player tries to move onto own piece
        """

        # ensure that action is valid
        assert self.action_space.contains(action)
        
        # check whether there are no legal moves for the cpu and pass
        legal_moves = [x for x in range(4) if not self.is_illegal_move(x)]
        if not legal_moves:
            if player_move:
                self.__switch_player()
                self.__shells.roll()
                self.observation[self.ROLL] = self.__shells.state
                observation, reward, done = self.step(self.opponent_policy(self), player_move=False)
                self.__switch_player()
                self.__shells.roll()
                observation[self.ROLL] = self.__shells.state
                return observation, reward, done
            else:
                return self.observation, 0, False
        
        # get position array
        pos_array = self.MOVE_PATH[self.player]

        # get piece number
        piece_number = action

        # figure out how many spaces to move
        number_moves = self.MOVES[self.observation[self.ROLL]]

        # get piece position indices
        piece_pos_indices = self.__get_piece_pos_indices()

        # get current position index
        curr_pos_idx = piece_pos_indices[piece_number]

        # if move is out of bounds, end the game and punish the player
        if self.is_illegal_move(piece_number) and player_move:
            return self.observation, -5, True

        # get current position and position to move
        curr_pos = pos_array[curr_pos_idx]
        pos_to_move = pos_array[curr_pos_idx + number_moves]

        if pos_to_move not in self.SAFE_SQUARES and self.observation[self.__other_player()][pos_to_move] > 0:
            self.observation[self.__other_player()][pos_to_move] -= 1
            self.observation[self.__other_player()][
                self.MOVE_PATH[self.__other_player()][0]
            ] += 1

        # move piece to correct spot
        self.observation[self.player][curr_pos] -= 1
        self.observation[self.player][pos_to_move] += 1

        # you must re-roll on 4 or 8
        if number_moves in {4, 8}:
            if player_move:
                self.__shells.roll()
                self.observation[self.ROLL] = self.__shells.state
                return self.observation, 0, False
            else:
                if self.observation[self.player][2, 2] == self.NUM_PIECES:
                    self.observation[self.player][curr_pos] += 1
                    self.observation[self.player][pos_to_move] -= 1
                    return self.observation, 0, False
                self.__shells.roll()
                self.observation[self.ROLL] = self.__shells.state
                return self.step(self.opponent_policy(self), player_move=False)

        # check for win condition (otherwise, return default values)
        if self.observation[self.player][2, 2] == self.NUM_PIECES:
            return (
                self.observation,
                self.__get_reward(self.player),
                True,
            )
        else:
            # if it's the player's turn, roll + do an opponent move and roll again
            if player_move:
                self.__switch_player()
                self.__shells.roll()
                self.observation[self.ROLL] = self.__shells.state
                observation, reward, done = self.step(self.opponent_policy(self), player_move=False)
                self.__switch_player()
                self.__shells.roll()
                observation[self.ROLL] = self.__shells.state
                return observation, reward, done
            # otherwise, the game isn't over so 
            else:
                return self.observation, 0, False

    def __other_player(self):
        # get other player symbol
        if self.player == self.PLAYER_ONE:
            return self.PLAYER_TWO
        else:
            return self.PLAYER_ONE
    
    def __get_piece_pos_indices(self):
        piece_indices = []
        pos_array = self.MOVE_PATH[self.player]

        # get piece positions of current player in order of how far along they are
        # (from nearest to farthest)
        for pos_idx, _ in enumerate(pos_array):
            pieces_here = self.observation[self.player][pos_array[pos_idx]]
            while pieces_here > 0:
                piece_indices.append(pos_idx)
                pieces_here -= 1

        return piece_indices
    
    def is_illegal_move(self, piece_number):
        # get piece position indices
        piece_pos_indices = self.__get_piece_pos_indices()

        # get current position index
        curr_pos_idx = piece_pos_indices[piece_number]

        # compute number of moves
        number_moves = self.MOVES[self.observation[self.ROLL]]

        # if move is out of bounds, end the game and punish the player
        if curr_pos_idx + number_moves >= 25:
            return True
        else:
            pos_to_move = self.MOVE_PATH[self.player][curr_pos_idx + number_moves]
            return pos_to_move not in self.SAFE_SQUARES and self.observation[self.player][pos_to_move] > 0

    @staticmethod
    def __get_reward(player):
        # map player symbol to their reward, if they win
        if player == AshtaChammaEnv.PLAYER_ONE:
            return 1
        elif player == AshtaChammaEnv.PLAYER_TWO:
            return -1

    def render(self):
        """
        renders the board onto the console
        """
        observation = self.observation

        # print top line
        print("-" * 31)
        state = np.vstack((observation[self.PLAYER_ONE], observation[self.PLAYER_TWO]))

        for row1, row2 in zip(state, state[int(len(state)/2):]):
            # print bar before each square
            print("| ", end="")

            # print player one piece count
            for square_1 in row1:
                if square_1 != 0:
                    print(self.PLAYER_ONE, square_1, "| ", end="")
                else:
                    print("    | ", end="")

            # separate player one and player two piece count
            print()

            # print bar after each square
            print("| ", end="")

            # print player two piece count
            for square_2 in row2:
                if square_2 != 0:
                    print(self.PLAYER_TWO, square_2, "| ", end="")
                else:
                    print("    | ", end="")
            
            # print bottom line after each row
            print()
            print("-------------------------------")
    
    def __switch_player(self):
        self.player = self.__other_player()
    
    @staticmethod
    def to_array(state):
        return np.vstack((state[AshtaChammaEnv.PLAYER_ONE], state[AshtaChammaEnv.PLAYER_TWO]))
