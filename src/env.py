import numpy as np
import gymnasium as gym
from util.shells import Shells
from gymnasium import spaces
from copy import deepcopy


class AshtaChammaEnv(gym.Env):
    """
    subclasses the gymnasium environment to create the game environment
    """

    # set relevant constants
    NUM_PIECES = 4
    PLAYER_ONE = "O"
    PLAYER_TWO = "X"
    ROLL = "ROLL"
    PLAYER_ONE_CAPTURE = "OC"
    PLAYER_TWO_CAPTURE = "XC"
    REMEMBERED_STATE = "RS"
    MULTI_TURN = "MT"

    # save default remembered state
    DEFAULT_REMEMBERED = {
        "X": np.zeros((5, 5), dtype=int),
        "O": np.zeros((5, 5), dtype=int),
        "OC": 1,
        "XC": 1,
        "ROLL": 1,
    }

    REMEMBERED_ZERO_ARRAY = np.zeros(54)

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
        ],
    }

    # array containing safe squares
    SAFE_SQUARES = [
        (0, 2),
        (1, 1),
        (1, 3),
        (2, 0),
        (2, 2),
        (2, 4),
        (3, 1),
        (3, 3),
        (4, 2),
    ]

    # distances of possible self.MOVES
    MOVES = [1, 2, 3, 4, 8]

    def __init__(self, opponent_policy):
        # observation space consists of two BOARD_SIZE x BOARD_SIZE arrays, each between 0 and self.NUM_PIECES
        # (one corresponding to self.PLAYER_ONE, one corresponding to self.PLAYER_TWO),
        # the current roll, and whether player one or player two has captured yet
        self.observation_space = spaces.Dict(
            {
                self.PLAYER_ONE: spaces.Box(low=0, high=self.NUM_PIECES, shape=(5, 5)),
                self.PLAYER_TWO: spaces.Box(low=0, high=self.NUM_PIECES, shape=(5, 5)),
                self.ROLL: spaces.Discrete(len(self.MOVES)),
                self.PLAYER_ONE_CAPTURE: spaces.Discrete(2),
                self.PLAYER_TWO_CAPTURE: spaces.Discrete(2),
                self.REMEMBERED_STATE: spaces.Dict(
                    {
                        self.PLAYER_ONE: spaces.Box(
                            low=0, high=self.NUM_PIECES, shape=(5, 5)
                        ),
                        self.PLAYER_TWO: spaces.Box(
                            low=0, high=self.NUM_PIECES, shape=(5, 5)
                        ),
                        self.ROLL: spaces.Discrete(len(self.MOVES)),
                        self.PLAYER_ONE_CAPTURE: spaces.Discrete(2),
                        self.PLAYER_TWO_CAPTURE: spaces.Discrete(2),
                    }
                ),
                self.MULTI_TURN: spaces.Discrete(2),
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
        self.observation = {
            self.PLAYER_ONE: np.zeros((5, 5), dtype=int),
            self.PLAYER_TWO: np.zeros((5, 5), dtype=int),
        }
        self.observation[self.PLAYER_ONE][4, 2] = self.NUM_PIECES
        self.observation[self.PLAYER_TWO][0, 2] = self.NUM_PIECES
        self.observation[self.PLAYER_ONE_CAPTURE] = 1
        self.observation[self.PLAYER_TWO_CAPTURE] = 1
        self.observation[self.ROLL] = self.__shells.state
        self.__reset_remembered()
        self.observation[self.MULTI_TURN] = 1

        # reset player
        self.player = self.PLAYER_ONE

        return self.observation, self.player

    def __reset_remembered(self):
        self.observation[self.MULTI_TURN] = 1
        self.observation[self.REMEMBERED_STATE] = self.DEFAULT_REMEMBERED

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

        # check whether neither player has any legal moves left,
        # and draw the game if necessary
        if (
            self.observation[self.PLAYER_ONE_CAPTURE] == 1
            and self.observation[self.PLAYER_TWO_CAPTURE] == 1
        ):
            if all(x > 8 for x in self.__get_piece_pos_indices()):
                self.__switch_player()
                if all(x > 8 for x in self.__get_piece_pos_indices()):
                    return self.observation, 0, True
                else:
                    self.__switch_player()

        # check whether there are no legal moves
        legal_moves = [x for x in range(4) if not self.is_illegal_move(x)]
        if not legal_moves:
            # check if it was a multi-turn and then reset the board to the remembered state
            if self.observation[self.MULTI_TURN] == 2:
                self.observation = deepcopy(self.observation[self.REMEMBERED_STATE])
                self.__reset_remembered()
                self.__switch_player()
                self.__shells.roll()
                self.observation
                return self.observation, 0, False

            # if it's the player's turn, allow cpu to complete a move
            if player_move:
                self.__switch_player()
                self.__shells.roll()
                self.observation[self.ROLL] = self.__shells.state
                self.__reset_remembered()
                observation, reward, done = self.step(
                    self.opponent_policy(self), player_move=False
                )
                self.__switch_player()
                self.__shells.roll()
                observation[self.ROLL] = self.__shells.state
                self.__reset_remembered()
                return observation, reward, done

            # if it's the cpu's turn, relinquish control to player
            else:
                self.__reset_remembered()
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

        # capture a piece, if applicable
        capture = False
        if (
            pos_to_move not in self.SAFE_SQUARES
            and self.observation[self.__other_player()][pos_to_move] > 0
        ):
            # remember current state in case we need to revert
            if self.observation[self.MULTI_TURN] == 1:
                remembered_state = deepcopy(self.observation)
                del remembered_state[self.REMEMBERED_STATE]
                del remembered_state[self.MULTI_TURN]
                self.observation[self.REMEMBERED_STATE] = remembered_state
                self.observation[self.MULTI_TURN] = 2
            # send piece back to start
            self.observation[self.__other_player()][pos_to_move] -= 1
            self.observation[self.__other_player()][
                self.MOVE_PATH[self.__other_player()][0]
            ] += 1

            # set captured flag
            match self.player:
                case self.PLAYER_ONE:
                    self.observation[self.PLAYER_ONE_CAPTURE] = 2
                case self.PLAYER_TWO:
                    self.observation[self.PLAYER_TWO_CAPTURE] = 2

            capture = True

        if number_moves in {4, 8}:
            # remember current state in case we need to revert
            if self.observation[self.MULTI_TURN] == 1:
                remembered_state = deepcopy(self.observation)
                del remembered_state[self.REMEMBERED_STATE]
                del remembered_state[self.MULTI_TURN]
                self.observation[self.REMEMBERED_STATE] = remembered_state
                self.observation[self.MULTI_TURN] = 2

        # move piece to correct spot
        self.observation[self.player][curr_pos] -= 1
        self.observation[self.player][pos_to_move] += 1

        # you must re-roll on 4 or 8 or capture
        if number_moves in {4, 8} or capture:
            if player_move:
                self.__shells.roll()
                self.observation[self.ROLL] = self.__shells.state
                return self.observation, 0, False
            else:
                self.__shells.roll()
                self.observation[self.ROLL] = self.__shells.state
                out = self.step(self.opponent_policy(self), player_move=False)
                self.__reset_remembered()
                return out

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
                self.__reset_remembered()
                self.__switch_player()
                self.__shells.roll()
                self.observation[self.ROLL] = self.__shells.state
                observation, reward, done = self.step(
                    self.opponent_policy(self), player_move=False
                )
                self.__switch_player()
                self.__shells.roll()
                observation[self.ROLL] = self.__shells.state
                return observation, reward, done
            # otherwise, the game isn't over so continue
            else:
                self.__reset_remembered()
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
        """
        checks whether a move is illegal
        """

        # get piece position indices
        piece_pos_indices = self.__get_piece_pos_indices()

        # get current position index
        curr_pos_idx = piece_pos_indices[piece_number]

        # compute number of moves
        number_moves = self.MOVES[self.observation[self.ROLL]]

        # check whether player has captured
        match self.player:
            case self.PLAYER_ONE:
                captured = self.observation[self.PLAYER_ONE_CAPTURE] == 2
            case self.PLAYER_TWO:
                captured = self.observation[self.PLAYER_TWO_CAPTURE] == 2

        # if move is out of bounds, end the game and punish the player
        idx_to_move = curr_pos_idx + number_moves
        if (not captured and idx_to_move > 12) or idx_to_move >= 25:
            return True
        else:
            pos_to_move = self.MOVE_PATH[self.player][curr_pos_idx + number_moves]
            return (
                pos_to_move not in self.SAFE_SQUARES
                and self.observation[self.player][pos_to_move] > 0
            )

    def moving_to_safe(self, piece_number):
        """
        checks whether a move will land on a safe square
        """

        # get piece position indices
        piece_pos_indices = self.__get_piece_pos_indices()

        # get current position index
        curr_pos_idx = piece_pos_indices[piece_number]

        # compute number of moves
        number_moves = self.MOVES[self.observation[self.ROLL]]

        # compute position we're moving to
        pos_to_move = self.MOVE_PATH[self.player][curr_pos_idx + number_moves]

        # check if we're moving onto a safe square
        return pos_to_move in self.SAFE_SQUARES

    def is_capture_move(self, piece_number):
        """
        checks whether a move will capture a piece
        """

        # get piece position indices
        piece_pos_indices = self.__get_piece_pos_indices()

        # get current position index
        curr_pos_idx = piece_pos_indices[piece_number]

        # compute number of moves
        number_moves = self.MOVES[self.observation[self.ROLL]]

        # compute position we're moving to
        pos_to_move = self.MOVE_PATH[self.player][curr_pos_idx + number_moves]

        # check if we're going to capture a piece
        return (
            pos_to_move not in self.SAFE_SQUARES
            and self.observation[self.__other_player()][pos_to_move] > 0
        )

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

        for row1, row2 in zip(state, state[int(len(state) / 2) :]):
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

        print("Roll:", self.MOVES[self.observation[self.ROLL]])

    def __switch_player(self):
        self.player = self.__other_player()

    @staticmethod
    def to_array(state):
        if state[AshtaChammaEnv.MULTI_TURN] == 2:
            return np.concatenate(
                (
                    state[AshtaChammaEnv.PLAYER_ONE],
                    state[AshtaChammaEnv.PLAYER_TWO],
                    state[AshtaChammaEnv.PLAYER_ONE_CAPTURE],
                    state[AshtaChammaEnv.PLAYER_TWO_CAPTURE],
                    [state[AshtaChammaEnv.ROLL]] * 10,  # emphasize importance of roll
                    state[AshtaChammaEnv.REMEMBERED_STATE][AshtaChammaEnv.PLAYER_ONE],
                    state[AshtaChammaEnv.REMEMBERED_STATE][AshtaChammaEnv.PLAYER_TWO],
                    state[AshtaChammaEnv.REMEMBERED_STATE][
                        AshtaChammaEnv.PLAYER_ONE_CAPTURE
                    ],
                    state[AshtaChammaEnv.REMEMBERED_STATE][
                        AshtaChammaEnv.PLAYER_TWO_CAPTURE
                    ],
                    state[AshtaChammaEnv.REMEMBERED_STATE][AshtaChammaEnv.ROLL],
                    state[AshtaChammaEnv.MULTI_TURN],
                ),
                axis=None,
            )
        else:
            return np.concatenate(
                (
                    state[AshtaChammaEnv.PLAYER_ONE],
                    state[AshtaChammaEnv.PLAYER_TWO],
                    state[AshtaChammaEnv.PLAYER_ONE_CAPTURE],
                    state[AshtaChammaEnv.PLAYER_TWO_CAPTURE],
                    [state[AshtaChammaEnv.ROLL]] * 10,  # emphasize importance of roll
                    AshtaChammaEnv.REMEMBERED_ZERO_ARRAY,
                ),
                axis=None,
            )
