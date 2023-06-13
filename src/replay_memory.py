from collections import namedtuple, deque
import random

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """
    keeps track of memory for past games
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        save a transition
        """

        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        sample from the memory
        """

        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
