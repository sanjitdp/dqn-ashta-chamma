import numpy as np

class Shells:
    def __init__(self):
        self.__state = 4
    
    def roll_shells(self):
        # use empirical probabilities computed in section 4.1 of this paper:
        # https://iaeme.com/MasterAdmin/Journal_uploads/IJCET/VOLUME_10_ISSUE_1/IJCET_10_01_019.pdf
        empirical_probabilities = [0.243, 0.381, 0.236, 0.074, 0.066]
        self.__state = np.random.choice(np.arange(5), size=(1,), p=empirical_probabilities)
    
    @property
    def state(self):
        return self.__state
