import numpy as np


def roll_shells():
    # use empirical probabilities computed in section 4.1 of this paper:
    # https://iaeme.com/MasterAdmin/Journal_uploads/IJCET/VOLUME_10_ISSUE_1/IJCET_10_01_019.pdf
    rolls = [1, 2, 3, 4, 8]
    empirical_probabilities = [0.243, 0.381, 0.236, 0.074, 0.066]
    return np.random.choice(rolls, size=(1,), p=empirical_probabilities)
