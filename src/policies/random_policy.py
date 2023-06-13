from random import choice


def random_policy(env):
    """
    makes a random legal move
    """

    legal_moves = [x for x in range(4) if not env.is_illegal_move(x)]

    # try to make any legal move
    return choice(legal_moves) if legal_moves else 0
