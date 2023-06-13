from random import choice


def offense_policy(env):
    """
    always captures a piece, whenever possible
    """

    legal_moves = [x for x in range(4) if not env.is_illegal_move(x)]

    # try to capture a piece
    capture_moves = [x for x in legal_moves if env.is_capture_move(x)]
    if capture_moves:
        return choice(capture_moves)

    # otherwise, try to make any legal move
    elif legal_moves:
        return choice(legal_moves)
    else:
        return 0
