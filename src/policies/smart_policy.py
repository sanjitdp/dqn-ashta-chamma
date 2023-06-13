from random import choice


def smart_policy(env):
    """
    captures a piece, tries to move to a safe square, and makes any legal move (in that order)
    """

    legal_moves = [x for x in range(4) if not env.is_illegal_move(x)]

    # try to capture a piece
    capture_moves = [x for x in legal_moves if env.is_capture_move(x)]
    if capture_moves:
        return choice(capture_moves)

    # if we cannot capture a piece, try to move to a safe square
    safe_moves = [x for x in legal_moves if env.moving_to_safe(x)]
    if safe_moves:
        return choice(safe_moves)

    # otherwise, try to make any legal move
    elif legal_moves:
        return choice(legal_moves)
    else:
        return 0
