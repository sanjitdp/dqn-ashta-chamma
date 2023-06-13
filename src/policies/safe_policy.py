from random import choice


def safe_policy(env):
    """
    always moves to a safe square, when possible
    """

    legal_moves = [x for x in range(4) if not env.is_illegal_move(x)]

    # try to move onto a safe square
    safe_moves = [x for x in legal_moves if env.moving_to_safe(x)]
    if safe_moves:
        return choice(safe_moves)

    # otherwise, try to make any legal move
    elif legal_moves:
        return choice(legal_moves)
    else:
        return 0
