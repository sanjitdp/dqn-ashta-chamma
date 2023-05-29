from random import choice

def random_policy(env):
    return choice([x for x in range(4) if not env.is_illegal_move(x)])

