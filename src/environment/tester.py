from env import AshtaChammaEnv
from policies.random_policy import random_policy

if __name__ == "__main__":

    env = AshtaChammaEnv(opponent_policy=random_policy)

    # test capture
    print(env.reset())
    print(env.step(1))
    env.render()
    print(env.step(0))
    env.render()
